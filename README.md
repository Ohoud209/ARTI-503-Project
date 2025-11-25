Title: Beyond Words: Parallelizing Text Cleaning and Analysis for Large Literary Collections

Parallel Processing: Race Condition vs. Fix.
This project demonstrates the way of how common data can be safely managed in case parallel computing is used.
Race Conditions.py (Race Condition): Multiple workers attempt to access a single shared resource simultaneously, and this corrupts the data and the output becomes unreliable.
Fixed code.py (Isolation/Fix): Every worker is assigned different portion of the data and produces individual output only. These individual results are then safely joined together by the main process to produce a correct result.
