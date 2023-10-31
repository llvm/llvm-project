# 'ArmSME' Dialect

Basic dialect to target Arm SME architectures This dialect contains the
definitions necessary to target Arm SME scalable matrix operations.

## References
* https://developer.arm.com/documentation/ddi0616
* https://developer.arm.com/documentation/ddi0602/2023-03/SME-Instructions

## Operations

[include "Dialects/ArmSMEOps.md"]

## Operations for LLVM IR Intrinsics

[include "Dialects/ArmSMEIntrinsicOps.md"]
