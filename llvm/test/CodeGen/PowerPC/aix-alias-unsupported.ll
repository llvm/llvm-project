; RUN: not --crash llc < %s -mtriple powerpc-ibm-aix-xcoff 2>&1 | FileCheck %s
; RUN: not --crash llc < %s -mtriple powerpc64-ibm-aix-xcoff 2>&1 | FileCheck %s
; CHECK: ERROR: alias without a base object is not yet supported on AIX

@bar = global i32 42
@test = alias i32, inttoptr(i32 42 to i32*)
