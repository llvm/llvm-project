// RUN: not llvm-tblgen %s -DMACRO=1 2>&1 | FileCheck %s --check-prefix=CHECK-TEST-1
// RUN: not llvm-tblgen %s -D0MAC 2>&1 | FileCheck %s --check-prefix=CHECK-TEST-2
// RUN: not llvm-tblgen %s -D_MAC# 2>&1 | FileCheck %s --check-prefix=CHECK-TEST-3
// RUN: not llvm-tblgen %s -D 2>&1 | FileCheck %s --check-prefix=CHECK-TEST-4

// CHECK-TEST-1: error: invalid macro name `MACRO=1` specified on command line
// CHECK-TEST-2: error: invalid macro name `0MAC` specified on command line
// CHECK-TEST-3: error: invalid macro name `_MAC#` specified on command line
// CHECK-TEST-4: for the -D option: requires a value!
