/// Test that llvm-cov supports a fake gcov 4.2 format used before clang 11.

// RUN: rm -rf %t && mkdir %t && cd %t
// RUN: llvm-cov gcov test. --gcno=%S/Inputs/gcov-fake-4.2.gcno --gcda=%S/Inputs/gcov-fake-4.2.gcda | FileCheck %s
// RUN: FileCheck %s --check-prefix=C < test.cpp.gcov
// RUN: FileCheck %s --check-prefix=H < test.h.gcov

// CHECK:       File 'test.cpp'
// CHECK-NEXT:  Lines executed:84.21% of 38
// CHECK-NEXT:  Creating 'test.cpp.gcov'
// CHECK-EMPTY:
// CHECK-NEXT:  File './test.h'
// CHECK-NEXT:  Lines executed:100.00% of 1
// CHECK-NEXT:  Creating 'test.h.gcov'
// CHECK-EMPTY:

//      C:        -:    0:Source:test.cpp
// C-NEXT:        -:    0:Graph:{{.*}}gcov-fake-4.2.gcno
// C-NEXT:        -:    0:Data:{{.*}}gcov-fake-4.2.gcda
/// `Runs` is stored in GCOV_TAG_OBJECT_SUMMARY with a length of 9.
// C-NEXT:        -:    0:Runs:2
// C-NEXT:        -:    0:Programs:1
// C-NEXT:        -:    1:
// C-NEXT:        -:    2:
// C-NEXT:        -:    3:
// C-NEXT:        -:    4:
// C-NEXT:        -:    5:
// C-NEXT:        -:    6:
// C-NEXT:        -:    7:
// C-NEXT:        -:    8:
// C-NEXT:        -:    9:
// C-NEXT:8589934592:   10:

//      H:        -:    0:Source:./test.h
// H-NEXT:        -:    0:Graph:{{.*}}gcov-fake-4.2.gcno
// H-NEXT:        -:    0:Data:{{.*}}gcov-fake-4.2.gcda
// H-NEXT:        -:    0:Runs:2
// H-NEXT:        -:    0:Programs:1
// H-NEXT:        4:    1:
