// RUN: llvm-tblgen -I %p %s 2>&1 | FileCheck %s 

// CHECK: def DEF_A
#define FILE_A
include "prep-undef-included-file.inc"

// CHECK: def DEF_B
#define FILE_B
include "prep-undef-included-file.inc"
