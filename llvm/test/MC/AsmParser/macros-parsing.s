// RUN: not llvm-mc -triple x86_64-apple-darwin10 %s 2> %t.err
// RUN: FileCheck --check-prefix=CHECK-ERRORS %s < %t.err

.macro .test0
.endmacro

.macros_off
// CHECK-ERRORS: 9:1: error: unknown directive
.test0
.macros_on

.test0

// CHECK-ERRORS: macro '.test0' is already defined
.macro .test0
.endmacro

// CHECK-ERRORS: unexpected '.endmacro' in file
.endmacro

// CHECK-ERRORS: no matching '.endmacro' in definition
.macro dummy

