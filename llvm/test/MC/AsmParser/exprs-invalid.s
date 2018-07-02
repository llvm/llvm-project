// RUN: not llvm-mc -triple x86_64-apple-darwin10 %s 2> %t.err | FileCheck %s
// RUN: FileCheck --check-prefix=CHECK-ERRORS %s < %t.err
// CHECK: 	.section	__TEXT,__text,regular,pure_instructions
// CHECK-ERRORS: [[@LINE+1]]:10: error: invalid octal number in '.long' directive
.long 80+08

// CHECK-ERRORS: [[@LINE+1]]:10: error: invalid hexadecimal number in '.long' directive
.long 80+0xzz

// CHECK-ERRORS: [[@LINE+1]]:7: error: out of range literal value in '.byte' directive
.byte 256

// CHECK-ERRORS: [[@LINE+1]]:7: error: out of range literal value in '.long' directive
.long 4e71cf69 // double floating point constant due to missing "0x"

// CHECK-ERRORS: [[@LINE+1]]:7:  error: out of range literal value in '.word' directive
.word 0xfffffffff
