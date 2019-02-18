// RUN: llvm-mc -triple=powerpc64-pc-linux %s -o - | FileCheck %s

NUMBER = 0x6ffffff9

        .text
.globl  main
main:
	cmpwi   8,(NUMBER)@l
	cmpwi   8,-(NUMBER)@l
	cmpwi   8,(-NUMBER)@l
	cmpwi   8,NUMBER@l
	cmpwi   8,-NUMBER@l


// CHECK: main:
// CHECK: cmpwi 8, -7
// CHECK: cmpwi 8, 7
// CHECK: cmpwi 8, 7
// CHECK: cmpwi 8, -7
// CHECK: cmpwi 8, 7

