; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff -mcpu=pwr7 < %s | FileCheck %s

; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff -mcpu=pwr7 --filetype=obj -o %t.o < %s
; RUN: llvm-objdump --syms %t.o | FileCheck %s --check-prefix=OBJ

%struct.anon = type {}

@a = internal constant %struct.anon zeroinitializer, align 1
@b = internal constant [6 x i8] c"hello\00", align 1

; CHECK:      	.csect L.._MergedGlobals[RO],2
; CHECK-NEXT: 	.lglobl	a                               # @_MergedGlobals
; CHECK-NEXT: 	.lglobl	b
; CHECK-NEXT: a:
; CHECK-NEXT: b:
; CHECK-NEXT: 	.string	"hello"

; OBJ:      0000000000000000 l       .text	0000000000000006 L.._MergedGlobals
; OBJ-NEXT: 0000000000000000 l       .text (csect: L.._MergedGlobals) 	0000000000000000 a
; OBJ-NEXT: 0000000000000000 l       .text (csect: L.._MergedGlobals) 	0000000000000000 b
