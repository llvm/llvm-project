; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck %s

; CHECK-LABEL:	.section	__LLVM_STACKMAPS,__llvm_stackmaps
; CHECK-NEXT: __LLVM_StackMaps:
; version
; CHECK-NEXT: 	.byte	1
; reserved
; CHECK-NEXT: 	.byte	0
; reserved
; CHECK-NEXT: 	.short	0
; # functions
; CHECK-NEXT: 	.long	2
; # constants
; CHECK-NEXT: 	.long	2
; # records
; CHECK-NEXT: 	.long	2
; function address & stack size
; CHECK-NEXT: 	.quad	_foo
; CHECK-NEXT: 	.quad	8
; function address & stack size
; CHECK-NEXT: 	.quad	_bar
; CHECK-NEXT: 	.quad	8

; Constants Array:
; CHECK-NEXT: 	.quad	9223372036854775807
; CHECK-NEXT: 	.quad	-9223372036854775808

; Patchpoint ID
; CHECK-NEXT: 	.quad	0
; Instruction offset
; CHECK-NEXT: 	.long	L{{.*}}-_foo
; reserved
; CHECK-NEXT: 	.short	0
; # locations
; CHECK-NEXT: 	.short	1
; ConstantIndex
; CHECK-NEXT: 	.byte	5
; reserved
; CHECK-NEXT: 	.byte	8
; Dwarf RegNum
; CHECK-NEXT: 	.short	0
; Offset
; CHECK-NEXT: 	.long	0
; padding
; CHECK-NEXT: 	.short	0
; NumLiveOuts
; CHECK-NEXT: 	.short	0

; CHECK-NEXT: 	.p2align	3

declare void @llvm.experimental.stackmap(i64, i32, ...)

define void @foo() {
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64 0, i32 0, i64 9223372036854775807)
  ret void
}

; Patchpoint ID
; CHECK-NEXT: 	.quad	0
; Instruction Offset
; CHECK-NEXT: 	.long	L{{.*}}-_bar
; reserved
; CHECK-NEXT: 	.short	0
; # locations
; CHECK-NEXT: 	.short	1
; ConstantIndex
; CHECK-NEXT: 	.byte	5
; reserved
; CHECK-NEXT: 	.byte	8
; Dwarf RegNum
; CHECK-NEXT: 	.short	0
; Offset
; CHECK-NEXT: 	.long	1
; padding
; CHECK-NEXT: 	.short	0
; NumLiveOuts
; CHECK-NEXT: 	.short	0


define void @bar() {
  tail call void (i64, i32, ...) @llvm.experimental.stackmap(i64 0, i32 0, i64 -9223372036854775808)
  ret void
}
