; RUN: llc < %s -asm-verbose=false -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -fast-isel | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

%SmallStruct = type { i32 }
%OddStruct = type { i32, i8, i32 }
%AlignedStruct = type { double, double }
%BigStruct = type { double, double, double, double, double, double, double, double, double, double, double, i8, i8, i8 }

%BigArray = type { [33 x i8] }

declare void @ext_func(%SmallStruct*)
declare void @ext_byval_func(%SmallStruct* byval)
declare void @ext_byval_func_align8(%SmallStruct* byval align 8)
declare void @ext_byval_func_alignedstruct(%AlignedStruct* byval)
declare void @ext_byval_func_bigarray(%BigArray* byval)

; CHECK-LABEL: byval_arg
define void @byval_arg(%SmallStruct* %ptr) {
 ; CHECK: .param i32
 ; Subtract 16 from SP (SP is 16-byte aligned)
 ; CHECK: i32.const [[L1:.+]]=, __stack_pointer
 ; CHECK-NEXT: i32.load [[L1]]=, 0([[L1]])
 ; CHECK-NEXT: i32.const [[L2:.+]]=, 16
 ; CHECK-NEXT: i32.sub [[SP:.+]]=, [[L1]], [[L2]]
 ; Ensure SP is stored back before the call
 ; CHECK-NEXT: i32.const [[L3:.+]]=, __stack_pointer
 ; CHECK-NEXT: i32.store {{.*}}=, 0([[L3]]), [[SP]]
 ; Copy the SmallStruct argument to the stack (SP+12, original SP-4)
 ; CHECK-NEXT: i32.load $push[[L4:.+]]=, 0($0)
 ; CHECK-NEXT: i32.store {{.*}}=, 12([[SP]]), $pop[[L4]]
 ; Pass a pointer to the stack slot to the function
 ; CHECK-NEXT: i32.const [[L5:.+]]=, 12
 ; CHECK-NEXT: i32.add [[ARG:.+]]=, [[SP]], [[L5]]
 ; CHECK-NEXT: call ext_byval_func@FUNCTION, [[L5]]
 call void @ext_byval_func(%SmallStruct* byval %ptr)
 ; Restore the stack
 ; CHECK-NEXT: i32.const [[L6:.+]]=, 16
 ; CHECK-NEXT: i32.add [[SP]]=, [[SP]], [[L6]]
 ; CHECK-NEXT: i32.const [[L7:.+]]=, __stack_pointer
 ; CHECK-NEXT: i32.store {{.*}}=, 0([[L7]]), [[SP]]
 ; CHECK-NEXT: return
 ret void
}

; CHECK-LABEL: byval_arg_align8
define void @byval_arg_align8(%SmallStruct* %ptr) {
 ; CHECK: .param i32
 ; Don't check the entire SP sequence, just enough to get the alignment.
 ; CHECK: i32.const [[L2:.+]]=, 16
 ; CHECK-NEXT: i32.sub [[SP:.+]]=, {{.+}}, [[L2]]
 ; Copy the SmallStruct argument to the stack (SP+8, original SP-8)
 ; CHECK: i32.load $push[[L4:.+]]=, 0($0):p2align=3
 ; CHECK-NEXT: i32.store {{.*}}=, 8([[SP]]):p2align=3, $pop[[L4]]
 ; Pass a pointer to the stack slot to the function
 ; CHECK-NEXT: i32.const [[L5:.+]]=, 8
 ; CHECK-NEXT: i32.add [[ARG:.+]]=, [[SP]], [[L5]]
 ; CHECK-NEXT: call ext_byval_func_align8@FUNCTION, [[L5]]
 call void @ext_byval_func_align8(%SmallStruct* byval align 8 %ptr)
 ret void
}

; CHECK-LABEL: byval_arg_double
define void @byval_arg_double(%AlignedStruct* %ptr) {
 ; CHECK: .param i32
 ; Subtract 16 from SP (SP is 16-byte aligned)
 ; CHECK: i32.const [[L2:.+]]=, 16
 ; CHECK-NEXT: i32.sub [[SP:.+]]=, {{.+}}, [[L2]]
 ; Copy the AlignedStruct argument to the stack (SP+0, original SP-16)
 ; Just check the last load/store pair of the memcpy
 ; CHECK: i64.load $push[[L4:.+]]=, 0($0)
 ; CHECK-NEXT: i64.store {{.*}}=, 0([[SP]]), $pop[[L4]]
 ; Pass a pointer to the stack slot to the function
 ; CHECK-NEXT: call ext_byval_func_alignedstruct@FUNCTION, [[SP]]
 tail call void @ext_byval_func_alignedstruct(%AlignedStruct* byval %ptr)
 ret void
}

; CHECK-LABEL: byval_arg_big
define void @byval_arg_big(%BigArray* %ptr) {
 ; CHECK: .param i32
 ; Subtract 48 from SP (SP is 16-byte aligned)
 ; CHECK: i32.const [[L2:.+]]=, 48
 ; CHECK-NEXT: i32.sub [[SP:.+]]=, {{.+}}, [[L2]]
 ; Copy the AlignedStruct argument to the stack (SP+12, original SP-36)
 ; CHECK: i64.load $push[[L4:.+]]=, 0($0):p2align=0
 ; CHECK: i64.store {{.*}}=, 12([[SP]]):p2align=2, $pop[[L4]]
 ; Pass a pointer to the stack slot to the function
 ; CHECK-NEXT: i32.const [[L5:.+]]=, 12
 ; CHECK-NEXT: i32.add [[ARG:.+]]=, [[SP]], [[L5]]
 ; CHECK-NEXT: call ext_byval_func_bigarray@FUNCTION, [[ARG]]
 call void @ext_byval_func_bigarray(%BigArray* byval %ptr)
 ret void
}

; CHECK-LABEL: byval_param
define void @byval_param(%SmallStruct* byval align 32 %ptr) {
 ; CHECK: .param i32
 ; %ptr is just a pointer to a struct, so pass it directly through
 ; CHECK: call ext_func@FUNCTION, $0
 call void @ext_func(%SmallStruct* %ptr)
 ret void
}
