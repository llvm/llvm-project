; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -disable-wasm-explicit-locals -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -disable-wasm-explicit-locals -verify-machineinstrs -fast-isel | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

%SmallStruct = type { i32 }
%OddStruct = type { i32, i8, i32 }
%AlignedStruct = type { double, double }
%BigStruct = type { double, double, double, double, double, double, double, double, double, double, double, i8, i8, i8 }
%EmptyStruct = type { }

%BigArray = type { [33 x i8] }

declare void @ext_func(%SmallStruct*)
declare void @ext_func_empty(%EmptyStruct* byval)
declare void @ext_byval_func(%SmallStruct* byval)
declare void @ext_byval_func_align8(%SmallStruct* byval align 8)
declare void @ext_byval_func_alignedstruct(%AlignedStruct* byval)
declare void @ext_byval_func_bigarray(%BigArray* byval)
declare void @ext_byval_func_empty(%EmptyStruct* byval)

; CHECK-LABEL: byval_arg
define void @byval_arg(%SmallStruct* %ptr) {
 ; CHECK: .param i32
 ; Subtract 16 from SP (SP is 16-byte aligned)
 ; CHECK-NEXT: get_global $push[[L2:.+]]=, __stack_pointer
 ; CHECK-NEXT: i32.const $push[[L3:.+]]=, 16
 ; CHECK-NEXT: i32.sub $push[[L11:.+]]=, $pop[[L2]], $pop[[L3]]
 ; Ensure SP is stored back before the call
 ; CHECK-NEXT: tee_local $push[[L10:.+]]=, $[[SP:.+]]=, $pop[[L11]]{{$}}
 ; CHECK-NEXT: set_global __stack_pointer, $pop[[L10]]{{$}}
 ; Copy the SmallStruct argument to the stack (SP+12, original SP-4)
 ; CHECK-NEXT: i32.load $push[[L0:.+]]=, 0($0)
 ; CHECK-NEXT: i32.store 12($[[SP]]), $pop[[L0]]
 ; Pass a pointer to the stack slot to the function
 ; CHECK-NEXT: i32.const $push[[L5:.+]]=, 12{{$}}
 ; CHECK-NEXT: i32.add $push[[ARG:.+]]=, $[[SP]], $pop[[L5]]{{$}}
 ; CHECK-NEXT: call ext_byval_func@FUNCTION, $pop[[ARG]]{{$}}
 call void @ext_byval_func(%SmallStruct* byval %ptr)
 ; Restore the stack
 ; CHECK-NEXT: i32.const $push[[L6:.+]]=, 16
 ; CHECK-NEXT: i32.add $push[[L8:.+]]=, $[[SP]], $pop[[L6]]
 ; CHECK-NEXT: set_global __stack_pointer, $pop[[L8]]
 ; CHECK-NEXT: return
 ret void
}

; CHECK-LABEL: byval_arg_align8
define void @byval_arg_align8(%SmallStruct* %ptr) {
 ; CHECK: .param i32
 ; Don't check the entire SP sequence, just enough to get the alignment.
 ; CHECK: i32.const $push[[L1:.+]]=, 16
 ; CHECK-NEXT: i32.sub $push[[L11:.+]]=, {{.+}}, $pop[[L1]]
 ; CHECK-NEXT: tee_local $push[[L10:.+]]=, $[[SP:.+]]=, $pop[[L11]]{{$}}
 ; CHECK-NEXT: set_global __stack_pointer, $pop[[L10]]{{$}}
 ; Copy the SmallStruct argument to the stack (SP+8, original SP-8)
 ; CHECK-NEXT: i32.load $push[[L0:.+]]=, 0($0){{$}}
 ; CHECK-NEXT: i32.store 8($[[SP]]), $pop[[L0]]{{$}}
 ; Pass a pointer to the stack slot to the function
 ; CHECK-NEXT: i32.const $push[[L5:.+]]=, 8{{$}}
 ; CHECK-NEXT: i32.add $push[[ARG:.+]]=, $[[SP]], $pop[[L5]]{{$}}
 ; CHECK-NEXT: call ext_byval_func_align8@FUNCTION, $pop[[ARG]]{{$}}
 call void @ext_byval_func_align8(%SmallStruct* byval align 8 %ptr)
 ret void
}

; CHECK-LABEL: byval_arg_double
define void @byval_arg_double(%AlignedStruct* %ptr) {
 ; CHECK: .param i32
 ; Subtract 16 from SP (SP is 16-byte aligned)
 ; CHECK: i32.const $push[[L1:.+]]=, 16
 ; CHECK-NEXT: i32.sub $push[[L14:.+]]=, {{.+}}, $pop[[L1]]
 ; CHECK-NEXT: tee_local $push[[L13:.+]]=, $[[SP:.+]]=, $pop[[L14]]
 ; CHECK-NEXT: set_global __stack_pointer, $pop[[L13]]
 ; Copy the AlignedStruct argument to the stack (SP+0, original SP-16)
 ; Just check the last load/store pair of the memcpy
 ; CHECK: i64.load $push[[L4:.+]]=, 0($0)
 ; CHECK-NEXT: i64.store 0($[[SP]]), $pop[[L4]]
 ; Pass a pointer to the stack slot to the function
 ; CHECK-NEXT: call ext_byval_func_alignedstruct@FUNCTION, $[[SP]]
 tail call void @ext_byval_func_alignedstruct(%AlignedStruct* byval %ptr)
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

; CHECK-LABEL: byval_empty_caller
define void @byval_empty_caller(%EmptyStruct* %ptr) {
 ; CHECK: .param i32
 ; CHECK: call ext_byval_func_empty@FUNCTION, $0
 call void @ext_byval_func_empty(%EmptyStruct* byval %ptr)
 ret void
}

; CHECK-LABEL: byval_empty_callee
define void @byval_empty_callee(%EmptyStruct* byval %ptr) {
 ; CHECK: .param i32
 ; CHECK: call ext_func_empty@FUNCTION, $0
 call void @ext_func_empty(%EmptyStruct* %ptr)
 ret void
}

; Call memcpy for "big" byvals.
; CHECK-LABEL: big_byval:
; CHECK:      get_global $push[[L2:.+]]=, __stack_pointer{{$}}
; CHECK-NEXT: i32.const $push[[L3:.+]]=, 131072
; CHECK-NEXT: i32.sub $push[[L11:.+]]=, $pop[[L2]], $pop[[L3]]
; CHECK-NEXT: tee_local $push[[L10:.+]]=, $[[SP:.+]]=, $pop[[L11]]{{$}}
; CHECK-NEXT: set_global __stack_pointer, $pop[[L10]]{{$}}
; CHECK-NEXT: i32.const $push[[L0:.+]]=, 131072
; CHECK-NEXT: i32.call       $push[[L11:.+]]=, memcpy@FUNCTION, $[[SP]], ${{.+}}, $pop{{.+}}
; CHECK-NEXT: tee_local      $push[[L9:.+]]=, $[[SP:.+]]=, $pop[[L11]]{{$}}
; CHECK-NEXT: call           big_byval_callee@FUNCTION,
%big = type [131072 x i8]
declare void @big_byval_callee(%big* byval align 1)
define void @big_byval(%big* byval align 1 %x) {
  call void @big_byval_callee(%big* byval align 1 %x)
  ret void
}
