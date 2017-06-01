; RUN: opt -consthoist -S < %s | FileCheck %s
target triple = "thumbv6m-none-eabi"

; Allocas in the entry block get handled (for free) by
; prologue/epilogue. Elsewhere they're fair game though.
define void @avoid_allocas() {
; CHECK-LABEL: @avoid_allocas
; CHECK: %addr1 = alloca i8, i32 1000
; CHECK: %addr2 = alloca i8, i32 1020

  %addr1 = alloca i8, i32 1000
  %addr2 = alloca i8, i32 1020
  br label %elsewhere

elsewhere:
; CHECK: [[BASE:%.*]] = bitcast i32 1000 to i32
; CHECK: alloca i8, i32 [[BASE]]
; CHECK: [[NEXT:%.*]] = add i32 [[BASE]], 20
; CHECK: alloca i8, i32 [[NEXT]]

  %addr3 = alloca i8, i32 1000
  %addr4 = alloca i8, i32 1020

  ret void
}

; The case values of switch instructions are required to be constants.
define void @avoid_switch(i32 %in) {
; CHECK-LABEL: @avoid_switch
; CHECK:   switch i32 %in, label %default [
; CHECK:       i32 1000, label %bb1
; CHECK:       i32 1020, label %bb2
; CHECK:   ]

  switch i32 %in, label %default
      [ i32 1000, label %bb1
        i32 1020, label %bb2 ]

bb1:
  ret void

bb2:
  ret void

default:
  ret void
}

; We don't want to convert constant divides because the benefit from converting
; them to a mul in the backend is larget than constant materialization savings.
define void @signed_const_division(i32 %in1, i32 %in2, i32* %addr) {
; CHECK-LABEL: @signed_const_division
; CHECK: %res1 = sdiv i32 %l1, 1000000000
; CHECK: %res2 = srem i32 %l2, 1000000000
entry:
  br label %loop

loop:
  %l1 = phi i32 [%res1, %loop], [%in1, %entry]
  %l2 = phi i32 [%res2, %loop], [%in2, %entry]
  %res1 = sdiv i32 %l1, 1000000000
  store volatile i32 %res1, i32* %addr
  %res2 = srem i32 %l2, 1000000000
  store volatile i32 %res2, i32* %addr
  %again = icmp eq i32 %res1, %res2
  br i1 %again, label %loop, label %end

end:
  ret void
}

define void @unsigned_const_division(i32 %in1, i32 %in2, i32* %addr) {
; CHECK-LABEL: @unsigned_const_division
; CHECK: %res1 = udiv i32 %l1, 1000000000
; CHECK: %res2 = urem i32 %l2, 1000000000

entry:
  br label %loop

loop:
  %l1 = phi i32 [%res1, %loop], [%in1, %entry]
  %l2 = phi i32 [%res2, %loop], [%in2, %entry]
  %res1 = udiv i32 %l1, 1000000000
  store volatile i32 %res1, i32* %addr
  %res2 = urem i32 %l2, 1000000000
  store volatile i32 %res2, i32* %addr
  %again = icmp eq i32 %res1, %res2
  br i1 %again, label %loop, label %end

end:
  ret void
}

;PR 28282: even when data type is larger than 64-bit, the bit width of the
;constant operand could be smaller than 64-bit. In this case, there is no
;benefit to hoist the constant.
define i32 @struct_type_test(i96 %a0, i96 %a1) {
;CHECK-LABEL: @struct_type_test
entry:
;CHECK-NOT: %const = bitcast i96 32 to i96
;CHECK: lshr0 = lshr i96 %a0, 32
  %lshr0 = lshr i96 %a0, 32
  %cast0 = trunc i96 %lshr0 to i32
;CHECK: lshr1 = lshr i96 %a1, 32
  %lshr1 = lshr i96 %a1, 32
  %cast1 = trunc i96 %lshr1 to i32
  %ret = add i32 %cast0, %cast1
  ret i32 %ret
}
