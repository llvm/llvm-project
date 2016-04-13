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
