; RUN: llc < %s | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv8-arm-none-eabi"

; Check the loads happen after the stores (note: directly returning 0 is also valid)
; CHECK-LABEL: somesortofhash
; CHECK-NOT: ldr
; CHECK: str

define i64 @somesortofhash() {
entry:
  %helper = alloca i8, i32 64, align 8
  %helper.0.4x32 = bitcast i8* %helper to <4 x i32>*
  %helper.20 = getelementptr inbounds i8, i8* %helper, i32 20
  %helper.24 = getelementptr inbounds i8, i8* %helper, i32 24
  store <4 x i32> zeroinitializer, <4 x i32>* %helper.0.4x32, align 8
  %helper.20.32 = bitcast i8* %helper.20 to i32*
  %helper.24.32 = bitcast i8* %helper.24 to i32*
  store i32 0, i32* %helper.20.32
  store i32 0, i32* %helper.24.32, align 8
  %helper.20.64 = bitcast i8* %helper.20 to i64*
  %load.helper.20.64 = load i64, i64* %helper.20.64, align 4
  ret i64 %load.helper.20.64
}
