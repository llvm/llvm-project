; RUN: opt < %s -passes=bounds-checking -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32-S128"

%struct.s2_packed = type <{ i64, i32, i32, i32, i16, i8 }>

; CHECK-LABEL: @f
; CHECK-NOT: trap
define i16 @f() {
entry:
  %packed1 = alloca %struct.s2_packed, align 8
  %gep = getelementptr inbounds %struct.s2_packed, %struct.s2_packed* %packed1, i32 0, i32 4
  %ptr = bitcast i16* %gep to i32*
  %val = load i32, i32* %ptr, align 4
  %valt = trunc i32 %val to i16
  ret i16 %valt
}

; CHECK-LABEL: @f
; CHECK: call void @llvm.trap()
define i16 @f2() {
entry:
  %packed1 = alloca %struct.s2_packed, align 8
  %gep = getelementptr inbounds %struct.s2_packed, %struct.s2_packed* %packed1, i32 0, i32 4
  %ptr = bitcast i16* %gep to i48*
  %val = load i48, i48* %ptr, align 4
  %valt = trunc i48 %val to i16
  ret i16 %valt
}
