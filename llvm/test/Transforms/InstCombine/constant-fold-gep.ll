; RUN: opt < %s -instcombine -S | FileCheck %s
target datalayout = "E-p:64:64:64-p1:16:16:16-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"

; Constant folding should fix notionally out-of-bounds indices
; and add inbounds keywords.

%struct.X = type { [3 x i32], [3 x i32] }

@Y = internal global [3 x %struct.X] zeroinitializer

define void @frob() {
; CHECK: store i32 1, i32* getelementptr inbounds ([3 x %struct.X], [3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 0), align 16
  store i32 1, i32* getelementptr ([3 x %struct.X], [3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 0), align 4
; CHECK: store i32 1, i32* getelementptr inbounds ([3 x %struct.X], [3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 1), align 4
  store i32 1, i32* getelementptr ([3 x %struct.X], [3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 1), align 4
; CHECK: store i32 1, i32* getelementptr inbounds ([3 x %struct.X], [3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 2), align 8
  store i32 1, i32* getelementptr ([3 x %struct.X], [3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 2), align 4
; CHECK: store i32 1, i32* getelementptr inbounds ([3 x %struct.X], [3 x %struct.X]* @Y, i64 0, i64 0, i32 1, i64 0), align 4
  store i32 1, i32* getelementptr ([3 x %struct.X], [3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 3), align 4
; CHECK: store i32 1, i32* getelementptr inbounds ([3 x %struct.X], [3 x %struct.X]* @Y, i64 0, i64 0, i32 1, i64 1), align 4
  store i32 1, i32* getelementptr ([3 x %struct.X], [3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 4), align 4
; CHECK: store i32 1, i32* getelementptr inbounds ([3 x %struct.X], [3 x %struct.X]* @Y, i64 0, i64 0, i32 1, i64 2), align 4
  store i32 1, i32* getelementptr ([3 x %struct.X], [3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 5), align 4
; CHECK: store i32 1, i32* getelementptr inbounds ([3 x %struct.X], [3 x %struct.X]* @Y, i64 0, i64 1, i32 0, i64 0), align 8
  store i32 1, i32* getelementptr ([3 x %struct.X], [3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 6), align 4
; CHECK: store i32 1, i32* getelementptr inbounds ([3 x %struct.X], [3 x %struct.X]* @Y, i64 0, i64 1, i32 0, i64 1), align 4
  store i32 1, i32* getelementptr ([3 x %struct.X], [3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 7), align 4
; CHECK: store i32 1, i32* getelementptr inbounds ([3 x %struct.X], [3 x %struct.X]* @Y, i64 0, i64 1, i32 0, i64 2), align 8
  store i32 1, i32* getelementptr ([3 x %struct.X], [3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 8), align 4
; CHECK: store i32 1, i32* getelementptr inbounds ([3 x %struct.X], [3 x %struct.X]* @Y, i64 0, i64 1, i32 1, i64 0), align 4
  store i32 1, i32* getelementptr ([3 x %struct.X], [3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 9), align 4
; CHECK: store i32 1, i32* getelementptr inbounds ([3 x %struct.X], [3 x %struct.X]* @Y, i64 0, i64 1, i32 1, i64 1), align 4
  store i32 1, i32* getelementptr ([3 x %struct.X], [3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 10), align 4
; CHECK: store i32 1, i32* getelementptr inbounds ([3 x %struct.X], [3 x %struct.X]* @Y, i64 0, i64 1, i32 1, i64 2), align 4
  store i32 1, i32* getelementptr ([3 x %struct.X], [3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 11), align 4
; CHECK: store i32 1, i32* getelementptr inbounds ([3 x %struct.X], [3 x %struct.X]* @Y, i64 0, i64 2, i32 0, i64 0), align 16
  store i32 1, i32* getelementptr ([3 x %struct.X], [3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 12), align 4
; CHECK: store i32 1, i32* getelementptr inbounds ([3 x %struct.X], [3 x %struct.X]* @Y, i64 0, i64 2, i32 0, i64 1), align 4
  store i32 1, i32* getelementptr ([3 x %struct.X], [3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 13), align 4
; CHECK: store i32 1, i32* getelementptr inbounds ([3 x %struct.X], [3 x %struct.X]* @Y, i64 0, i64 2, i32 0, i64 2), align 8
  store i32 1, i32* getelementptr ([3 x %struct.X], [3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 14), align 8
; CHECK: store i32 1, i32* getelementptr inbounds ([3 x %struct.X], [3 x %struct.X]* @Y, i64 0, i64 2, i32 1, i64 0), align 8
  store i32 1, i32* getelementptr ([3 x %struct.X], [3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 15), align 8
; CHECK: store i32 1, i32* getelementptr inbounds ([3 x %struct.X], [3 x %struct.X]* @Y, i64 0, i64 2, i32 1, i64 1), align 8
  store i32 1, i32* getelementptr ([3 x %struct.X], [3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 16), align 8
; CHECK: store i32 1, i32* getelementptr inbounds ([3 x %struct.X], [3 x %struct.X]* @Y, i64 0, i64 2, i32 1, i64 2), align 8
  store i32 1, i32* getelementptr ([3 x %struct.X], [3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 17), align 8
; CHECK: store i32 1, i32* getelementptr inbounds ([3 x %struct.X], [3 x %struct.X]* @Y, i64 1, i64 0, i32 0, i64 0), align 8
  store i32 1, i32* getelementptr ([3 x %struct.X], [3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 18), align 8
; CHECK: store i32 1, i32* getelementptr ([3 x %struct.X], [3 x %struct.X]* @Y, i64 2, i64 0, i32 0, i64 0), align 16
  store i32 1, i32* getelementptr ([3 x %struct.X], [3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 36), align 8
; CHECK: store i32 1, i32* getelementptr ([3 x %struct.X], [3 x %struct.X]* @Y, i64 1, i64 0, i32 0, i64 1), align 8
  store i32 1, i32* getelementptr ([3 x %struct.X], [3 x %struct.X]* @Y, i64 0, i64 0, i32 0, i64 19), align 8
  ret void
}


; PR8883 - Constant fold exotic gep subtract
; CHECK-LABEL: @test2(
@X = global [1000 x i8] zeroinitializer, align 16

define i64 @test2() {
entry:
  %A = bitcast i8* getelementptr inbounds ([1000 x i8], [1000 x i8]* @X, i64 1, i64 0) to i8*
  %B = bitcast i8* getelementptr inbounds ([1000 x i8], [1000 x i8]* @X, i64 0, i64 0) to i8*

  %B2 = ptrtoint i8* %B to i64
  %C = sub i64 0, %B2
  %D = getelementptr i8, i8* %A, i64 %C
  %E = ptrtoint i8* %D to i64

  ret i64 %E
  ; CHECK: ret i64 1000
}

@X_as1 = addrspace(1) global [1000 x i8] zeroinitializer, align 16

define i16 @test2_as1() {
; CHECK-LABEL: @test2_as1(
  ; CHECK: ret i16 1000

entry:
  %A = bitcast i8 addrspace(1)* getelementptr inbounds ([1000 x i8], [1000 x i8] addrspace(1)* @X_as1, i64 1, i64 0) to i8 addrspace(1)*
  %B = bitcast i8 addrspace(1)* getelementptr inbounds ([1000 x i8], [1000 x i8] addrspace(1)* @X_as1, i64 0, i64 0) to i8 addrspace(1)*

  %B2 = ptrtoint i8 addrspace(1)* %B to i16
  %C = sub i16 0, %B2
  %D = getelementptr i8, i8 addrspace(1)* %A, i16 %C
  %E = ptrtoint i8 addrspace(1)* %D to i16

  ret i16 %E
}
