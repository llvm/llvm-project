; RUN: opt < %s -basicaa -dse -S | FileCheck %s
target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"

; Ensure that the dead store is deleted in this case.  It is wholely
; overwritten by the second store.
define void @test1(i32 *%V) {
        %V2 = bitcast i32* %V to i8*            ; <i8*> [#uses=1]
        store i8 0, i8* %V2
        store i32 1234567, i32* %V
        ret void
; CHECK: @test1
; CHECK-NEXT: store i32 1234567
}

; Note that we could do better by merging the two stores into one.
define void @test2(i32* %P) {
; CHECK: @test2
  store i32 0, i32* %P
; CHECK: store i32
  %Q = bitcast i32* %P to i16*
  store i16 1, i16* %Q
; CHECK: store i16
  ret void
}


define i32 @test3(double %__x) {
; CHECK: @test3
; CHECK: store double
  %__u = alloca { [3 x i32] }
  %tmp.1 = bitcast { [3 x i32] }* %__u to double*
  store double %__x, double* %tmp.1
  %tmp.4 = getelementptr { [3 x i32] }* %__u, i32 0, i32 0, i32 1
  %tmp.5 = load i32* %tmp.4
  %tmp.6 = icmp slt i32 %tmp.5, 0
  %tmp.7 = zext i1 %tmp.6 to i32
  ret i32 %tmp.7
}

; PR6043
define void @test4(i8* %P) {
; CHECK: @test4
; CHECK-NEXT: bitcast
; CHECK-NEXT: store double

  store i8 19, i8* %P  ;; dead
  %A = getelementptr i8* %P, i32 3
  
  store i8 42, i8* %A  ;; dead
  
  %Q = bitcast i8* %P to double*
  store double 0.0, double* %Q
  ret void
}

; PR8657
declare void @test5a(i32*)
define void @test5(i32 %i) nounwind ssp {
  %A = alloca i32
  %B = bitcast i32* %A to i8*
  %C = getelementptr i8* %B, i32 %i
  store i8 10, i8* %C        ;; Dead store to variable index.
  store i32 20, i32* %A
  
  call void @test5a(i32* %A)
  ret void
; CHECK: @test5(
; CHECK-NEXT: alloca
; CHECK-NEXT: store i32 20
; CHECK-NEXT: call void @test5a
}
