; RUN: opt < %s -basic-aa -dse -S | FileCheck %s

target datalayout = "e-p:64:64:64"

declare void @free(i8* nocapture)
declare noalias i8* @malloc(i64)

; CHECK-LABEL: @test(
; CHECK-NEXT: bitcast
; CHECK-NEXT: @free
; CHECK-NEXT: ret void
define void @test(i32* %Q, i32* %P) {
        %DEAD = load i32, i32* %Q            ; <i32> [#uses=1]
        store i32 %DEAD, i32* %P
        %1 = bitcast i32* %P to i8*
        tail call void @free(i8* %1) nounwind
        ret void
}

; CHECK-LABEL: @test2(
; CHECK-NEXT: bitcast
; CHECK-NEXT: @free
; CHECK-NEXT: ret void
define void @test2({i32, i32}* %P) {
	%Q = getelementptr {i32, i32}, {i32, i32} *%P, i32 0, i32 1
	store i32 4, i32* %Q
        %1 = bitcast {i32, i32}* %P to i8*
        tail call void @free(i8* %1) nounwind
	ret void
}

; CHECK-LABEL: @test3(
; CHECK-NOT: store
; CHECK: ret void
define void @test3() {
  %m = call i8* @malloc(i64 24)
  store i8 0, i8* %m
  %m1 = getelementptr i8, i8* %m, i64 1
  store i8 1, i8* %m1
  call void @free(i8* %m) nounwind
  ret void
}

; PR11240
; CHECK-LABEL: @test4(
; CHECK-NOT: store
; CHECK: ret void
define void @test4(i1 %x) nounwind {
entry:
  %alloc1 = tail call noalias i8* @malloc(i64 4) nounwind
  br i1 %x, label %skipinit1, label %init1

init1:
  store i8 1, i8* %alloc1
  br label %skipinit1

skipinit1:
  tail call void @free(i8* %alloc1) nounwind
  ret void
}

; CHECK-LABEL: @test5(
define void @test5() {
  br label %bb

bb:
  tail call void @free(i8* undef) nounwind
  br label %bb
}

