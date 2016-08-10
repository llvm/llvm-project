; RUN: opt < %s -S -early-cse | FileCheck %s
; RUN: opt < %s -S -passes=early-cse | FileCheck %s

declare {}* @llvm.invariant.start.p0i8(i64, i8* nocapture) nounwind readonly
declare void @llvm.invariant.end.p0i8({}*, i64, i8* nocapture) nounwind

; Check that we do load-load forwarding over invariant.start, since it does not
; clobber memory
define i8 @test1(i8 *%P) {
  ; CHECK-LABEL: @test1(
  ; CHECK-NEXT: %V1 = load i8, i8* %P
  ; CHECK-NEXT: %i = call {}* @llvm.invariant.start.p0i8(i64 1, i8* %P)
  ; CHECK-NEXT: ret i8 0


  %V1 = load i8, i8* %P
  %i = call {}* @llvm.invariant.start.p0i8(i64 1, i8* %P)
  %V2 = load i8, i8* %P
  %Diff = sub i8 %V1, %V2
  ret i8 %Diff
}


; Trivial Store->load forwarding over invariant.start
define i8 @test2(i8 *%P) {
  ; CHECK-LABEL: @test2(
  ; CHECK-NEXT: store i8 42, i8* %P
  ; CHECK-NEXT: %i = call {}* @llvm.invariant.start.p0i8(i64 1, i8* %P)
  ; CHECK-NEXT: ret i8 42


  store i8 42, i8* %P
  %i = call {}* @llvm.invariant.start.p0i8(i64 1, i8* %P)
  %V1 = load i8, i8* %P
  ret i8 %V1
}

; We can DSE over invariant.start calls, since the first store to
; %P is valid, and the second store is actually unreachable based on semantics
; of invariant.start.
define void @test3(i8* %P) {

; CHECK-LABEL: @test3(
; CHECK-NEXT:  %i = call {}* @llvm.invariant.start.p0i8(i64 1, i8* %P)
; CHECK-NEXT:  store i8 60, i8* %P


  store i8 50, i8* %P
  %i = call {}* @llvm.invariant.start.p0i8(i64 1, i8* %P)
  store i8 60, i8* %P
  ret void
}


; FIXME: Now the first store can actually be eliminated, since there is no read within
; the invariant region, between start and end.
define void @test4(i8* %P) {

; CHECK-LABEL: @test4(
; CHECK-NEXT: store i8 50, i8* %P
; CHECK-NEXT:  %i = call {}* @llvm.invariant.start.p0i8(i64 1, i8* %P)
; CHECK-NEXT: call void @llvm.invariant.end.p0i8({}* %i, i64 1, i8* %P)
; CHECK-NEXT:  store i8 60, i8* %P


  store i8 50, i8* %P
  %i = call {}* @llvm.invariant.start.p0i8(i64 1, i8* %P)
  call void @llvm.invariant.end.p0i8({}* %i, i64 1, i8* %P)
  store i8 60, i8* %P
  ret void
}
