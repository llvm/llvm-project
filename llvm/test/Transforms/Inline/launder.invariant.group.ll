; RUN: opt -S -inline < %s | FileCheck %s
; RUN: opt -S -O3 < %s | FileCheck %s
; RUN: opt -S -inline -inline-threshold=1 < %s | FileCheck %s

%struct.A = type <{ i32 (...)**, i32, [4 x i8] }>

; This test checks if value returned from the launder is considered aliasing
; with its argument.  Due to bug caused by handling launder in capture tracking
; sometimes it would be considered noalias.
; CHECK-LABEL: define i32 @bar(%struct.A* noalias
define i32 @bar(%struct.A* noalias) {
; CHECK-NOT: noalias
  %2 = bitcast %struct.A* %0 to i8*
  %3 = call i8* @llvm.launder.invariant.group.p0i8(i8* %2)
  %4 = getelementptr inbounds i8, i8* %3, i64 8
  %5 = bitcast i8* %4 to i32*
  store i32 42, i32* %5, align 8
  %6 = getelementptr inbounds %struct.A, %struct.A* %0, i64 0, i32 1
  %7 = load i32, i32* %6, align 8
  ret i32 %7
}

; CHECK-LABEL: define i32 @foo(%struct.A* noalias
define i32 @foo(%struct.A* noalias)  {
  ; CHECK-NOT: call i32 @bar(
  ; CHECK-NOT: !noalias
  %2 = tail call i32 @bar(%struct.A* %0)
  ret i32 %2
}


; This test checks if invariant group intrinsics have zero cost for inlining.
; CHECK-LABEL: define i8* @caller(i8*
define i8* @caller(i8* %p) {
; CHECK-NOT: call i8* @lot_of_launders_and_strips
  %a1 = call i8* @lot_of_launders_and_strips(i8* %p)
  %a2 = call i8* @lot_of_launders_and_strips(i8* %a1)
  %a3 = call i8* @lot_of_launders_and_strips(i8* %a2)
  %a4 = call i8* @lot_of_launders_and_strips(i8* %a3)
  ret i8* %a4
}

define i8* @lot_of_launders_and_strips(i8* %p) {
  %a1 = call i8* @llvm.launder.invariant.group.p0i8(i8* %p)
  %a2 = call i8* @llvm.launder.invariant.group.p0i8(i8* %a1)
  %a3 = call i8* @llvm.launder.invariant.group.p0i8(i8* %a2)
  %a4 = call i8* @llvm.launder.invariant.group.p0i8(i8* %a3)

  %s1 = call i8* @llvm.strip.invariant.group.p0i8(i8* %a4)
  %s2 = call i8* @llvm.strip.invariant.group.p0i8(i8* %s1)
  %s3 = call i8* @llvm.strip.invariant.group.p0i8(i8* %s2)
  %s4 = call i8* @llvm.strip.invariant.group.p0i8(i8* %s3)

   ret i8* %s4
}


declare i8* @llvm.launder.invariant.group.p0i8(i8*)
declare i8* @llvm.strip.invariant.group.p0i8(i8*)
