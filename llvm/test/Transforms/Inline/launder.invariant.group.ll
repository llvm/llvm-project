; RUN: opt -S -inline < %s | FileCheck %s
; RUN: opt -S -O3 < %s | FileCheck %s

; This test checks if value returned from the launder is considered aliasing
; with its argument.  Due to bug caused by handling launder in capture tracking
; sometimes it would be considered noalias.

%struct.A = type <{ i32 (...)**, i32, [4 x i8] }>

; CHECK: define i32 @bar(%struct.A* noalias
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
  ; CHECK-NOT: noalias
  %2 = tail call i32 @bar(%struct.A* %0)
  ret i32 %2
}

declare i8* @llvm.launder.invariant.group.p0i8(i8*)
