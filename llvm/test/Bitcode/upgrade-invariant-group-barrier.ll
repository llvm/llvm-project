; RUN: opt -S < %s | FileCheck %s

; The intrinsic firstly only took ptr, then it was made polimorphic, then
; it was renamed to launder.invariant.group
define void @test(ptr %p1, ptr %p16) {
; CHECK-LABEL: @test
; CHECK: %p2 = call ptr @llvm.launder.invariant.group.p0(ptr %p1)
; CHECK: %p3 = call ptr @llvm.launder.invariant.group.p0(ptr %p1)
; CHECK: %p4 = call ptr @llvm.launder.invariant.group.p0(ptr %p16)
  %p2 = call ptr @llvm.invariant.group.barrier(ptr %p1)
  %p3 = call ptr @llvm.invariant.group.barrier.p0(ptr %p1)
  %p4 = call ptr @llvm.invariant.group.barrier.p0(ptr %p16)
  ret void
}

; CHECK: Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(inaccessiblemem: readwrite)
; CHECK: declare ptr @llvm.launder.invariant.group.p0(ptr)
declare ptr @llvm.invariant.group.barrier(ptr)
declare ptr @llvm.invariant.group.barrier.p0(ptr)
