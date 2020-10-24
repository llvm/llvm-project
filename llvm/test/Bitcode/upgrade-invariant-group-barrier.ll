; RUN: opt -S < %s | FileCheck %s

; The intrinsic firstly only took i8*, then it was made polimorphic, then
; it was renamed to launder.invariant.group
define void @test(i8* %p1, i16* %p16) {
; CHECK-LABEL: @test
; CHECK: %p2 = call i8* @llvm.launder.invariant.group.p0i8(i8* %p1)
; CHECK: %p3 = call i8* @llvm.launder.invariant.group.p0i8(i8* %p1)
; CHECK: %p4 = call i16* @llvm.launder.invariant.group.p0i16(i16* %p16)
  %p2 = call i8* @llvm.invariant.group.barrier(i8* %p1)
  %p3 = call i8* @llvm.invariant.group.barrier.p0i8(i8* %p1)
  %p4 = call i16* @llvm.invariant.group.barrier.p0i16(i16* %p16)
  ret void
}

; CHECK: Function Attrs: inaccessiblememonly nofree nosync nounwind speculatable willreturn
; CHECK: declare i8* @llvm.launder.invariant.group.p0i8(i8*)
; CHECK: Function Attrs: inaccessiblememonly nofree nosync nounwind speculatable willreturn
; CHECK: declare i16* @llvm.launder.invariant.group.p0i16(i16*)
declare i8* @llvm.invariant.group.barrier(i8*)
declare i8* @llvm.invariant.group.barrier.p0i8(i8*)
declare i16* @llvm.invariant.group.barrier.p0i16(i16*)
