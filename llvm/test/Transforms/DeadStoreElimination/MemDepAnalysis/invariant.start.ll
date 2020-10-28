; Test to make sure llvm.invariant.start calls are not treated as clobbers.
; RUN: opt < %s -basic-aa -dse -enable-dse-memoryssa=false -S | FileCheck %s

declare {}* @llvm.invariant.start.p0i8(i64, i8* nocapture) nounwind readonly

; We cannot remove the store 1 to %p.
; FIXME: By the semantics of invariant.start, the store 3 to p is unreachable.
define void @test(i8 *%p) {
  store i8 1, i8* %p, align 4
  %i = call {}* @llvm.invariant.start.p0i8(i64 1, i8* %p)
  store i8 3, i8* %p, align 4
  ret void
; CHECK-LABEL: @test(
; CHECK-NEXT: store i8 1, i8* %p, align 4
; CHECK-NEXT: %i = call {}* @llvm.invariant.start.p0i8(i64 1, i8* %p)
; CHECK-NEXT: store i8 3, i8* %p, align 4
; CHECK-NEXT: ret void
}

; FIXME: We should be able to remove the first store to p, even though p and q
; may alias.
define void @test2(i8* %p, i8* %q) {
  store i8 1, i8* %p, align 4
  store i8 2, i8* %q, align 4
  %i = call {}* @llvm.invariant.start.p0i8(i64 1, i8* %q)
  store i8 3, i8* %p, align 4
  ret void
; CHECK-LABEL: @test2(
; CHECK-NEXT: store i8 1, i8* %p, align 4
; CHECK-NEXT: store i8 2, i8* %q, align 4
; CHECK-NEXT: %i = call {}* @llvm.invariant.start.p0i8(i64 1, i8* %q)
; CHECK-NEXT: store i8 3, i8* %p, align 4
; CHECK-NEXT: ret void
}
