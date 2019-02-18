; RUN: llc -march=hexagon -mattr=+hvxv60,hvx-length64b,-packets < %s | FileCheck %s

target triple = "hexagon"

; Check that the only (multi-instruction) packet is the one with vgather.

; CHECK-LABEL: fred:
; CHECK:      {
; CHECK-NEXT:   allocframe(r29,#64):raw
; CHECK-NEXT: }
; CHECK-NEXT: {
; CHECK-NEXT:   m0 = r2
; CHECK-NEXT: }
; CHECK-NEXT: {
; CHECK-NEXT:   r29 = and(r29,#-64)
; CHECK-NEXT: }
; CHECK-NEXT: {
; CHECK-NEXT:   vtmp.w = vgather(r1,m0,v0.w).w
; CHECK-NEXT:   vmem(r0+#0) = vtmp.new
; CHECK-NEXT: }
; CHECK-NEXT: {
; CHECK-NEXT:   r0 = add(r29,#0)
; CHECK-NEXT: }
; CHECK-NEXT: {
; CHECK-NEXT:   call foo
; CHECK-NEXT: }
; CHECK-NEXT: {
; CHECK-NEXT:   r31:30 = dealloc_return(r30):raw
; CHECK-NEXT: }

define void @fred(i8* %p, i32 %x, i32 %y) local_unnamed_addr #0 {
entry:
  %v = alloca <16 x i32>, align 64
  %0 = bitcast <16 x i32>* %v to i8*
  call void @llvm.lifetime.start(i64 64, i8* nonnull %0) #3
  tail call void @llvm.hexagon.V6.vgathermw(i8* %p, i32 %x, i32 %y, <16 x i32> undef)
  call void @foo(i8* nonnull %0) #0
  call void @llvm.lifetime.end(i64 64, i8* nonnull %0) #3
  ret void
}

declare void @llvm.lifetime.start(i64, i8* nocapture) #1
declare void @llvm.hexagon.V6.vgathermw(i8*, i32, i32, <16 x i32>) #1
declare void @foo(i8*) local_unnamed_addr #0
declare void @llvm.lifetime.end(i64, i8* nocapture) #1

attributes #0 = { nounwind "target-cpu"="hexagonv65" }
attributes #1 = { argmemonly nounwind }
