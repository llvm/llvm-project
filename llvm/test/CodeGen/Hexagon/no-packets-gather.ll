; RUN: llc -march=hexagon -mattr=+hvxv60,hvx-length64b < %s | FileCheck %s --check-prefix=CHECK-GATHER
; RUN: not llc -march=hexagon -mattr=+hvxv60,hvx-length64b,-packets %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

target triple = "hexagon"

; CHECK-GATHER-LABEL: fred:
; CHECK-GATHER: vgather
; CHECK-ERROR: LLVM ERROR: Support for gather requires packets, which are disabled

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
