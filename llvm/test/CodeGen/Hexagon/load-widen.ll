; RUN: llc -mtriple=hexagon < %s | FileCheck %s
; RUN: llc -mtriple=hexagon -disable-load-widen < %s | FileCheck %s --check-prefix=CHECK-DISABLE

%struct.node32 = type { ptr, ptr }

%struct.node16_4 = type { i16, i16, i16, i16 }

define void @test1(ptr nocapture %node) nounwind {
entry:
; There should be a memd and not two memw
; CHECK-LABEL: test1
; CHECK: memd
  %0 = load ptr, ptr %node, align 8
  %cgep = getelementptr inbounds %struct.node32, ptr %node, i32 0, i32 1
  %1 = load ptr, ptr %cgep, align 4
  store ptr %0, ptr %1, align 8
  ret void
}

define void @test2(ptr nocapture %node) nounwind {
entry:
; Same as test1 but with load widening disabled.
; CHECK-DISABLE-LABEL: test2
; CHECK-DISABLE: memw
; CHECK-DISABLE: memw
  %0 = load ptr, ptr %node, align 8
  %cgep = getelementptr inbounds %struct.node32, ptr %node, i32 0, i32 1
  %1 = load ptr, ptr %cgep, align 4
  store ptr %0, ptr %1, align 8
  ret void
}

define void @test3(ptr nocapture %node) nounwind {
entry:
; No memd because first load is not 8 byte aligned
; CHECK-LABEL: test3
; CHECK-NOT: memd
  %0 = load ptr, ptr %node, align 4
  %cgep = getelementptr inbounds %struct.node32, ptr %node, i32 0, i32 1
  %1 = load ptr, ptr %cgep, align 4
  store ptr %0, ptr %1, align 8
  ret void
}
