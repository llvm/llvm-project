; RUN: llc -mtriple=hexagon -mcpu=hexagonv5 < %s | FileCheck %s
; CHECK: r{{[0-9]*}} = memw(r{{[0-9]+}}++#-4:circ(m0))


%union.vect64 = type { i64 }
%union.vect32 = type { i32 }

define ptr @HallowedBeThyName(ptr nocapture %pRx, ptr %pLut, ptr nocapture %pOut, i64 %dc.coerce, i32 %shift, i32 %numSamples) nounwind {
entry:
  %vLutNext = alloca i32, align 4
  %0 = call ptr @llvm.hexagon.circ.ldw(ptr %pLut, ptr %vLutNext, i32 83886144, i32 -4)
  ret ptr %0
}

declare ptr @llvm.hexagon.circ.ldw(ptr, ptr, i32, i32) nounwind
