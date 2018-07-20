; RUN: llc -march=hexagon -mcpu=hexagonv5 < %s | FileCheck %s
; CHECK: r{{[0-9]*}} = memw(r{{[0-9]+}}++#-4:circ(m0))


%union.vect64 = type { i64 }
%union.vect32 = type { i32 }

define i32* @HallowedBeThyName(%union.vect64* nocapture %pRx, %union.vect32* %pLut, %union.vect64* nocapture %pOut, i64 %dc.coerce, i32 %shift, i32 %numSamples) nounwind {
entry:
  %vLutNext = alloca i32, align 4
  %0 = bitcast %union.vect32* %pLut to i8*
  %1 = bitcast i32* %vLutNext to i8*
  %2 = call i8* @llvm.hexagon.circ.ldw(i8* %0, i8* %1, i32 83886144, i32 -4)
  %3 = bitcast i8* %2 to i32*
  ret i32* %3
}

declare i8* @llvm.hexagon.circ.ldw(i8*, i8*, i32, i32) nounwind
