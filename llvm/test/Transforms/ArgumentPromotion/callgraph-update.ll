; RUN: opt < %s -argpromotion -simplifycfg -constmerge | llvm-dis
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin10.0"

%struct.VEC2 = type { double, double, double }
%struct.VERTEX = type { %struct.VEC2, %struct.VERTEX*, %struct.VERTEX* }
%struct.edge_rec = type { %struct.VERTEX*, %struct.edge_rec*, i32, i8* }

declare %struct.edge_rec* @alloc_edge() nounwind ssp

define i64 @build_delaunay(%struct.VERTEX* %tree, %struct.VERTEX* %extra) nounwind ssp {
entry:
  br i1 undef, label %bb11, label %bb12

bb11:                                             ; preds = %bb10
  %a = call %struct.edge_rec* @alloc_edge() nounwind ; <%struct.edge_rec*> [#uses=0]
  ret i64 123

bb12:                                             ; preds = %bb10
  %b = call %struct.edge_rec* @alloc_edge() nounwind ; <%struct.edge_rec*> [#uses=1]
  %c = ptrtoint %struct.edge_rec* %b to i64
  ret i64 %c
}
