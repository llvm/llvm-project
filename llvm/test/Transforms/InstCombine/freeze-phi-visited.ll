; RUN: opt < %s -passes=instcombine -S -debug 2>&1 | FileCheck %s
; REQUIRES: asserts

; Repeatedly pushing freeze thru PHIs can be expensive so we keep track of PHIs
; for which freeze has already been pushed thru.

define i1 @a(i1 %min.iters.check2957, <4 x double> %wide.load2970, <4 x i1> %0, <4 x i1> %1) {
.lr.ph1566:
  br i1 %min.iters.check2957, label %vec.epilog.ph2984, label %vector.ph2959

vector.ph2959:                                    ; preds = %.lr.ph1566
  %2 = fcmp olt <4 x double> %wide.load2970, zeroinitializer
  %bin.rdx2976 = or <4 x i1> %2, %0
  %bin.rdx2977 = or <4 x i1> %1, %bin.rdx2976
  %3 = call i1 @llvm.vector.reduce.or.v4i1(<4 x i1> %bin.rdx2977)
  %rdx.select2979 = select i1 %3, i32 1, i32 0
  br label %vec.epilog.ph2984

vec.epilog.ph2984:                                ; preds = %vector.ph2959, %.lr.ph1566
  %bc.merge.rdx2982 = phi i32 [ %rdx.select2979, %vector.ph2959 ], [ 0, %.lr.ph1566 ]
  %4 = icmp ne i32 %bc.merge.rdx2982, 0
  %broadcast.splatinsert2992 = insertelement <2 x i1> zeroinitializer, i1 %4, i64 0
  %5 = call i1 @llvm.vector.reduce.or.v2i1(<2 x i1> %broadcast.splatinsert2992)
  %6 = freeze i1 %5
  %7 = freeze i1 %6
  ret i1 %7
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i1 @llvm.vector.reduce.or.v4i1(<4 x i1>) #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i1 @llvm.vector.reduce.or.v2i1(<2 x i1>) #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

; CHECK: freeze has already been pushed through PHI 'bc.merge.rdx2982', skipping.
