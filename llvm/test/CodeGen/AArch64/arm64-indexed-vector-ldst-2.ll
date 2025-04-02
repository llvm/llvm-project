; RUN: llc < %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios7.0.0"

; This used to assert with "Overran sorted position" in AssignTopologicalOrder
; due to a cycle created in performPostLD1Combine.

; Function Attrs: nounwind ssp
define void @f(ptr %P1) #0 {
entry:
  %arrayidx4 = getelementptr inbounds double, ptr %P1, i64 1
  %0 = load double, ptr %arrayidx4, align 8, !tbaa !1
  %1 = load double, ptr %P1, align 8, !tbaa !1
  %2 = insertelement <2 x double> undef, double %0, i32 0
  %3 = insertelement <2 x double> %2, double %1, i32 1
  %4 = fsub <2 x double> zeroinitializer, %3
  %5 = fmul <2 x double> undef, %4
  %6 = extractelement <2 x double> %5, i32 0
  %cmp168 = fcmp olt double %6, undef
  br i1 %cmp168, label %if.then172, label %return

if.then172:                                       ; preds = %cond.end90
  %7 = tail call i64 @llvm.objectsize.i64.p0(ptr undef, i1 false)
  br label %return

return:                                           ; preds = %if.then172, %cond.end90, %entry
  ret void
}

; Avoid an assert/bad codegen in LD1LANEPOST lowering by not forming
; LD1LANEPOST ISD nodes with a non-constant lane index.
define <4 x i32> @f2(ptr %p, <4 x i1> %m, <4 x i32> %v1, <4 x i32> %v2, i32 %idx) {
  %L0 = load i32, ptr %p
  %p1 = getelementptr i32, ptr %p, i64 1
  %L1 = load i32, ptr %p1
  %v = select <4 x i1> %m, <4 x i32> %v1, <4 x i32> %v2
  %vret = insertelement <4 x i32> %v, i32 %L0, i32 %idx
  store i32 %L1, ptr %p
  ret <4 x i32> %vret
}

; Check that a cycle is avoided during isel between the LD1LANEPOST instruction and the load of %L1.
define <4 x i32> @f3(ptr %p, <4 x i1> %m, <4 x i32> %v1, <4 x i32> %v2) {
  %L0 = load i32, ptr %p
  %p1 = getelementptr i32, ptr %p, i64 1
  %L1 = load i32, ptr %p1
  %v = select <4 x i1> %m, <4 x i32> %v1, <4 x i32> %v2
  %vret = insertelement <4 x i32> %v, i32 %L0, i32 %L1
  ret <4 x i32> %vret
}

; This test used to crash in performPostLD1Combine when the combine attempted to
; replace a load that already had index writeback, resulting in an incorrect
; CombineTo, which would have changed the number of SDValue results of the
; instruction.
define i32 @rdar138004275(ptr %arg, i1 %arg1) {
bb:
  br label %bb3

bb2:                                              ; preds = %bb3
  store volatile <8 x half> %shufflevector10, ptr null, align 16
  ret i32 0

bb3:                                              ; preds = %bb3, %bb
  %phi = phi ptr [ null, %bb ], [ %getelementptr11, %bb3 ]
  %load = load <2 x half>, ptr %phi, align 4
  %shufflevector = shufflevector <2 x half> %load, <2 x half> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %getelementptr = getelementptr i8, ptr %phi, i64 4
  %load4 = load half, ptr %getelementptr, align 2
  %insertelement = insertelement <2 x half> zeroinitializer, half %load4, i64 0
  %shufflevector5 = shufflevector <2 x half> %insertelement, <2 x half> zeroinitializer, <8 x i32> <i32 0, i32 0, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %shufflevector6 = shufflevector <8 x half> %shufflevector, <8 x half> %shufflevector5, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 8, i32 9, i32 6, i32 7>
  store <8 x half> %shufflevector6, ptr %arg, align 16
  %getelementptr7 = getelementptr i8, ptr %phi, i64 6
  %load8 = load <2 x half>, ptr %getelementptr7, align 4
  %shufflevector9 = shufflevector <2 x half> %load8, <2 x half> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %shufflevector10 = shufflevector <8 x half> %shufflevector9, <8 x half> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 10, i32 11, i32 poison, i32 poison, i32 14, i32 15>
  %getelementptr11 = getelementptr i8, ptr %phi, i64 6
  br i1 %arg1, label %bb2, label %bb3
}


; Function Attrs: nounwind readnone
declare i64 @llvm.objectsize.i64.p0(ptr, i1) #1

attributes #0 = { nounwind ssp "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!1 = !{!2, !2, i64 0}
!2 = !{!"double", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
