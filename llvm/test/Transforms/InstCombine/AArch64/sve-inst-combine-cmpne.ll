; RUN: opt -S -mtriple=aarch64-unknown-linux-gnu -O2 < %s | FileCheck %s
 
; Function Attrs: nofree nosync nounwind readnone uwtable vscale_range(1,16)
define dso_local i32 @testInstCombineSVECmpNE() local_unnamed_addr #0 {
entry:
  %0 = tail call <vscale x 16 x i8> @llvm.aarch64.sve.index.nxv16i8(i8 42, i8 1)
  %1 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 31)
  br label %for.body
 
for.cond.cleanup:                                 ; preds = %for.inc
  %2 = tail call i1 @llvm.aarch64.sve.ptest.any.nxv16i1(<vscale x 16 x i1> %1, <vscale x 16 x i1> %cmp_rslt.1)
  %not. = xor i1 %2, true
  %. = zext i1 %not. to i32
  ret i32 %.
 
for.body:                                         ; preds = %entry, %for.inc
  %i.010 = phi i64 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp1 = icmp ugt i64 %i.010, 32
  %3 = tail call <vscale x 16 x i8> @llvm.aarch64.sve.dupq.lane.nxv16i8(<vscale x 16 x i8> %0, i64 %i.010)
  br i1 %cmp1, label %if.then, label %if.else
 
if.then:                                          ; preds = %for.body
  %4 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.cmpne.nxv16i8(<vscale x 16 x i1> %1, <vscale x 16 x i8> %3, <vscale x 16 x i8> zeroinitializer)
  br label %for.inc
  ; CHECK: %4 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.cmpne.nxv16i8(<vscale x 16 x i1> %1, <vscale x 16 x i8> %3, <vscale x 16 x i8> zeroinitializer)
  ; CHECK-NEXT: br label %for.inc
 
if.else:                                          ; preds = %for.body
  %5 = tail call <vscale x 16 x i1> @llvm.aarch64.sve.cmpne.nxv16i8(<vscale x 16 x i1> %1, <vscale x 16 x i8> %3, <vscale x 16 x i8> shufflevector (<vscale x 16 x i8> insertelement (<vscale x 16 x i8> poison, i8 1, i32 0), <vscale x 16 x i8> poison, <vscale x 16 x i32> zeroinitializer))
  br label %for.inc
 
for.inc:                                          ; preds = %if.then, %if.else
  %cmp_rslt.1 = phi <vscale x 16 x i1> [ %4, %if.then ], [ %5, %if.else ]
  %inc = add nuw nsw i64 %i.010, 1
  %exitcond.not = icmp eq i64 %inc, 63
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body, !llvm.loop !6
}
 
; Function Attrs: mustprogress nocallback nofree nosync nounwind readnone willreturn
declare <vscale x 16 x i8> @llvm.aarch64.sve.index.nxv16i8(i8, i8) #1
 
; Function Attrs: mustprogress nocallback nofree nosync nounwind readnone willreturn
declare <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 immarg) #1
 
; Function Attrs: mustprogress nocallback nofree nosync nounwind readnone willreturn
declare <vscale x 16 x i8> @llvm.aarch64.sve.dupq.lane.nxv16i8(<vscale x 16 x i8>, i64) #1
 
; Function Attrs: mustprogress nocallback nofree nosync nounwind readnone willreturn
declare <vscale x 16 x i1> @llvm.aarch64.sve.cmpne.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x i8>, <vscale x 16 x i8>) #1
 
; Function Attrs: mustprogress nocallback nofree nosync nounwind readnone willreturn
declare i1 @llvm.aarch64.sve.ptest.any.nxv16i1(<vscale x 16 x i1>, <vscale x 16 x i1>) #1
 
attributes #0 = { nofree nosync nounwind readnone uwtable vscale_range(1,16) "frame-pointer"="non-leaf" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+neon,+sve,+v8.2a" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind readnone willreturn }
 
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}
