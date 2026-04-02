; Make sure we do not assert for the cases we do not handle.
; RUN: llc -march=hexagon -mattr=+hvx,+hvx-length128b,+hvxv75,+v75,-long-calls < %s | FileCheck %s

; Mainly make sure we do not core dump.
; CHECK-NOT: scatter

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(argmem: write, inaccessiblemem: readwrite)
define dso_local void @foo(ptr noundef writeonly captures(none) %cptr, i32 noundef %T, i32 noundef %W) local_unnamed_addr #0 {
entry:
  %invariant.gep11 = getelementptr i8, ptr %cptr, i32 0
  %invariant.gep13 = getelementptr i8, ptr %invariant.gep11, i32 0
  %cmp.not15 = icmp ugt i32 8, %T
  br i1 %cmp.not15, label %for.cond.cleanup, label %for.cond1.preheader.lr.ph

for.cond1.preheader.lr.ph:                        ; preds = %entry
  %cmp3.not8 = icmp ugt i32 8, %W
  %conv.ripple.LS.instance = trunc i32 %W to i8
  %conv.ripple.LS.instance.ripple.bcast.splatinsert = insertelement <64 x i8> poison, i8 %conv.ripple.LS.instance, i64 0
  %conv.ripple.LS.instance.ripple.bcast.splat = shufflevector <64 x i8> %conv.ripple.LS.instance.ripple.bcast.splatinsert, <64 x i8> poison, <64 x i32> zeroinitializer
  br label %for.cond1.preheader

for.cond.loopexit:                                ; preds = %for.body5, %for.cond1.preheader
  %add = add i32 %add17, 8
  %cmp.not = icmp ugt i32 %add, %T
  br i1 %cmp.not, label %for.cond.cleanup, label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond1.preheader.lr.ph, %for.cond.loopexit
  %add17 = phi i32 [ 8, %for.cond1.preheader.lr.ph ], [ %add, %for.cond.loopexit ]
  %t.016 = phi i32 [ 0, %for.cond1.preheader.lr.ph ], [ %add17, %for.cond.loopexit ]
  br i1 %cmp3.not8, label %for.cond.loopexit, label %for.body5.lr.ph

for.body5.lr.ph:                                  ; preds = %for.cond1.preheader
  %gep14 = getelementptr i8, ptr %invariant.gep13, i32 %t.016
  br label %for.body5

for.cond.cleanup:                                 ; preds = %for.cond.loopexit, %entry
  ret void

for.body5:                                        ; preds = %for.body5.lr.ph, %for.body5
  %add210 = phi i32 [ 8, %for.body5.lr.ph ], [ %add2, %for.body5 ]
  %w.09 = phi i32 [ 0, %for.body5.lr.ph ], [ %add210, %for.body5 ]
  %gep = getelementptr i8, ptr %gep14, i32 %w.09
  %gep.ripple.LS.instance = getelementptr i8, ptr %gep, <64 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14>
  call void @llvm.masked.scatter.v64i8.v64p0(<64 x i8> %conv.ripple.LS.instance.ripple.bcast.splat, <64 x ptr> %gep.ripple.LS.instance, i32 1, <64 x i1> splat (i1 true))
  %add2 = add i32 %add210, 8
  %cmp3.not = icmp ugt i32 %add2, %W
  br i1 %cmp3.not, label %for.cond.loopexit, label %for.body5
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.ripple.block.setsize.i32(i32 immarg %0, i32 immarg %1, i32 %2) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: read)
declare i32 @llvm.ripple.block.index.i32(i32 immarg %0, i32 immarg %1) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: read)
declare i32 @llvm.ripple.block.getsize.i32(i32 immarg %0, i32 immarg %1) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(write)
declare void @llvm.masked.scatter.v64i8.v64p0(<64 x i8> %0, <64 x ptr> %1, i32 immarg %2, <64 x i1> %3) #3
