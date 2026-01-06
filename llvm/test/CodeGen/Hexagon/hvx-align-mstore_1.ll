; Test that we take alignment hint properly for mload/mstore
; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; Test that we take alignment hint properly

; CHECK-NOT: vmemu
; CHECK-NOT: vlalign

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite, inaccessiblemem: readwrite)
define dso_local void @foo(ptr noundef writeonly align 128 captures(none) %A, ptr noundef readonly align 128 captures(none) %B, ptr noundef readonly align 128 captures(none) %C, i32 noundef %key) local_unnamed_addr #0 {
entry:
  %key.ripple.bcast.splatinsert = insertelement <32 x i32> poison, i32 %key, i64 0
  %key.ripple.bcast.splat = shufflevector <32 x i32> %key.ripple.bcast.splatinsert, <32 x i32> poison, <32 x i32> zeroinitializer
  %cmp1.ripple.LS.instance = icmp ugt <32 x i32> %key.ripple.bcast.splat, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  br label %if.then.ripple.branch.clone

for.cond.cleanup:                                 ; preds = %if.then.ripple.branch.clone
  ret void

if.then.ripple.branch.clone:                      ; preds = %entry, %if.then.ripple.branch.clone
  %lsr.iv25 = phi ptr [ %A, %entry ], [ %scevgep26, %if.then.ripple.branch.clone ]
  %lsr.iv23 = phi ptr [ %C, %entry ], [ %scevgep24, %if.then.ripple.branch.clone ]
  %lsr.iv = phi ptr [ %B, %entry ], [ %scevgep, %if.then.ripple.branch.clone ]
  %i.08 = phi i32 [ 0, %entry ], [ %add7, %if.then.ripple.branch.clone ]
  %.ripple.masked.load = tail call <32 x i32> @llvm.masked.load.v32i32.p0(ptr %lsr.iv, i32 4, <32 x i1> %cmp1.ripple.LS.instance, <32 x i32> poison)
  %.ripple.masked.load21 = tail call <32 x i32> @llvm.masked.load.v32i32.p0(ptr %lsr.iv23, i32 4, <32 x i1> %cmp1.ripple.LS.instance, <32 x i32> poison)
  %add4.ripple.LS.instance.ripple.branch.clone = add <32 x i32> %.ripple.masked.load21, %.ripple.masked.load
  tail call void @llvm.masked.store.v32i32.p0(<32 x i32> %add4.ripple.LS.instance.ripple.branch.clone, ptr %lsr.iv25, i32 4, <32 x i1> %cmp1.ripple.LS.instance)
  %add7 = add i32 %i.08, 32
  %scevgep = getelementptr i8, ptr %lsr.iv, i32 128
  %scevgep24 = getelementptr i8, ptr %lsr.iv23, i32 128
  %scevgep26 = getelementptr i8, ptr %lsr.iv25, i32 128
  %cmp = icmp ult i32 %add7, 128
  br i1 %cmp, label %if.then.ripple.branch.clone, label %for.cond.cleanup, !llvm.loop !3
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare <32 x i32> @llvm.masked.load.v32i32.p0(ptr captures(none) %0, i32 immarg %1, <32 x i1> %2, <32 x i32> %3) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: write)
declare void @llvm.masked.store.v32i32.p0(<32 x i32> %0, ptr captures(none) %1, i32 immarg %2, <32 x i1> %3) #2

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: readwrite, inaccessiblemem: readwrite) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv75" "target-features"="+hvx-length128b,+hvxv75,+v75,-long-calls" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{!"Clang $LLVM_VERSION_MAJOR.$LLVM_VERSION_MINOR"}
!3 = distinct !{!3, !4}
!4 = !{!"llvm.loop.mustprogress"}

