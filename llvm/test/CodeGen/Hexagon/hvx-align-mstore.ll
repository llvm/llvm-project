; Test that we take alignment hint properly for mload/mstore
; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; Test that we take alignment hint properly

; CHECK-NOT: vmemu

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
define dso_local void @_Z31puzzle_prefix_assign_ripple_badPDF16_PKDF16_j(ptr noalias noundef writeonly align 128 captures(none) %dest, ptr noalias noundef readonly align 128 captures(none) %in, i32 noundef %len) local_unnamed_addr #0 {
entry:
  %and.ripple.LS.instance = and i32 %len, 3
  %and.ripple.LS.instance.ripple.bcast.splatinsert = insertelement <32 x i32> poison, i32 %and.ripple.LS.instance, i64 0
  %and.ripple.LS.instance.ripple.bcast.splat = shufflevector <32 x i32> %and.ripple.LS.instance.ripple.bcast.splatinsert, <32 x i32> poison, <32 x i32> zeroinitializer
  %cmp.ripple.LS.instance = icmp samesign ugt <32 x i32> %and.ripple.LS.instance.ripple.bcast.splat, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %.ripple.masked.load = tail call <32 x half> @llvm.masked.load.v32f16.p0(ptr %in, i32 2, <32 x i1> %cmp.ripple.LS.instance, <32 x half> poison)
  tail call void @llvm.masked.store.v32f16.p0(<32 x half> %.ripple.masked.load, ptr %dest, i32 2, <32 x i1> %cmp.ripple.LS.instance)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare <32 x half> @llvm.masked.load.v32f16.p0(ptr captures(none) %0, i32 immarg %1, <32 x i1> %2, <32 x half> %3) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: write)
declare void @llvm.masked.store.v32f16.p0(<32 x half> %0, ptr captures(none) %1, i32 immarg %2, <32 x i1> %3) #2

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv75" "target-features"="+hvx-length128b,+hvxv75,+v75,-long-calls" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{!"Clang $LLVM_VERSION_MAJOR.$LLVM_VERSION_MINOR"}
