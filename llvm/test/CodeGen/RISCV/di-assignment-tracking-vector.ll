; RUN: llc -mtriple=riscv64 < %s | FileCheck %s --implicit-check-not=DEBUG_VALUE

;; Verify that tagged and untagged non-contiguous stores are handled correctly
;; by assignment tracking.
;; * The store to "i" is untagged, and results in the memory location being
;;   dropped in favour of the debug value 1010 after the store.
;; * The store to "j" is tagged with a corresponding dbg_assign, which allows
;;   us to keep using the memory location.

; CHECK-LABEL: foo:
; CHECK-NEXT:  .Lfunc_begin0:
; CHECK:       # %bb.0
; CHECK:         addi    a1, sp, 48
; CHECK-NEXT:    #DEBUG_VALUE: foo:i <- [DW_OP_deref] $x12
; CHECK-NEXT:    #DEBUG_VALUE: foo:j <- [DW_OP_deref] $x12
; CHECK:         vsse32.v
; CHECK-NEXT:    #DEBUG_VALUE: foo:i <- 1010
; CHECK-NEXT:    vsse32.v


target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "riscv64-unknown-linux-gnu"

define void @foo() #0 !dbg !5 {
entry:
  %i = alloca i64, align 8, !DIAssignID !6
  %j = alloca i64, align 8, !DIAssignID !12
  %sar_height.i = getelementptr i8, ptr %i, i64 24
  store ptr %sar_height.i, ptr null, align 8
  %vui.i = getelementptr i8, ptr %i, i64 44
  %0 = load i32, ptr %vui.i, align 4
  %sar_width.i = getelementptr i8, ptr %i, i64 20
  %i_sar_width.i = getelementptr i8, ptr %i, i64 48
  %j_sar_width.j = getelementptr i8, ptr %j, i64 48
    #dbg_assign(i32 1010, !7, !DIExpression(), !6, ptr %i_sar_width.i, !DIExpression(), !9)
    #dbg_assign(i32 2121, !17, !DIExpression(), !12, ptr %i_sar_width.i, !DIExpression(), !9)
  %1 = load <2 x i32>, ptr %sar_width.i, align 4
  call void @llvm.experimental.vp.strided.store.v2i32.p0.i64(<2 x i32> %1, ptr align 4 %i_sar_width.i, i64 -4, <2 x i1> splat (i1 true), i32 2)
  call void @llvm.experimental.vp.strided.store.v2i32.p0.i64(<2 x i32> %1, ptr align 4 %j_sar_width.j, i64 -4, <2 x i1> splat (i1 true), i32 2), !DIAssignID !13
    #dbg_assign(i32 1010, !7, !DIExpression(), !14, ptr %i_sar_width.i, !DIExpression(), !9)
    #dbg_assign(i32 2121, !17, !DIExpression(), !13, ptr %i_sar_width.i, !DIExpression(), !9)
  ret void
}

attributes #0 = { "target-features"="+v" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, producer: "clang version 21.0.0git")
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!5 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: !1, file: !1, line: 1, scopeLine: 1, type: !10, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!6 = distinct !DIAssignID()
!7 = !DILocalVariable(name: "i", scope: !5, file: !1, line: 7, type: !8)
!8 = !DIBasicType(name: "int32_t", size: 32, encoding: DW_ATE_signed)
!9 = !DILocation(line: 5, scope: !5)
!10 = !DISubroutineType(types: !2)
!11 = !{!7, !17}
!12 = distinct !DIAssignID()
!13 = distinct !DIAssignID()
!14 = distinct !DIAssignID()
!17 = !DILocalVariable(name: "j", scope: !5, file: !1, line: 7, type: !8)
