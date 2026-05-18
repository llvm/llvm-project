; RUN: llc -mtriple=mips -mfix4300 -verify-machineinstrs < %s | FileCheck %s

; Function Attrs: mustprogress nofree norecurse nosync nounwind readnone willreturn
define dso_local float @fun_s(float %x) local_unnamed_addr !dbg !7  {
entry:
; CHECK-LABEL: fun_s
; CHECK: mul.s
; CHECK-NEXT: #DEBUG_VALUE: i <- 1
; CHECK-NEXT: nop
  %mul = fmul float %x, %x
  call void @llvm.dbg.value(metadata i32 1, metadata !13, metadata !DIExpression()), !dbg !17
  %mul1 = fmul float %mul, %x
  ret float %mul1
}

define dso_local double @fun_d(double %x) local_unnamed_addr #0 {
entry:
; CHECK-LABEL: fun_d
; CHECK: mul.d
; CHECK-NEXT: nop
; CHECK: mul.d
  %mul = fmul double %x, %x
  %mul1 = fmul double %mul, %x
  ret double %mul1
}


; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "vr4300-mulmul.ll", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!7 = distinct !DISubprogram(name: "fun_s", linkageName: "fun_s", scope: !1, file: !1, line: 1, type: !8, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !10}
!10 = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
!13 = !DILocalVariable(name: "i", scope: !14, file: !1, line: 3, type: !15)
!14 = distinct !DILexicalBlock(scope: !7, file: !1, line: 3, column: 5)
!15 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!17 = !DILocation(line: 0, scope: !14)
