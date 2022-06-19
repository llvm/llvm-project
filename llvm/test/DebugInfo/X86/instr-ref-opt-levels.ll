; RUN: llc -mtriple=x86_64-- %s -o - -O0 -stop-before=finalize-isel \
; RUN:      -experimental-debug-variable-locations \
; RUN:    | FileCheck %s --check-prefix=O0 --implicit-check-not=DBG_INSTR_REF
; RUN: llc -mtriple=x86_64-- %s -o - -O2 -stop-before=finalize-isel \
; RUN:      -experimental-debug-variable-locations \
; RUN:    | FileCheck %s --check-prefix=O2 --implicit-check-not=DBG_VALUE
; RUN: llc -mtriple=x86_64-- %s -o - -stop-before=finalize-isel \
; RUN:      -fast-isel -experimental-debug-variable-locations \
; RUN:    | FileCheck %s --check-prefix=FASTISEL --implicit-check-not=DBG_VALUE

; Test that instruction-referencing variable locations are issued at -O2, but
; normal DBG_VALUEs are issued at -O0. This behaviour is desired as the former
; is slow when applied to unoptimized code.
; (Copy + pasted from ./dead-store-elimination-marks-undef.ll),

; O0-LABEL: name: main
; O0:       DBG_VALUE
; O0-LABEL: name: fn_optnone
; O0:       DBG_VALUE

; O2-LABEL: name: main
; O2:       DBG_INSTR_REF
; O2-LABEL: name: fn_optnone
; O2:       DBG_VALUE

; FASTISEL-LABEL: name: main
; FASTISEL:       DBG_INSTR_REF
; FASTISEL-LABEL: name: fn_optnone
; FASTISEL:       DBG_VALUE

@b = common dso_local local_unnamed_addr global i32 0, align 1

define dso_local i32 @main() local_unnamed_addr !dbg !7 {
  %1 = alloca i32, align 4
  %2 = load i32, i32* @b, align 1, !dbg !13
  call void @llvm.dbg.value(metadata i32 %2, metadata !12, metadata !DIExpression()), !dbg !13
  store i32 %2, i32* %1, align 4, !dbg !13
  ret i32 0, !dbg !13
}

define dso_local i32 @fn_optnone() local_unnamed_addr #0 !dbg !27 {
  %1 = alloca i32, align 4
  %2 = load i32, i32* @b, align 1, !dbg !33
  call void @llvm.dbg.value(metadata i32 %2, metadata !32, metadata !DIExpression()), !dbg !33
  store i32 %2, i32* %1, align 4, !dbg !33
  ret i32 0, !dbg !33
}

attributes #0 = {optnone noinline}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 10.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !2, nameTableKind: None)
!1 = !DIFile(filename: "dead-store-elimination-marks-undef.ll", directory: "/temp/bz45080")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !{!"clang version 10.0.0"}
!7 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 3, type: !8, scopeLine: 3, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!12}
!12 = !DILocalVariable(name: "l_2864", scope: !7, file: !1, line: 4, type: !10)
!13 = !DILocation(line: 5, column: 12, scope: !7)
!27 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 3, type: !8, scopeLine: 3, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !31)
!31 = !{!32}
!32 = !DILocalVariable(name: "l_2864", scope: !27, file: !1, line: 4, type: !10)
!33 = !DILocation(line: 5, column: 12, scope: !27)
