; RUN: opt -S %s -o - --passes=declare-to-assign \
; RUN: | FileCheck %s --implicit-check-not="call void @llvm.dbg"
; RUN: opt --try-experimental-debuginfo-iterators -S %s -o - --passes=declare-to-assign \
; RUN: | FileCheck %s --implicit-check-not="call void @llvm.dbg"

;; Assignment tracking doesn't add any value when optimisations are disabled.
;; Check it doesn't get applied to functions marked optnone.

; CHECK: @_Z3funv
; CHECK: #dbg_assign

; CHECK: @_Z3funv2
; CHECK: #dbg_declare

define dso_local void @_Z3funv() local_unnamed_addr !dbg !16 {
entry:
  %a = alloca i32
  call void @llvm.dbg.declare(metadata ptr %a, metadata !20, metadata !DIExpression()), !dbg !25
  ret void
}

; Function Attrs: noinline optnone
define dso_local void @_Z3funv2() local_unnamed_addr #1 !dbg !39 {
entry:
  %a = alloca i32
  call void @llvm.dbg.declare(metadata ptr %a, metadata !41, metadata !DIExpression()), !dbg !44
  ret void
}

attributes #1 = {noinline optnone}

declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!8, !9, !10, !11, !12, !13}
!llvm.ident = !{!15}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "G", scope: !2, file: !3, line: 1, type: !7, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 17.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test.cpp", directory: "/")
!4 = !{!0, !5}
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
!6 = distinct !DIGlobalVariable(name: "F", scope: !2, file: !3, line: 1, type: !7, isLocal: false, isDefinition: true)
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !{i32 7, !"Dwarf Version", i32 5}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{i32 1, !"wchar_size", i32 4}
!11 = !{i32 8, !"PIC Level", i32 2}
!12 = !{i32 7, !"PIE Level", i32 2}
!13 = !{i32 7, !"uwtable", i32 2}
!15 = !{!"clang version 17.0.0"}
!16 = distinct !DISubprogram(name: "fun", linkageName: "_Z3funv", scope: !3, file: !3, line: 3, type: !17, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !19)
!17 = !DISubroutineType(types: !18)
!18 = !{null}
!19 = !{!20}
!20 = !DILocalVariable(name: "X", scope: !16, file: !3, line: 4, type: !21)
!21 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!22 = !{}
!25 = !DILocation(line: 0, scope: !16)
!27 = distinct !DILexicalBlock(scope: !16, file: !3, line: 7, column: 7)
!39 = distinct !DISubprogram(name: "fun2", linkageName: "_Z3funv2", scope: !3, file: !3, line: 3, type: !17, scopeLine: 3, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !40)
!40 = !{!41}
!41 = !DILocalVariable(name: "X", scope: !39, file: !3, line: 10, type: !21)
!44 = !DILocation(line: 0, scope: !39)
