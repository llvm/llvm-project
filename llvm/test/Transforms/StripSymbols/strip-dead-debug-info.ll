; RUN: opt -strip-dead-debug-info -verify %s -S | FileCheck %s
; RUN: opt -passes='strip-dead-debug-info,verify' %s -S | FileCheck %s

; CHECK: ModuleID = '{{.*}}'
; CHECK-NOT: "bar"
; CHECK-NOT: "abcd"
; CHECK-NOT: "GCC"
; CHECK: "Globals"
; CHECK: "abcd2"

source_filename = "test/Transforms/StripSymbols/strip-dead-debug-info.ll"

@xyz = global i32 2, !dbg !0

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) #0

; Function Attrs: nounwind readnone ssp
define i32 @fn() #1 !dbg !10 {
entry:
  ret i32 0, !dbg !13
}

; Function Attrs: nounwind readonly ssp
define i32 @foo(i32 %i) #2 !dbg !15 {
entry:
  tail call void @llvm.dbg.value(metadata i32 %i, metadata !18, metadata !19), !dbg !20
  %.0 = load i32, i32* @xyz, align 4, !dbg !30
  ret i32 %.0, !dbg !21
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind readnone ssp }
attributes #2 = { nounwind readonly ssp }

!llvm.dbg.cu = !{!4, !23, !24, !28}
!llvm.module.flags = !{!9}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "xyz", scope: !2, file: !2, line: 3, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "g.c", directory: "/tmp/")
!3 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!4 = distinct !DICompileUnit(language: DW_LANG_C89, file: !2, producer: "4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !6)
!5 = !{}
!6 = !{!7, !0}
!7 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression())
!8 = !DIGlobalVariable(name: "abcd", scope: !2, file: !2, line: 2, type: !3, isLocal: true, isDefinition: true)
!9 = !{i32 1, !"Debug Info Version", i32 3}
!10 = distinct !DISubprogram(name: "fn", linkageName: "fn", scope: null, file: !2, line: 6, type: !11, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: true, unit: !4)
!11 = !DISubroutineType(types: !12)
!12 = !{!3}
!13 = !DILocation(line: 6, scope: !14)
!14 = distinct !DILexicalBlock(scope: !10, file: !2, line: 6)
!15 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: null, file: !2, line: 7, type: !16, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: true, unit: !4)
!16 = !DISubroutineType(types: !17)
!17 = !{!3, !3}
!18 = !DILocalVariable(name: "i", arg: 1, scope: !15, file: !2, line: 7, type: !3)
!19 = !DIExpression()
!20 = !DILocation(line: 7, scope: !15)
!21 = !DILocation(line: 10, scope: !22)
!22 = distinct !DILexicalBlock(scope: !15, file: !2, line: 7)
!23 = distinct !DICompileUnit(language: DW_LANG_C89, file: !2, producer: "GCC", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !5)
!24 = distinct !DICompileUnit(language: DW_LANG_C89, file: !2, producer: "Globals", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !25)
!25 = !{!26}
!26 = !DIGlobalVariableExpression(var: !27, expr: !DIExpression(DW_OP_constu, 0, DW_OP_stack_value))
!27 = !DIGlobalVariable(name: "abcd2", scope: !2, file: !2, line: 2, type: !3, isLocal: true, isDefinition: true)
!28 = distinct !DICompileUnit(language: DW_LANG_C89, file: !2, producer: "InlineTest", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, retainedTypes: !5, globals: !5)
!29 = distinct !DISubprogram(name: "inlinefunc", linkageName: "inlinefunc", scope: null, file: !2, line: 7, type: !16, isLocal: false, isDefinition: true, isOptimized: true, unit: !28)
!30 = !DILocation(line: 100, scope: !29, inlinedAt: !21)
