; RUN: opt -strip-debug < %s -S | FileCheck %s

; CHECK-NOT: call void @llvm.dbg.value

source_filename = "test/Transforms/StripSymbols/2010-06-30-StripDebug.ll"

@x = common global i32 0, !dbg !0

; Function Attrs: nounwind optsize readnone ssp
define void @foo() #0 !dbg !7 {
entry:
  tail call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !10, metadata !12), !dbg !13
  ret void, !dbg !14
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind optsize readnone ssp }
attributes #1 = { nounwind readnone }

!llvm.module.flags = !{!4}
!llvm.dbg.cu = !{!5}

!0 = !DIGlobalVariableExpression(var: !1)
!1 = !DIGlobalVariable(name: "x", scope: !2, file: !2, line: 1, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "b.c", directory: "/tmp")
!3 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!4 = !{i32 1, !"Debug Info Version", i32 3}
!5 = distinct !DICompileUnit(language: DW_LANG_C89, file: !2, producer: "4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !6)
!6 = !{!0}
!7 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: !2, file: !2, line: 2, type: !8, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: true, unit: !5)
!8 = !DISubroutineType(types: !9)
!9 = !{null}!10 = !DILocalVariable(name: "y", scope: !11, file: !2, line: 3, type: !3)!11 = distinct !DILexicalBlock(scope: !7, file: !2, line: 2)!12 = !DIExpression()!13 = !DILocation(line: 3, scope: !11)!14 = !DILocation(line: 4, scope: !11)