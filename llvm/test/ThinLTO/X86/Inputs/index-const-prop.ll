target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

@gBar = local_unnamed_addr global i32 2, align 4, !dbg !0
@gFoo = internal unnamed_addr global i32 1, align 4, !dbg !6

; Function Attrs: norecurse nounwind readonly
define i32 @foo() local_unnamed_addr #0 !dbg !14 {
  %1 = load i32, i32* @gFoo, align 4, !dbg !17
  ret i32 %1, !dbg !18
}

; Function Attrs: norecurse nounwind readonly
define i32 @bar() local_unnamed_addr #0 !dbg !19 {
  %1 = load i32, i32* @gBar, align 4, !dbg !20
  ret i32 %1, !dbg !21
}

define void @baz() local_unnamed_addr !dbg !22 {
  %1 = tail call i32 @rand(), !dbg !25
  store i32 %1, i32* @gFoo, align 4, !dbg !26
  %2 = tail call i32 @rand(), !dbg !27
  store i32 %2, i32* @gBar, align 4, !dbg !28
  ret void, !dbg !29
}

declare i32 @rand() local_unnamed_addr

attributes #0 = { norecurse nounwind readonly }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!9, !10, !11, !12}
!llvm.ident = !{!13}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "gBar", scope: !2, file: !3, line: 4, type: !8, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 7.0.0 (trunk 332246)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "foo.c", directory: "/data/work/lto/roref/test")
!4 = !{}
!5 = !{!0, !6}
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = distinct !DIGlobalVariable(name: "gFoo", scope: !2, file: !3, line: 3, type: !8, isLocal: true, isDefinition: true)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !{i32 2, !"Dwarf Version", i32 4}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{i32 1, !"wchar_size", i32 4}
!12 = !{i32 7, !"PIC Level", i32 2}
!13 = !{!"clang version 7.0.0 (trunk 332246)"}
!14 = distinct !DISubprogram(name: "foo", scope: !3, file: !3, line: 6, type: !15, isLocal: false, isDefinition: true, scopeLine: 6, isOptimized: true, unit: !2, retainedNodes: !4)
!15 = !DISubroutineType(types: !16)
!16 = !{!8}
!17 = !DILocation(line: 7, column: 10, scope: !14)
!18 = !DILocation(line: 7, column: 3, scope: !14)
!19 = distinct !DISubprogram(name: "bar", scope: !3, file: !3, line: 10, type: !15, isLocal: false, isDefinition: true, scopeLine: 10, isOptimized: true, unit: !2, retainedNodes: !4)
!20 = !DILocation(line: 11, column: 10, scope: !19)
!21 = !DILocation(line: 11, column: 3, scope: !19)
!22 = distinct !DISubprogram(name: "baz", scope: !3, file: !3, line: 14, type: !23, isLocal: false, isDefinition: true, scopeLine: 14, isOptimized: true, unit: !2, retainedNodes: !4)
!23 = !DISubroutineType(types: !24)
!24 = !{null}
!25 = !DILocation(line: 15, column: 10, scope: !22)
!26 = !DILocation(line: 15, column: 8, scope: !22)
!27 = !DILocation(line: 16, column: 10, scope: !22)
!28 = !DILocation(line: 16, column: 8, scope: !22)
!29 = !DILocation(line: 17, column: 1, scope: !22)
