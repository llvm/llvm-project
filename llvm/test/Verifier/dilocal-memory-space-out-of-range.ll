; RUN: not opt -S %s -o /dev/null 2>&1 | FileCheck %s
; CHECK: value for 'memorySpace' too large, limit is

define dso_local i32 @foo(i32 %var) !dbg !4 {
entry:
  call void @llvm.dbg.value(metadata i32 %var, metadata !9, metadata !DIExpression()), !dbg !10
  ret i32 %var
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_OpenCL, file: !1, producer: "clang version 16.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cl", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
!5 = !DISubroutineType(types: !6)
!6 = !{!7, !7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !{!9}
!9 = !DILocalVariable(name: "var", arg: 1, scope: !4, file: !1, line: 1, type: !7, memorySpace: 65536)
!10 = !DILocation(scope: !4, line: 1)
