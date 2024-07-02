; RUN: opt -passes=check-debugify < %s -S 2>&1 | FileCheck %s
; RUN: opt --experimental-debuginfo-iterators=false -passes=check-debugify < %s -S 2>&1 | FileCheck %s

define void @test_phi(i1 %cond) !dbg !6 {
  br i1 %cond, label %1, label %2, !dbg !11

1:                                                ; preds = %0
  br label %2, !dbg !12

2:                                                ; preds = %1, %0
  %v = phi i32 [ 0, %0 ], [ 1, %1 ], !dbg !13
  call void @llvm.dbg.value(metadata i32 %v, metadata !9, metadata !DIExpression()), !dbg !13
  ret void, !dbg !14
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!3, !4}
!llvm.module.flags = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "<stdin>", directory: "/")
!2 = !{}
!3 = !{i32 4}
!4 = !{i32 1}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "test_phi", linkageName: "test_phi", scope: null, file: !1, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
!7 = !DISubroutineType(types: !2)
!8 = !{!9}
!9 = !DILocalVariable(name: "1", scope: !6, file: !1, line: 3, type: !10)
!10 = !DIBasicType(name: "ty32", size: 32, encoding: DW_ATE_unsigned)
!11 = !DILocation(line: 1, column: 1, scope: !6)
!12 = !DILocation(line: 2, column: 1, scope: !6)
!13 = !DILocation(line: 3, column: 1, scope: !6)
!14 = !DILocation(line: 4, column: 1, scope: !6)

; CHECK-NOT: WARNING: Missing line 3
; CHECK: CheckModuleDebugify: PASS
