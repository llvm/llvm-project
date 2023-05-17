; RUN: opt -S -passes=gvn-hoist < %s | FileCheck %s

; Check that the debug info is dropped as per the debug info update guide

@G = external global i32, align 4

; CHECK-LABEL: @foo
; CHECK-NOT:  store {{.*}}dbg
define void @foo(i32 %c1) !dbg !6 {
entry:
  call void @llvm.dbg.value(metadata i32 0, metadata !9, metadata !DIExpression()), !dbg !11
  switch i32 %c1, label %exit1 [
    i32 0, label %sw0
    i32 1, label %sw1
  ], !dbg !11

sw0:                                              ; preds = %entry
  store i32 1, ptr @G, align 4, !dbg !12
  br label %exit, !dbg !13

sw1:                                              ; preds = %entry
  store i32 1, ptr @G, align 4, !dbg !14
  br label %exit, !dbg !15

exit1:                                            ; preds = %entry
  store i32 1, ptr @G, align 4, !dbg !16
  ret void, !dbg !17

exit:                                             ; preds = %sw1, %sw0
  ret void, !dbg !18
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #0

attributes #0 = { nofree nosync nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!3, !4}
!llvm.module.flags = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "abc.ll", directory: "/")
!2 = !{}
!3 = !{i32 8}
!4 = !{i32 1}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: null, file: !1, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
!7 = !DISubroutineType(types: !2)
!8 = !{!9}
!9 = !DILocalVariable(name: "1", scope: !6, file: !1, line: 1, type: !10)
!10 = !DIBasicType(name: "ty32", size: 32, encoding: DW_ATE_unsigned)
!11 = !DILocation(line: 1, column: 1, scope: !6)
!12 = !DILocation(line: 2, column: 1, scope: !6)
!13 = !DILocation(line: 3, column: 1, scope: !6)
!14 = !DILocation(line: 4, column: 1, scope: !6)
!15 = !DILocation(line: 5, column: 1, scope: !6)
!16 = !DILocation(line: 6, column: 1, scope: !6)
!17 = !DILocation(line: 7, column: 1, scope: !6)
!18 = !DILocation(line: 8, column: 1, scope: !6)
