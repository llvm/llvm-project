; RUN: opt -hotcoldsplit -S < %s | FileCheck %s

; CHECK-LABEL: define {{.*}}@foo_if.end
; CHECK-NOT: llvm.dbg.value

define void @foo(i32 %arg1) !dbg !6 {
entry:
  %var = add i32 0, 0, !dbg !11
  br i1 undef, label %if.then, label %if.end, !dbg !12

if.then:                                          ; preds = %entry
  ret void, !dbg !13

if.end:                                           ; preds = %entry
  call void @llvm.dbg.value(metadata i32 %arg1, metadata !9, metadata !DIExpression()), !dbg !11
  br label %if.then12, !dbg !14

if.then12:                                        ; preds = %if.end
  br label %cleanup40, !dbg !15

cleanup40:                                        ; preds = %if.then12
  br i1 undef, label %if.then5, label %if.end1, !dbg !16

if.then5:
  br label %return, !dbg !17

if.end1:
  br label %return, !dbg !18

return:                                           ; preds = %cleanup40
  unreachable, !dbg !19
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!3, !4}
!llvm.module.flags = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "<stdin>", directory: "/")
!2 = !{}
!3 = !{i32 7}
!4 = !{i32 1}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: null, file: !1, line: 1, type: !7, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: true, unit: !0, retainedNodes: !8)
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
!19 = !DILocation(line: 9, column: 1, scope: !6)
