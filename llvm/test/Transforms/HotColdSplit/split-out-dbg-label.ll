; RUN: opt -passes=hotcoldsplit -hotcoldsplit-threshold=0 -S < %s | FileCheck %s

; When an llvm.dbg.label intrinsic is extracted into a new function, make sure
; that its metadata argument is a DILabel that points to a scope within the new
; function.
;
; In this example, the label "bye" points to the scope for @foo before
; splitting, and should point to the scope for @foo.cold.1 after splitting.

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.14.0"

; CHECK-LABEL: define {{.*}}@foo.cold.1
; CHECK: #dbg_label([[LABEL:![0-9]+]],  [[LINE:![0-9]+]]
; CHECK: #dbg_label([[LABEL_IN_INLINE_ME:![0-9]+]],  [[LINE2:![0-9]+]]
; CHECK: #dbg_label([[SCOPED_LABEL:![0-9]+]],  [[LINE]]

; CHECK: [[FILE:![0-9]+]] = !DIFile
; CHECK: [[INLINE_ME_SCOPE:![0-9]+]] = distinct !DISubprogram(name: "inline_me"
; CHECK: [[SCOPE:![0-9]+]] = distinct !DISubprogram(name: "foo.cold.1"
; CHECK: [[LABEL]] = !DILabel(scope: [[SCOPE]], name: "bye", file: [[FILE]], line: 28
; CHECK: [[LINE]] = !DILocation(line: 1, column: 1, scope: [[SCOPE]]
; CHECK: [[LABEL_IN_INLINE_ME]] = !DILabel(scope: [[INLINE_ME_SCOPE]], name: "label_in_@inline_me", file: [[FILE]], line: 29
; CHECK: [[LINE2]] = !DILocation(line: 2, column: 2, scope: [[INLINE_ME_SCOPE]], inlinedAt: [[LINE]]
; CHECK: [[SCOPED_LABEL]] = !DILabel(scope: [[SCOPE_IN_FOO:![0-9]+]], name: "scoped_label_in_foo", file: [[FILE]], line: 30
; CHECK: [[SCOPE_IN_FOO]] = !DILexicalBlock(scope: [[SCOPE]], file: [[FILE]], line: 31, column: 31)

define void @foo(i32 %arg1) !dbg !6 {
entry:
  %var = add i32 0, 0, !dbg !11
  br i1 undef, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  ret void

if.end:                                           ; preds = %entry
  call void @llvm.dbg.label(metadata !12), !dbg !11
  call void @llvm.dbg.label(metadata !14), !dbg !15
  call void @llvm.dbg.label(metadata !16), !dbg !11
  call void @sink()
  ret void
}

declare void @llvm.dbg.label(metadata)

declare void @sink() cold

define void @inline_me() !dbg !13 {
  ret void
}

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
!12 = !DILabel(scope: !6, name: "bye", file: !1, line: 28)
!13 = distinct !DISubprogram(name: "inline_me", linkageName: "inline_me", scope: null, file: !1, line: 1, type: !7, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: true, unit: !0, retainedNodes: !8)
!14 = !DILabel(scope: !13, name: "label_in_@inline_me", file: !1, line: 29)
!15 = !DILocation(line: 2, column: 2, scope: !13, inlinedAt: !11)
!16 = !DILabel(scope: !17, name: "scoped_label_in_foo", file: !1, line: 30)
!17 = distinct !DILexicalBlock(scope: !6, file: !1, line: 31, column: 31)
