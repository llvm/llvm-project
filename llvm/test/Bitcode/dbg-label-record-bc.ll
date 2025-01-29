;; Tests that we can parse and print a function containing a debug label record
;; and no other debug record kinds.

; RUN: llvm-as --write-experimental-debuginfo-iterators-to-bitcode=true %s -o - \
; RUN: | opt -S | FileCheck %s --check-prefixes=CHECK,INTRINSIC

; RUN: llvm-as --write-experimental-debuginfo-iterators-to-bitcode=true %s -o - \
; RUN: | opt -S --preserve-input-debuginfo-format=true \
; RUN: | FileCheck %s --check-prefixes=CHECK,RECORD

source_filename = "bbi-94196.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: void @foo()
; CHECK: bar:
; INTRINSIC-NEXT: #dbg_label(![[LABEL:[0-9]+]],  ![[LOC:[0-9]+]]
; RECORD-NEXT: #dbg_label(![[LABEL:[0-9]+]], ![[LOC:[0-9]+]])

; CHECK-DAG: ![[LABEL]] = !DILabel({{.*}}name: "bar"
; CHECK-DAG: ![[LOC]] = !DILocation(line: 5, column: 1

define dso_local void @foo() !dbg !5 {
entry:
  br label %bar, !dbg !9

bar:                                              ; preds = %entry
  tail call void @llvm.dbg.label(metadata !10), !dbg !11
  ret void, !dbg !12
}

declare void @llvm.dbg.label(metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!llvm.ident = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 19.0.0git", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "<stdin>", directory: "/home/gbtozers/dev/llvm-project-ddd-textual-ir")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{!"clang version 19.0.0git"}
!5 = distinct !DISubprogram(name: "foo", scope: !6, file: !6, line: 1, type: !7, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !0)
!6 = !DIFile(filename: "bbi-94196.c", directory: "/home/gbtozers/dev/llvm-project-ddd-textual-ir")
!7 = !DISubroutineType(types: !8)
!8 = !{null}
!9 = !DILocation(line: 3, column: 3, scope: !5)
!10 = !DILabel(scope: !5, name: "bar", file: !6, line: 5)
!11 = !DILocation(line: 5, column: 1, scope: !5)
!12 = !DILocation(line: 6, column: 3, scope: !5)
