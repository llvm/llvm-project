; Check the nonflattened part of the ctxsplit profile will be read in thinlto
; postlink phase while flattened part of the ctxsplit profile will not be read.
; RUN: opt < %s -passes='thinlto<O2>' -pgo-kind=pgo-sample-use-pipeline -profile-file=%S/Inputs/ctxsplit.extbinary.afdo -S | FileCheck %s --check-prefix=POSTLINK
;
; Check both the flattened and nonflattened parts of the ctxsplit profile will
; be read in thinlto prelink phase.
; RUN: opt < %s -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline -profile-file=%S/Inputs/ctxsplit.extbinary.afdo -S | FileCheck %s --check-prefix=PRELINK
;
; Check both the flattened and nonflattened parts of the ctxsplit profile will
; be read in non-thinlto mode.
; RUN: opt < %s -passes='default<O2>' -pgo-kind=pgo-sample-use-pipeline -profile-file=%S/Inputs/ctxsplit.extbinary.afdo -S | FileCheck %s --check-prefix=NOTHINLTO

; POSTLINK: define dso_local i32 @goo() {{.*}} !prof ![[ENTRY1:[0-9]+]] {
; POSTLINK: define dso_local i32 @foo() {{.*}} !prof ![[ENTRY2:[0-9]+]] {
; POSTLINK: ![[ENTRY1]] = !{!"function_entry_count", i64 1001}
; POSTLINK: ![[ENTRY2]] = !{!"function_entry_count", i64 -1}
; PRELINK: define dso_local i32 @goo() {{.*}} !prof ![[ENTRY1:[0-9]+]] {
; PRELINK: define dso_local i32 @foo() {{.*}} !prof ![[ENTRY2:[0-9]+]] {
; PRELINK: ![[ENTRY1]] = !{!"function_entry_count", i64 1001}
; PRELINK: ![[ENTRY2]] = !{!"function_entry_count", i64 3001}
; NOTHINLTO: define dso_local i32 @goo() {{.*}} !prof ![[ENTRY1:[0-9]+]] {
; NOTHINLTO: define dso_local i32 @foo() {{.*}} !prof ![[ENTRY2:[0-9]+]] {
; NOTHINLTO: ![[ENTRY1]] = !{!"function_entry_count", i64 1001}
; NOTHINLTO: ![[ENTRY2]] = !{!"function_entry_count", i64 3001}

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: norecurse nounwind readnone uwtable
define dso_local i32 @goo() #0 !dbg !10 {
entry:
  ret i32 -1, !dbg !11
}

; Function Attrs: norecurse nounwind readnone uwtable
define dso_local i32 @foo() #0 !dbg !7 {
entry:
  ret i32 -1, !dbg !9
}

attributes #0 = { "use-sample-profile" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 8.0.0 (trunk 345241)", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "a.c", directory: "")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 8.0.0 (trunk 345241)"}
!7 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: true, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 2, column: 3, scope: !7)
!10 = distinct !DISubprogram(name: "goo", scope: !1, file: !1, line: 8, type: !8, isLocal: false, isDefinition: true, scopeLine: 8, isOptimized: true, unit: !0, retainedNodes: !2)
!11 = !DILocation(line: 10, column: 3, scope: !10)

