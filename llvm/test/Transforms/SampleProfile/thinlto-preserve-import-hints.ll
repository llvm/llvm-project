; Preserve hints through actual pre-link inlining without changing entry counts.
; Multiple callers, retained bodies, and both writers are covered by
; Transforms/Inline/sample-pending-imports.ll.
;
; RUN: split-file %s %t
; DEFINE: %{pre} = opt -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline \
; DEFINE:   -sample-profile-file=%t/profile.txt -S
; DEFINE: %{check} = FileCheck %s --check-prefixes=ENTRY,ON
; RUN: %{pre} %t/input.ll -sample-profile-preserve-orphaned-inlines=false \
; RUN:   | FileCheck %s --check-prefix=ENTRY -DCOUNT=1000 --implicit-check-not=sample.pending.imports
; RUN: %{pre} %t/input.ll -o %t/pre.ll
; RUN: %{check} -DCOUNT=1000 --check-prefix=DELETED < %t/pre.ll
;; Reuse that output to check the critical edge and metadata cleanup.
; RUN: opt -module-summary %t/pre.ll -o %t/summary.bc
; RUN: llvm-dis %t/summary.bc -o - | FileCheck %s --check-prefix=SUMMARY --implicit-check-not=sample.pending.imports
;; The same behavior for a non-local, discardable definition and extbinary.
; RUN: sed -e 's/define internal/define linkonce_odr/' -e 's/DISPFlagLocalToUnit | //' -e 's/#0 !dbg !9/alwaysinline #0 !dbg !9/' %t/input.ll > %t/nonlocal.ll
; RUN: %{pre} %t/nonlocal.ll | %{check} -DCOUNT=1000
; RUN: llvm-profdata merge --sample --extbinary %t/profile.txt -o %t/profile.extbinary
; RUN: %{pre} %t/input.ll -sample-profile-file=%t/profile.extbinary | %{check} -DCOUNT=1000 --check-prefix=DELETED
;; Remove G's profile: its entry count stays unknown in both linkage cases.
; RUN: sed -n '/^R:/,$p' %t/profile.txt > %t/no-caller-profile.txt
; RUN: %{pre} %t/input.ll -sample-profile-file=%t/no-caller-profile.txt -sample-profile-preserve-orphaned-inlines=false \
; RUN:   | FileCheck %s --check-prefix=ENTRY -DCOUNT=-1 --implicit-check-not=sample.pending.imports
; RUN: %{pre} %t/input.ll -sample-profile-file=%t/no-caller-profile.txt | %{check} -DCOUNT=-1
; RUN: %{pre} %t/nonlocal.ll -sample-profile-file=%t/no-caller-profile.txt | %{check} -DCOUNT=-1

; ENTRY: define {{.*}} @G({{.*}} !prof ![[#G:]]
; ON-SAME: {{.*}}!sample.pending.imports ![[#P:]]
; DELETED-NOT: define {{.*}} @R(
; ENTRY: ![[#G]] = !{!"function_entry_count", i64 [[COUNT]]}
; ON: ![[#P]] = !{i64 {{-?[0-9]+}}, ![[#I:]]}
; ON: ![[#I]] = !{!"function_entry_count", i64 1000, i64 -2953423517159131221}
; SUMMARY-DAG: ^[[EXT:[0-9]+]] = gv: (name: "ext")
; SUMMARY-DAG: name: "G", {{.*}}calls: ((callee: ^[[EXT]], hotness: critical, tail: 1))

;--- profile.txt
G:2000:1000
 1: 1000
 2: 1000 R:1000
 !CFGChecksum: 1
R:2000:1000
 1: 1000
 2: ext:1000
  1: 1000
  !CFGChecksum: 3
  !Attributes: 2
 !CFGChecksum: 2

;--- input.ll
source_filename = "a.c"
target triple = "x86_64-unknown-linux-gnu"

define i64 @G(i64 %n) #0 !dbg !5 {
  call void @llvm.pseudoprobe(i64 -6455552227143004193, i64 1, i32 0, i64 -1), !dbg !6
  %r = call i64 @R(i64 %n), !dbg !7
  ret i64 %r
}

define internal i64 @R(i64 %n) #0 !dbg !9 {
  call void @llvm.pseudoprobe(i64 9084449867242398177, i64 1, i32 0, i64 -1), !dbg !10
  %r = call i64 @ext(i64 %n), !dbg !11
  ret i64 %r
}

declare i64 @ext(i64)
declare void @llvm.pseudoprobe(i64, i64, i32, i64)

attributes #0 = { "use-sample-profile" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}
!llvm.pseudo_probe_desc = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "a.c", directory: "")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i64 -6455552227143004193, i64 1, !"G"}
!4 = !{i64 9084449867242398177, i64 2, !"R"}
!5 = distinct !DISubprogram(name: "G", file: !1, line: 1, type: !13, spFlags: DISPFlagDefinition, unit: !0)
!6 = !DILocation(line: 1, scope: !5)
!7 = !DILocation(line: 1, scope: !8)
;; Discriminator 186646551 encodes direct call probe 2.
!8 = !DILexicalBlockFile(scope: !5, file: !1, discriminator: 186646551)
!9 = distinct !DISubprogram(name: "R", file: !1, line: 2, type: !13, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0)
!10 = !DILocation(line: 2, scope: !9)
!11 = !DILocation(line: 2, scope: !12)
!12 = !DILexicalBlockFile(scope: !9, file: !1, discriminator: 186646551)
!13 = !DISubroutineType(types: !{})
