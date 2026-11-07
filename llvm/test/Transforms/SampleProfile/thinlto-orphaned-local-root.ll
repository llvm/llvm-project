; The pre-inliner can keep a local function's profile top-level (R below) while
; ThinLTO pre-link still inlines it into its single caller (G) and deletes it.
; R's nested decision to inline ext, defined in another module, then has no
; owner: its import GUID is attached to R and deleted with it, and in post-link
; G's profile has no context for R. With -sample-profile-preserve-deferred-inlines,
; pre-link copies R's import GUIDs to G, and post-link resolves frames inlined
; from R using R's top-level profile.

; RUN: split-file %s %t
; RUN: opt %t/pre.ll -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline \
; RUN:   -sample-profile-file=%t/prof.txt -S | FileCheck %s --check-prefix=PRE-OFF
; RUN: opt %t/pre.ll -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline \
; RUN:   -sample-profile-file=%t/prof.txt -sample-profile-preserve-deferred-inlines -S \
; RUN:   | FileCheck %s --check-prefix=PRE-ON
; RUN: opt %t/post.ll -passes='thinlto<O2>' -pgo-kind=pgo-sample-use-pipeline \
; RUN:   -sample-profile-file=%t/prof.txt -pass-remarks=sample-profile-inline \
; RUN:   -o /dev/null 2>&1 | FileCheck %s --check-prefix=POST-OFF --allow-empty
; RUN: opt %t/post.ll -passes='thinlto<O2>' -pgo-kind=pgo-sample-use-pipeline \
; RUN:   -sample-profile-file=%t/prof.txt -sample-profile-preserve-deferred-inlines \
; RUN:   -pass-remarks=sample-profile-inline -o /dev/null 2>&1 | FileCheck %s --check-prefix=POST-ON

;; Post-link rejects R's profile when R is not local (it may have an
;; out-of-line copy elsewhere), when G's call site does not account for all of
;; R's entries, or when R's checksum is stale.
; RUN: sed 's/DISPFlagLocalToUnit | //' %t/post.ll > %t/post-nonlocal.ll
; RUN: opt %t/post-nonlocal.ll -passes='thinlto<O2>' -pgo-kind=pgo-sample-use-pipeline \
; RUN:   -sample-profile-file=%t/prof.txt -sample-profile-preserve-deferred-inlines \
; RUN:   -pass-remarks=sample-profile-inline -o /dev/null 2>&1 \
; RUN:   | FileCheck %s --check-prefix=POST-OFF --allow-empty
; RUN: sed 's/^R:2000:1000$/R:2000:2000/' %t/prof.txt > %t/prof-multi.txt
; RUN: opt %t/post.ll -passes='thinlto<O2>' -pgo-kind=pgo-sample-use-pipeline \
; RUN:   -sample-profile-file=%t/prof-multi.txt -sample-profile-preserve-deferred-inlines \
; RUN:   -pass-remarks=sample-profile-inline -o /dev/null 2>&1 \
; RUN:   | FileCheck %s --check-prefix=POST-OFF --allow-empty
; RUN: sed 's/^ !CFGChecksum: 2$/ !CFGChecksum: 9/' %t/prof.txt > %t/prof-stale.txt
; RUN: opt %t/post.ll -passes='thinlto<O2>' -pgo-kind=pgo-sample-use-pipeline \
; RUN:   -sample-profile-file=%t/prof-stale.txt -sample-profile-preserve-deferred-inlines \
; RUN:   -pass-remarks=sample-profile-inline -o /dev/null 2>&1 \
; RUN:   | FileCheck %s --check-prefix=POST-OFF --allow-empty

;; ext's GUID is -2953423517159131221.
; PRE-OFF: define {{.*}} @G({{.*}} !prof ![[#G:]]
; PRE-OFF: ![[#G]] = !{!"function_entry_count", i64 1000}

; PRE-ON: define {{.*}} @G({{.*}} !prof ![[#G:]]
; PRE-ON-NOT: define {{.*}} @R(
; PRE-ON: ![[#G]] = !{!"function_entry_count", i64 1000, i64 -2953423517159131221}

; POST-OFF-NOT: 'ext' inlined into 'G'
; POST-ON: remark: a.c:2:0: 'ext' inlined into 'G' to match profiling context with (cost=always): preinliner

;--- prof.txt
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

;--- pre.ll
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

;--- post.ll
;; pre.ll after pre-link with R inlined into G, and ext imported.
source_filename = "a.c"
target triple = "x86_64-unknown-linux-gnu"

define i64 @G(i64 %n) #0 !dbg !5 !guid !14 {
  call void @llvm.pseudoprobe(i64 -6455552227143004193, i64 1, i32 0, i64 -1), !dbg !6
  call void @llvm.pseudoprobe(i64 9084449867242398177, i64 1, i32 0, i64 -1), !dbg !15
  %r = call i64 @ext(i64 %n), !dbg !16
  ret i64 %r
}

define available_externally i64 @ext(i64 %n) #0 !dbg !18 !guid !19 {
  call void @llvm.pseudoprobe(i64 -2953423517159131221, i64 1, i32 0, i64 -1), !dbg !20
  ret i64 %n
}

declare void @llvm.pseudoprobe(i64, i64, i32, i64)

attributes #0 = { "use-sample-profile" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}
!llvm.pseudo_probe_desc = !{!3, !4, !17}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "a.c", directory: "")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i64 -6455552227143004193, i64 1, !"G"}
!4 = !{i64 9084449867242398177, i64 2, !"R"}
!5 = distinct !DISubprogram(name: "G", file: !1, line: 1, type: !13, spFlags: DISPFlagDefinition, unit: !0)
!6 = !DILocation(line: 1, scope: !5)
!7 = distinct !DILocation(line: 1, scope: !8)
!8 = !DILexicalBlockFile(scope: !5, file: !1, discriminator: 186646551)
!9 = distinct !DISubprogram(name: "R", file: !1, line: 2, type: !13, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0)
!12 = !DILexicalBlockFile(scope: !9, file: !1, discriminator: 186646551)
!13 = !DISubroutineType(types: !{})
!14 = !{i64 -6455552227143004193}
!15 = !DILocation(line: 2, scope: !9, inlinedAt: !7)
!16 = !DILocation(line: 2, scope: !12, inlinedAt: !7)
!17 = !{i64 -2953423517159131221, i64 3, !"ext"}
!18 = distinct !DISubprogram(name: "ext", file: !1, line: 3, type: !13, spFlags: DISPFlagDefinition, unit: !0)
!19 = !{i64 -2953423517159131221}
!20 = !DILocation(line: 3, scope: !18)
