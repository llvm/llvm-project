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

;; The same profiles in extbinary format. Its reader loads only the profiles of
;; functions in the module, so post-link must load R's profile explicitly. The
;; same gates apply.
; RUN: llvm-profdata merge --sample --extbinary %t/prof.txt -o %t/prof.extbinary
; RUN: llvm-profdata merge --sample --extbinary %t/prof-multi.txt -o %t/prof-multi.extbinary
; RUN: llvm-profdata merge --sample --extbinary %t/prof-stale.txt -o %t/prof-stale.extbinary
; DEFINE: %{post} = opt -passes='thinlto<O2>' -pgo-kind=pgo-sample-use-pipeline \
; DEFINE:   -pass-remarks=sample-profile-inline -o /dev/null
; RUN: opt %t/pre.ll -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline \
; RUN:   -sample-profile-file=%t/prof.extbinary -sample-profile-preserve-deferred-inlines -S \
; RUN:   | FileCheck %s --check-prefix=PRE-ON
; RUN: %{post} %t/post.ll -sample-profile-file=%t/prof.extbinary 2>&1 \
; RUN:   | FileCheck %s --check-prefix=POST-OFF --allow-empty
; RUN: %{post} %t/post.ll -sample-profile-file=%t/prof.extbinary \
; RUN:   -sample-profile-preserve-deferred-inlines 2>&1 | FileCheck %s --check-prefix=POST-ON
; RUN: %{post} %t/post-nonlocal.ll -sample-profile-file=%t/prof.extbinary \
; RUN:   -sample-profile-preserve-deferred-inlines 2>&1 \
; RUN:   | FileCheck %s --check-prefix=POST-OFF --allow-empty
; RUN: %{post} %t/post.ll -sample-profile-file=%t/prof-multi.extbinary \
; RUN:   -sample-profile-preserve-deferred-inlines 2>&1 \
; RUN:   | FileCheck %s --check-prefix=POST-OFF --allow-empty
; RUN: %{post} %t/post.ll -sample-profile-file=%t/prof-stale.extbinary \
; RUN:   -sample-profile-preserve-deferred-inlines 2>&1 \
; RUN:   | FileCheck %s --check-prefix=POST-OFF --allow-empty
;; MD5 names, including the Eytzinger offset table layout.
; RUN: llvm-profdata merge --sample --extbinary --use-md5 %t/prof.txt -o %t/prof.md5
; RUN: llvm-profdata merge --sample --extbinary --use-md5 --md5-indexed-tables \
; RUN:   %t/prof.txt -o %t/prof.md5.eytzinger
; RUN: %{post} %t/post.ll -sample-profile-file=%t/prof.md5 \
; RUN:   -sample-profile-preserve-deferred-inlines 2>&1 | FileCheck %s --check-prefix=POST-ON
; RUN: %{post} %t/post.ll -sample-profile-file=%t/prof.md5.eytzinger \
; RUN:   -sample-profile-preserve-deferred-inlines 2>&1 | FileCheck %s --check-prefix=POST-ON

;; Post-link loads a root only when a nested lookup misses and the caller
;; records a call to it. With R's root loaded, R's frame misses S, so a second
;; round loads S's root.
; RUN: llvm-profdata merge --sample --extbinary %t/prof-chain.txt -o %t/prof-chain.extbinary
; RUN: %{post} %t/post-chain.ll -sample-profile-file=%t/prof-chain.extbinary \
; RUN:   -sample-profile-preserve-deferred-inlines 2>&1 | FileCheck %s --check-prefix=CHAIN
; RUN: %if asserts %{ %{post} %t/post-chain.ll -sample-profile-file=%t/prof-chain.extbinary \
; RUN:   -sample-profile-preserve-deferred-inlines -stats 2>&1 \
; RUN:   | FileCheck %s --check-prefix=LOADED2 %}
;; R and Q, two promoted targets of one call, share the inline site in G. Both
;; roots are loaded.
; RUN: llvm-profdata merge --sample --extbinary %t/prof-shared.txt -o %t/prof-shared.extbinary
; RUN: %{post} %t/post-shared.ll -sample-profile-file=%t/prof-shared.extbinary \
; RUN:   -sample-profile-preserve-deferred-inlines 2>&1 | FileCheck %s --check-prefix=SHARED
; RUN: %if asserts %{ %{post} %t/post-shared.ll -sample-profile-file=%t/prof-shared.extbinary \
; RUN:   -sample-profile-preserve-deferred-inlines -stats 2>&1 \
; RUN:   | FileCheck %s --check-prefix=LOADED2 %}
;; No root is loaded when G's profile already has R nested, when the profile
;; has no root for R, or when G's call site does not record R.
; RUN: llvm-profdata merge --sample --extbinary %t/prof-nested.txt -o %t/prof-nested.extbinary
; RUN: sed '/^R:/,$d' %t/prof.txt | llvm-profdata merge --sample --extbinary - \
; RUN:   -o %t/prof-noroot.extbinary
; RUN: sed 's/^ 2: 1000 R:1000$/ 2: 1000/' %t/prof.txt | llvm-profdata merge --sample \
; RUN:   --extbinary - -o %t/prof-notarget.extbinary
; RUN: %{post} %t/post.ll -sample-profile-file=%t/prof-nested.extbinary \
; RUN:   -sample-profile-preserve-deferred-inlines 2>&1 | FileCheck %s --check-prefix=POST-ON
; RUN: %{post} %t/post.ll -sample-profile-file=%t/prof-noroot.extbinary \
; RUN:   -sample-profile-preserve-deferred-inlines 2>&1 \
; RUN:   | FileCheck %s --check-prefix=POST-OFF --allow-empty
; RUN: %{post} %t/post.ll -sample-profile-file=%t/prof-notarget.extbinary \
; RUN:   -sample-profile-preserve-deferred-inlines 2>&1 \
; RUN:   | FileCheck %s --check-prefix=POST-OFF --allow-empty
; RUN: %if asserts %{ %{post} %t/post.ll -sample-profile-file=%t/prof.extbinary \
; RUN:   -sample-profile-preserve-deferred-inlines -stats 2>&1 \
; RUN:   | FileCheck %s --check-prefix=LOADED1 %}
; RUN: %if asserts %{ %{post} %t/post.ll -sample-profile-file=%t/prof-nested.extbinary \
; RUN:   -sample-profile-preserve-deferred-inlines -stats 2>&1 \
; RUN:   | FileCheck %s --check-prefix=LOADED0 %}
; RUN: %if asserts %{ %{post} %t/post.ll -sample-profile-file=%t/prof-noroot.extbinary \
; RUN:   -sample-profile-preserve-deferred-inlines -stats 2>&1 \
; RUN:   | FileCheck %s --check-prefix=LOADED0 %}
; RUN: %if asserts %{ %{post} %t/post.ll -sample-profile-file=%t/prof-notarget.extbinary \
; RUN:   -sample-profile-preserve-deferred-inlines -stats 2>&1 \
; RUN:   | FileCheck %s --check-prefix=LOADED0 %}

;; ext's GUID is -2953423517159131221.
; PRE-OFF: define {{.*}} @G({{.*}} !prof ![[#G:]]
; PRE-OFF: ![[#G]] = !{!"function_entry_count", i64 1000}

; PRE-ON: define {{.*}} @G({{.*}} !prof ![[#G:]]
; PRE-ON-NOT: define {{.*}} @R(
; PRE-ON: ![[#G]] = !{!"function_entry_count", i64 1000, i64 -2953423517159131221}

; POST-OFF-NOT: 'ext' inlined into 'G'
; POST-ON: remark: a.c:2:0: 'ext' inlined into 'G' to match profiling context with (cost=always): preinliner
; CHAIN: remark: a.c:4:0: 'ext' inlined into 'G' to match profiling context with (cost=always): preinliner
; SHARED-DAG: remark: a.c:2:0: 'ext' inlined into 'G' {{.*}}preinliner
; SHARED-DAG: remark: a.c:5:0: 'ext2' inlined into 'G' {{.*}}preinliner
; LOADED2: 2 sample-profile - Number of orphaned root profiles loaded on demand
; LOADED1: 1 sample-profile - Number of orphaned root profiles loaded on demand
; LOADED0-NOT: orphaned root profiles

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

;--- prof-nested.txt
G:2000:1000
 1: 1000
 2: R:1000
  1: 1000
  2: ext:1000
   1: 1000
   !CFGChecksum: 3
   !Attributes: 2
  !CFGChecksum: 2
 !CFGChecksum: 1
R:2000:1000
 1: 1000
 !CFGChecksum: 2

;--- prof-chain.txt
G:2000:1000
 1: 1000
 2: 1000 R:1000
 !CFGChecksum: 1
R:2000:1000
 1: 1000
 2: 1000 S:1000
 !CFGChecksum: 2
S:2000:1000
 1: 1000
 2: ext:1000
  1: 1000
  !CFGChecksum: 3
  !Attributes: 2
 !CFGChecksum: 4

;--- prof-shared.txt
G:2000:1000
 1: 1000
 2: 1000 Q:500 R:500
 !CFGChecksum: 1
R:1000:500
 1: 500
 2: ext:500
  1: 500
  !CFGChecksum: 3
  !Attributes: 2
 !CFGChecksum: 2
Q:1000:500
 1: 500
 2: ext2:500
  1: 500
  !CFGChecksum: 6
  !Attributes: 2
 !CFGChecksum: 5

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

;--- post-chain.ll
;; G with local R and S inlined: G -> R -> S -> ext.
source_filename = "a.c"
target triple = "x86_64-unknown-linux-gnu"

define i64 @G(i64 %n) #0 !dbg !5 !guid !14 {
  call void @llvm.pseudoprobe(i64 -6455552227143004193, i64 1, i32 0, i64 -1), !dbg !6
  call void @llvm.pseudoprobe(i64 9084449867242398177, i64 1, i32 0, i64 -1), !dbg !15
  call void @llvm.pseudoprobe(i64 551554382388837469, i64 1, i32 0, i64 -1), !dbg !22
  %r = call i64 @ext(i64 %n), !dbg !23
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
!llvm.pseudo_probe_desc = !{!3, !4, !17, !24}

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
!16 = distinct !DILocation(line: 2, scope: !12, inlinedAt: !7)
!17 = !{i64 -2953423517159131221, i64 3, !"ext"}
!18 = distinct !DISubprogram(name: "ext", file: !1, line: 3, type: !13, spFlags: DISPFlagDefinition, unit: !0)
!19 = !{i64 -2953423517159131221}
!20 = !DILocation(line: 3, scope: !18)
!21 = distinct !DISubprogram(name: "S", file: !1, line: 4, type: !13, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0)
!22 = !DILocation(line: 4, scope: !21, inlinedAt: !16)
!23 = !DILocation(line: 4, scope: !25, inlinedAt: !16)
!24 = !{i64 551554382388837469, i64 4, !"S"}
!25 = !DILexicalBlockFile(scope: !21, file: !1, discriminator: 186646551)

;--- post-shared.ll
;; G with local R and Q inlined at the same site (!7).
source_filename = "a.c"
target triple = "x86_64-unknown-linux-gnu"

define i64 @G(i64 %n) #0 !dbg !5 !guid !14 {
  call void @llvm.pseudoprobe(i64 -6455552227143004193, i64 1, i32 0, i64 -1), !dbg !6
  call void @llvm.pseudoprobe(i64 9084449867242398177, i64 1, i32 0, i64 -1), !dbg !15
  %r = call i64 @ext(i64 %n), !dbg !16
  call void @llvm.pseudoprobe(i64 974280323357775344, i64 1, i32 0, i64 -1), !dbg !22
  %q = call i64 @ext2(i64 %r), !dbg !23
  ret i64 %q
}

define available_externally i64 @ext(i64 %n) #0 !dbg !18 !guid !19 {
  call void @llvm.pseudoprobe(i64 -2953423517159131221, i64 1, i32 0, i64 -1), !dbg !20
  ret i64 %n
}

define available_externally i64 @ext2(i64 %n) #0 !dbg !26 !guid !27 {
  call void @llvm.pseudoprobe(i64 3613183346563194775, i64 1, i32 0, i64 -1), !dbg !29
  ret i64 %n
}

declare void @llvm.pseudoprobe(i64, i64, i32, i64)

attributes #0 = { "use-sample-profile" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}
!llvm.pseudo_probe_desc = !{!3, !4, !17, !24, !28}

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
!21 = distinct !DISubprogram(name: "Q", file: !1, line: 5, type: !13, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0)
!22 = !DILocation(line: 5, scope: !21, inlinedAt: !7)
!23 = !DILocation(line: 5, scope: !25, inlinedAt: !7)
!24 = !{i64 974280323357775344, i64 5, !"Q"}
!25 = !DILexicalBlockFile(scope: !21, file: !1, discriminator: 186646551)
!26 = distinct !DISubprogram(name: "ext2", file: !1, line: 6, type: !13, spFlags: DISPFlagDefinition, unit: !0)
!27 = !{i64 3613183346563194775}
!28 = !{i64 3613183346563194775, i64 6, !"ext2"}
!29 = !DILocation(line: 6, scope: !26)
