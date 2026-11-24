;; ThinLTO post-link: R was inlined into G and deleted in pre-link, while G's
;; profile records R only as a call target. The inlinee-to-top-level map makes
;; the reader load R's top-level profile (R contains ext, which is imported),
;; which is grafted into G's profile at the call site, so sample profile
;; inlining continues into it, and ext is inlined into G as in the profiled
;; binary.

; RUN: split-file %s %t
; RUN: cat %t/post.ll %t/common.ll > %t/local.ll
; RUN: sed 's/DISPFlagLocalToUnit | //' %t/local.ll > %t/nonlocal.ll
; RUN: cat %t/local.ll %t/imported-root.ll > %t/imported.ll
; RUN: sed 's/define available_externally i64 @R/define i64 @R/' %t/imported.ll > %t/retained.ll
; RUN: cat %t/post-chain.ll %t/common.ll > %t/chain.ll
; RUN: cat %t/post-shared.ll %t/common.ll > %t/shared.ll
;; R's ext call moves from probe 2 to probe 3 and R's checksum changes.
; RUN: sed -e 's/i64 2, !"R"/i64 9, !"R"/' -e 's/!12 = !DILexicalBlockFile(scope: !9, file: !1, discriminator: 186646551)/!12 = !DILexicalBlockFile(scope: !9, file: !1, discriminator: 186646559)/' %t/local.ll > %t/stale.ll
; DEFINE: %{post} = opt -passes='thinlto<O2>' -pgo-kind=pgo-sample-use-pipeline \
; DEFINE:   -pass-remarks=sample-profile-inline -o /dev/null
; DEFINE: %{merge} = llvm-profdata merge --sample --extbinary --gen-inlinee-top-level-map

; RUN: %{merge} %t/prof.txt -o %t/prof.extbinary
; RUN: llvm-profdata merge --sample --extbinary %t/prof.txt -o %t/prof-nomap.extbinary
; RUN: %{merge} --use-md5 %t/prof.txt -o %t/prof.md5
; RUN: %{merge} --use-md5 --md5-indexed-tables %t/prof.txt -o %t/prof.md5.eytzinger
; RUN: sed 's/^R:2000:1000$/R:2000:2000/' %t/prof.txt | %{merge} - -o %t/prof-multi.extbinary
; RUN: sed 's/^R:2000:1000$/R:2000:0/' %t/prof.txt | %{merge} - -o %t/prof-zero-head.extbinary
; RUN: sed 's/^ 2: 1000 R:1000$/ 2: 1000/' %t/prof.txt | %{merge} - -o %t/prof-notarget.extbinary
; RUN: sed 's/^ !CFGChecksum: 2$/ !CFGChecksum: 9/' %t/prof.txt | %{merge} - -o %t/prof-stale.extbinary
; RUN: sed '/^R:/,$d' %t/prof.txt | %{merge} - -o %t/prof-noroot.extbinary
; RUN: sed -n '/^R:/,$p' %t/prof.txt | %{merge} - -o %t/inner-only.extbinary
; RUN: %{merge} %t/prof-nested.txt -o %t/prof-nested.extbinary
; RUN: %{merge} %t/prof-chain.txt -o %t/prof-chain.extbinary
; RUN: %{merge} %t/prof-shared.txt -o %t/prof-shared.extbinary

;; Text profiles have all top-level profiles loaded; extbinary profiles load
;; R's through the map, also with MD5 names and the Eytzinger layout.
; RUN: %{post} %t/local.ll -sample-profile-file=%t/prof.txt 2>&1 | FileCheck %s --check-prefix=ON
; RUN: %{post} %t/local.ll -sample-profile-file=%t/prof.extbinary 2>&1 | FileCheck %s --check-prefix=ON
; RUN: %{post} %t/local.ll -sample-profile-file=%t/prof.md5 2>&1 | FileCheck %s --check-prefix=ON
; RUN: %{post} %t/local.ll -sample-profile-file=%t/prof.md5.eytzinger 2>&1 | FileCheck %s --check-prefix=ON
; RUN: %{post} %t/nonlocal.ll -sample-profile-file=%t/prof.txt 2>&1 | FileCheck %s --check-prefix=ON
; RUN: %{post} %t/nonlocal.ll -sample-profile-file=%t/prof.extbinary 2>&1 | FileCheck %s --check-prefix=ON
;; Whether R's body is imported or still emitted, its inlined copy in G starts
;; from R's top-level profile.
; RUN: %{post} %t/imported.ll -sample-profile-file=%t/prof.txt 2>&1 | FileCheck %s --check-prefix=ON
; RUN: %{post} %t/retained.ll -sample-profile-file=%t/prof.txt 2>&1 | FileCheck %s --check-prefix=ON
;; The top-level profile is used however much of it the call site accounts
;; for, and whether or not G records the call.
; RUN: %{post} %t/local.ll -sample-profile-file=%t/prof-multi.extbinary 2>&1 | FileCheck %s --check-prefix=ON
; RUN: %{post} %t/local.ll -sample-profile-file=%t/prof-zero-head.extbinary 2>&1 | FileCheck %s --check-prefix=ON
; RUN: %{post} %t/local.ll -sample-profile-file=%t/prof-notarget.extbinary 2>&1 | FileCheck %s --check-prefix=ON
;; G has no profile, so it is not processed.
; RUN: %{post} %t/local.ll -sample-profile-file=%t/inner-only.extbinary 2>&1 | FileCheck %s --check-prefix=OFF --allow-empty
;; G's profile already has R inlined.
; RUN: %{post} %t/local.ll -sample-profile-file=%t/prof-nested.extbinary 2>&1 | FileCheck %s --check-prefix=ON

;; Disabled, without the map R's profile is not loaded, and without R's
;; profile there is nothing to graft.
; RUN: %{post} %t/local.ll -sample-profile-file=%t/prof-nomap.extbinary 2>&1 | FileCheck %s --check-prefix=OFF --allow-empty
; RUN: %{post} %t/local.ll -sample-profile-file=%t/prof.extbinary -sample-profile-inline-from-top-level=false 2>&1 \
; RUN:   | FileCheck %s --check-prefix=OFF --allow-empty
; RUN: %{post} %t/local.ll -sample-profile-file=%t/prof.txt -sample-profile-inline-from-top-level=false 2>&1 \
; RUN:   | FileCheck %s --check-prefix=OFF --allow-empty
; RUN: %{post} %t/local.ll -sample-profile-file=%t/prof-noroot.extbinary 2>&1 | FileCheck %s --check-prefix=OFF --allow-empty
;; Pre-link does not start from top-level profiles.
; RUN: opt -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline -pass-remarks=sample-profile-inline \
; RUN:   -o /dev/null %t/local.ll -sample-profile-file=%t/prof.extbinary 2>&1 | FileCheck %s --check-prefix=OFF --allow-empty

;; Like an inlined profile, a grafted top-level profile is used without a
;; checksum check. Stale inlined code is not matched against it.
; RUN: %{post} %t/local.ll -sample-profile-file=%t/prof-stale.extbinary 2>&1 | FileCheck %s --check-prefix=ON
; RUN: %{post} %t/stale.ll -sample-profile-file=%t/prof.txt 2>&1 | FileCheck %s --check-prefix=OFF --allow-empty

;; G -> R -> S -> ext: the map relates only S, which contains ext, to the
;; module, so R's frame gets an empty profile and S's top-level profile is
;; grafted under it. R and Q, inlined at the same site in G, are grafted side
;; by side.
; RUN: %{post} %t/chain.ll -sample-profile-file=%t/prof-chain.extbinary 2>&1 | FileCheck %s --check-prefix=CHAIN
; RUN: %{post} %t/shared.ll -sample-profile-file=%t/prof-shared.extbinary 2>&1 | FileCheck %s --check-prefix=SHARED

; ON: remark: a.c:2:0: 'ext' inlined into 'G' to match profiling context with (cost=always): preinliner at callsite R:0:0.2 @ G:0:0.2;
; OFF-NOT: 'ext' inlined into 'G'
; CHAIN: remark: a.c:4:0: 'ext' inlined into 'G' to match profiling context with (cost=always): preinliner at callsite S:0:0.2 @ R:0:0.2 @ G:0:0.2;
; SHARED-DAG: remark: a.c:2:0: 'ext' inlined into 'G' {{.*}}preinliner
; SHARED-DAG: remark: a.c:5:0: 'ext2' inlined into 'G' {{.*}}preinliner

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

!llvm.pseudo_probe_desc = !{!3, !4, !17}
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

!llvm.pseudo_probe_desc = !{!3, !4, !17, !24}
;--- post-shared.ll
;; G with local R and Q inlined at the same site (!7).
source_filename = "a.c"
target triple = "x86_64-unknown-linux-gnu"

define i64 @G(i64 %n) #0 !dbg !5 !guid !14 {
  call void @llvm.pseudoprobe(i64 -6455552227143004193, i64 1, i32 0, i64 -1), !dbg !6
  call void @llvm.pseudoprobe(i64 9084449867242398177, i64 1, i32 0, i64 -1), !dbg !15
  %r = call i64 @ext(i64 %n), !dbg !16
  call void @llvm.pseudoprobe(i64 974280323357775344, i64 1, i32 0, i64 -1), !dbg !27
  %q = call i64 @ext2(i64 %r), !dbg !29
  ret i64 %q
}

define available_externally i64 @ext2(i64 %n) #0 !dbg !31 !guid !32 {
  call void @llvm.pseudoprobe(i64 3613183346563194775, i64 1, i32 0, i64 -1), !dbg !34
  ret i64 %n
}

!llvm.pseudo_probe_desc = !{!3, !4, !17, !30, !33}
;--- common.ll
define available_externally i64 @ext(i64 %n) #0 !dbg !18 !guid !19 {
  call void @llvm.pseudoprobe(i64 -2953423517159131221, i64 1, i32 0, i64 -1), !dbg !20
  ret i64 %n
}

declare void @llvm.pseudoprobe(i64, i64, i32, i64)

attributes #0 = { "use-sample-profile" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

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
!10 = !DILocation(line: 2, scope: !9)
!11 = !DILocation(line: 2, scope: !12)
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
!26 = distinct !DISubprogram(name: "Q", file: !1, line: 5, type: !13, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0)
!27 = !DILocation(line: 5, scope: !26, inlinedAt: !7)
!28 = !DILexicalBlockFile(scope: !26, file: !1, discriminator: 186646551)
!29 = !DILocation(line: 5, scope: !28, inlinedAt: !7)
!30 = !{i64 974280323357775344, i64 5, !"Q"}
!31 = distinct !DISubprogram(name: "ext2", file: !1, line: 6, type: !13, spFlags: DISPFlagDefinition, unit: !0)
!32 = !{i64 3613183346563194775}
!33 = !{i64 3613183346563194775, i64 6, !"ext2"}
!34 = !DILocation(line: 6, scope: !31)
;--- imported-root.ll
define available_externally i64 @R(i64 %n) #0 !dbg !9 {
  call void @llvm.pseudoprobe(i64 9084449867242398177, i64 1, i32 0, i64 -1), !dbg !10
  %r = call i64 @ext(i64 %n), !dbg !11
  ret i64 %r
}
