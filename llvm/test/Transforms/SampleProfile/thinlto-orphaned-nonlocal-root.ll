; -sample-profile-preserve-orphaned-inlines handles local and non-local roots
; and is enabled by default. R below is a linkonce_odr function that pre-link
; inlines into H. H is defined in another module and imported into G's module.
; G's profile has H nested, and H's frame calls R only as a call target, so R's
; samples are in R's top-level profile.

;; Pre-link: G has no samples, so its entry count is -1. R's import GUIDs are
;; still copied to G.
; RUN: split-file %s %t
; DEFINE: %{pre} = opt %t/pre.ll -passes='thinlto-pre-link<O2>' \
; DEFINE:   -pgo-kind=pgo-sample-use-pipeline -sample-profile-file=%t/prof-pre.txt -S
; RUN: %{pre} -sample-profile-preserve-orphaned-inlines=false | FileCheck %s --check-prefix=PRE-OFF
; RUN: %{pre} | FileCheck %s --check-prefix=PRE-ON

;; Post-link: H has no top-level profile, so finding R's root needs a scan of
;; H's body from G's nested profile. R has no probe descriptor in G's module,
;; as import does not copy descriptors.
; RUN: llvm-profdata merge --sample --extbinary %t/prof.txt -o %t/prof.extbinary
; DEFINE: %{post} = opt %t/post.ll -passes='thinlto<O2>' -pgo-kind=pgo-sample-use-pipeline \
; DEFINE:   -sample-profile-file=%t/prof.extbinary \
; DEFINE:   -pass-remarks=sample-profile-inline -o /dev/null
; RUN: %{post} -sample-profile-preserve-orphaned-inlines=false 2>&1 | FileCheck %s --check-prefix=POST-OFF
; RUN: %{post} 2>&1 \
; RUN:   | FileCheck %s --check-prefix=POST-ON
;; The call-site check still applies.
; RUN: sed 's/^R:2000:1000$/R:2000:2000/' %t/prof.txt | llvm-profdata merge --sample \
; RUN:   --extbinary - -o %t/prof-multi.extbinary
; RUN: %{post} -sample-profile-file=%t/prof-multi.extbinary 2>&1 | FileCheck %s --check-prefix=POST-OFF

; PRE-OFF: define {{.*}} @G({{.*}} !prof ![[#G:]]
; PRE-OFF: ![[#G]] = !{!"function_entry_count", i64 -1}
; PRE-ON: define {{.*}} @G({{.*}} !prof ![[#G:]]
; PRE-ON: ![[#G]] = !{!"function_entry_count", i64 -1, i64 -2953423517159131221}

; POST-OFF: 'H' inlined into 'G'
; POST-OFF-NOT: 'ext' inlined into 'G'
; POST-ON: 'H' inlined into 'G'
; POST-ON: remark: a.c:2:0: 'ext' inlined into 'G' to match profiling context with (cost=always): preinliner

;--- prof-pre.txt
R:2000:1000
 1: 1000
 2: ext:1000
  1: 1000
  !CFGChecksum: 3
  !Attributes: 2
 !CFGChecksum: 2

;--- prof.txt
G:3000:1000
 1: 1000
 2: H:2000
  1: 1000
  2: 1000 R:1000
  !CFGChecksum: 4
  !Attributes: 2
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

define linkonce_odr i64 @R(i64 %n) #0 !dbg !9 {
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
!9 = distinct !DISubprogram(name: "R", file: !1, line: 2, type: !13, spFlags: DISPFlagDefinition, unit: !0)
!10 = !DILocation(line: 2, scope: !9)
!11 = !DILocation(line: 2, scope: !12)
!12 = !DILexicalBlockFile(scope: !9, file: !1, discriminator: 186646551)
!13 = !DISubroutineType(types: !{})

;--- post.ll
;; G's module after import: H is imported with R inlined into it, and ext is
;; imported. Only G and ext have probe descriptors.
source_filename = "a.c"
target triple = "x86_64-unknown-linux-gnu"

define i64 @G(i64 %n) #0 !dbg !5 !guid !14 {
  call void @llvm.pseudoprobe(i64 -6455552227143004193, i64 1, i32 0, i64 -1), !dbg !6
  %r = call i64 @H(i64 %n), !dbg !7
  ret i64 %r
}

define available_externally i64 @H(i64 %n) #0 !dbg !21 !guid !22 {
  call void @llvm.pseudoprobe(i64 1898973705195084225, i64 1, i32 0, i64 -1), !dbg !23
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
!llvm.pseudo_probe_desc = !{!3, !17}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "a.c", directory: "")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i64 -6455552227143004193, i64 1, !"G"}
!5 = distinct !DISubprogram(name: "G", file: !1, line: 1, type: !13, spFlags: DISPFlagDefinition, unit: !0)
!6 = !DILocation(line: 1, scope: !5)
!7 = !DILocation(line: 1, scope: !8)
!8 = !DILexicalBlockFile(scope: !5, file: !1, discriminator: 186646551)
!9 = distinct !DISubprogram(name: "R", file: !1, line: 2, type: !13, spFlags: DISPFlagDefinition, unit: !0)
!12 = !DILexicalBlockFile(scope: !9, file: !1, discriminator: 186646551)
!13 = !DISubroutineType(types: !{})
!14 = !{i64 -6455552227143004193}
!15 = !DILocation(line: 2, scope: !9, inlinedAt: !24)
!16 = !DILocation(line: 2, scope: !12, inlinedAt: !24)
!17 = !{i64 -2953423517159131221, i64 3, !"ext"}
!18 = distinct !DISubprogram(name: "ext", file: !1, line: 3, type: !13, spFlags: DISPFlagDefinition, unit: !0)
!19 = !{i64 -2953423517159131221}
!20 = !DILocation(line: 3, scope: !18)
!21 = distinct !DISubprogram(name: "H", file: !1, line: 4, type: !13, spFlags: DISPFlagDefinition, unit: !0)
!22 = !{i64 1898973705195084225}
!23 = !DILocation(line: 4, scope: !21)
!24 = distinct !DILocation(line: 4, scope: !25)
!25 = !DILexicalBlockFile(scope: !21, file: !1, discriminator: 186646551)
