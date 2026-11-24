;; ThinLTO post-link: G calls R, but G's profile records R only as a call
;; target, while R's top-level profile has ext inlined. Sample profile inlining
;; starts from R's top-level profile for the call, inlines R into G, and then
;; continues downward into the inlining recorded in R's profile, so ext is
;; inlined into G as well.

; RUN: split-file %s %t
; DEFINE: %{post} = opt %t/post.ll -passes='thinlto<O2>' -pgo-kind=pgo-sample-use-pipeline \
; DEFINE:   -pass-remarks=sample-profile-inline -o /dev/null
; RUN: llvm-profdata merge --sample --extbinary --gen-inlinee-top-level-map %t/prof.txt -o %t/prof.extbinary
; RUN: %{post} -sample-profile-file=%t/prof.txt 2>&1 | FileCheck %s --check-prefix=ON
; RUN: %{post} -sample-profile-file=%t/prof.extbinary 2>&1 | FileCheck %s --check-prefix=ON
; RUN: %{post} -sample-profile-file=%t/prof.txt -sample-profile-inline-from-top-level=false 2>&1 \
; RUN:   | FileCheck %s --check-prefix=OFF
;; Without a recorded call to R at the call site, there is nothing to start from.
; RUN: sed 's/^ 2: 1000 R:1000$/ 2: 1000/' %t/prof.txt > %t/prof-notarget.txt
; RUN: %{post} -sample-profile-file=%t/prof-notarget.txt 2>&1 | FileCheck %s --check-prefix=OFF
;; The inlined code of R is annotated with R's top-level profile, including
;; when it has no calls.
; RUN: %if asserts %{ %{post} -sample-profile-file=%t/prof.txt -debug-only=sample-profile,sample-profile-impl 2>&1 \
; RUN:   | FileCheck %s --check-prefix=ANNOTATE %}
; RUN: sed -e '/^define i64 @R/,/^}/{/@ext(/d;s/ret i64 %r/ret i64 %n/;}' %t/post.ll > %t/post-nocall.ll
; RUN: sed '/^ 2: ext:1000$/,/^  !Attributes: 2$/d' %t/prof.txt > %t/prof-nocall.txt
; RUN: %if asserts %{ opt %t/post-nocall.ll -passes='thinlto<O2>' -pgo-kind=pgo-sample-use-pipeline -o /dev/null \
; RUN:   -sample-profile-file=%t/prof-nocall.txt -debug-only=sample-profile,sample-profile-impl 2>&1 \
; RUN:   | FileCheck %s --check-prefix=ANNOTATE %}
;; Pre-link does not start from top-level profiles.
; RUN: opt %t/post.ll -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline -pass-remarks=sample-profile-inline \
; RUN:   -o /dev/null -sample-profile-file=%t/prof.txt 2>&1 | FileCheck %s --check-prefix=OFF

; ON-DAG: remark: a.c:1:0: 'R' inlined into 'G' to match profiling context {{.*}} at callsite G:0:0.2;
; ON-DAG: remark: a.c:2:0: 'ext' inlined into 'G' to match profiling context with (cost=always): preinliner at callsite R:0:0.2 @ G:0:0.2;
; ANNOTATE-LABEL: Processing Function G
; ANNOTATE: call void @llvm.pseudoprobe(i64 9084449867242398177, i64 1, i32 0, i64 -1){{.*}} - weight: 1000
; ANNOTATE-LABEL: Processing Function R
; OFF-NOT: inlined into 'G'
; OFF: 'ext' inlined into 'R' to match profiling context with (cost=always): preinliner
; OFF-NOT: inlined into 'G'

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

;--- post.ll
;; G's module after import, with ext imported.
source_filename = "a.c"
target triple = "x86_64-unknown-linux-gnu"

define i64 @G(i64 %n) #0 !dbg !5 {
  call void @llvm.pseudoprobe(i64 -6455552227143004193, i64 1, i32 0, i64 -1), !dbg !6
  %r = call i64 @R(i64 %n), !dbg !7
  ret i64 %r
}

define i64 @R(i64 %n) #0 !dbg !9 {
  call void @llvm.pseudoprobe(i64 9084449867242398177, i64 1, i32 0, i64 -1), !dbg !10
  %r = call i64 @ext(i64 %n), !dbg !11
  ret i64 %r
}

define available_externally i64 @ext(i64 %n) #0 !dbg !15 {
  call void @llvm.pseudoprobe(i64 -2953423517159131221, i64 1, i32 0, i64 -1), !dbg !16
  ret i64 %n
}

declare void @llvm.pseudoprobe(i64, i64, i32, i64)

attributes #0 = { "use-sample-profile" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}
!llvm.pseudo_probe_desc = !{!3, !4, !14}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "a.c", directory: "")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i64 -6455552227143004193, i64 1, !"G"}
!4 = !{i64 9084449867242398177, i64 2, !"R"}
!5 = distinct !DISubprogram(name: "G", file: !1, line: 1, type: !13, spFlags: DISPFlagDefinition, unit: !0)
!6 = !DILocation(line: 1, scope: !5)
;; Discriminator 186646551 encodes direct call probe 2.
!7 = !DILocation(line: 1, scope: !8)
!8 = !DILexicalBlockFile(scope: !5, file: !1, discriminator: 186646551)
!9 = distinct !DISubprogram(name: "R", file: !1, line: 2, type: !13, spFlags: DISPFlagDefinition, unit: !0)
!10 = !DILocation(line: 2, scope: !9)
!11 = !DILocation(line: 2, scope: !12)
!12 = !DILexicalBlockFile(scope: !9, file: !1, discriminator: 186646551)
!13 = !DISubroutineType(types: !{})
!14 = !{i64 -2953423517159131221, i64 3, !"ext"}
!15 = distinct !DISubprogram(name: "ext", file: !1, line: 3, type: !13, spFlags: DISPFlagDefinition, unit: !0)
!16 = !DILocation(line: 3, scope: !15)
