;; R is inlined into G and deleted in ThinLTO pre-link, while G's profile
;; records R only as a call target. In post-link, with ext imported, sample
;; profile inlining starts from R's top-level profile for R's inlined code, so
;; ext is inlined into G as in the profiled binary.

; RUN: split-file %s %t
; DEFINE: %{pre} = opt -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline
; DEFINE: %{post} = opt -passes='thinlto<O2>' -pgo-kind=pgo-sample-use-pipeline \
; DEFINE:   -pass-remarks=sample-profile-inline -o /dev/null
; RUN: llvm-profdata merge --sample --extbinary --gen-inlinee-top-level-map %t/profile.txt -o %t/profile.extbinary
; RUN: sed -e 's/define internal/define linkonce_odr/' -e 's/DISPFlagLocalToUnit | //' \
; RUN:   -e 's/#0 !dbg !9/alwaysinline #0 !dbg !9/' %t/input.ll > %t/nonlocal.ll

; RUN: %{pre} %t/input.ll -sample-profile-file=%t/profile.txt -S -o %t/pre.ll
; RUN: FileCheck %s --check-prefix=DELETED < %t/pre.ll
; RUN: llvm-link %t/pre.ll %t/ext.ll -o %t/post.bc
; RUN: %{post} %t/post.bc -sample-profile-file=%t/profile.txt 2>&1 | FileCheck %s --check-prefix=ON
; RUN: %{post} %t/post.bc -sample-profile-file=%t/profile.txt -sample-profile-inline-from-top-level=false 2>&1 \
; RUN:   | FileCheck %s --check-prefix=OFF --allow-empty

;; The same with an extbinary profile, whose reader loads R's top-level profile
;; through the inlinee-to-top-level map, as R contains the imported ext.
; RUN: %{pre} %t/input.ll -sample-profile-file=%t/profile.extbinary -o %t/pre.extbinary.bc
; RUN: llvm-link %t/pre.extbinary.bc %t/ext.ll -o %t/post.extbinary.bc
; RUN: %{post} %t/post.extbinary.bc -sample-profile-file=%t/profile.extbinary 2>&1 | FileCheck %s --check-prefix=ON

;; A non-local, discardable R.
; RUN: %{pre} %t/nonlocal.ll -sample-profile-file=%t/profile.txt -S -o %t/pre.nonlocal.ll
; RUN: FileCheck %s --check-prefix=DELETED < %t/pre.nonlocal.ll
; RUN: llvm-link %t/pre.nonlocal.ll %t/ext.ll -o %t/post.nonlocal.bc
; RUN: %{post} %t/post.nonlocal.bc -sample-profile-file=%t/profile.txt 2>&1 | FileCheck %s --check-prefix=ON
; RUN: %{post} %t/post.nonlocal.bc -sample-profile-file=%t/profile.extbinary 2>&1 | FileCheck %s --check-prefix=ON

; DELETED: define {{.*}} @G(
; DELETED-NOT: define {{.*}} @R(
; ON: remark: a.c:2:0: 'ext' inlined into 'G' to match profiling context with (cost=always): preinliner at callsite R:0:0.2 @ G:0:0.2;
; OFF-NOT: 'ext' inlined into 'G'

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
!8 = !DILexicalBlockFile(scope: !5, file: !1, discriminator: 186646551)
!9 = distinct !DISubprogram(name: "R", file: !1, line: 2, type: !13, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0)
!10 = !DILocation(line: 2, scope: !9)
!11 = !DILocation(line: 2, scope: !12)
!12 = !DILexicalBlockFile(scope: !9, file: !1, discriminator: 186646551)
!13 = !DISubroutineType(types: !{})
;--- ext.ll
;; ext as imported into G's module for post-link.
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define available_externally i64 @ext(i64 %n) #0 !dbg !3 {
  call void @llvm.pseudoprobe(i64 -2953423517159131221, i64 1, i32 0, i64 -1), !dbg !5
  ret i64 %n
}

declare void @llvm.pseudoprobe(i64, i64, i32, i64)

attributes #0 = { "use-sample-profile" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}
!llvm.pseudo_probe_desc = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "b.c", directory: "")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "ext", file: !1, line: 3, type: !4, spFlags: DISPFlagDefinition, unit: !0)
!4 = !DISubroutineType(types: !{})
!5 = !DILocation(line: 3, scope: !3)
!6 = !{i64 -2953423517159131221, i64 3, !"ext"}
