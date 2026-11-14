;; Test that ThinLTO backends locate synthesized CallsiteInfo records and
;; perform MemProf ICP for missing indirect tail call frames when profiled
;; targets in VP metadata include local (internal) aliases and their local
;; aliasees (whose VP candidate GUIDs are unprefixed original name GUIDs).
;;
;; Covers both in-process ThinLTO (llvm-lto2) and distributed ThinLTO backends
;; across four scenarios where main.o calls noinline callers on notcold+cold
;; contexts (emitting U <caller>.memprof.1):
;; 1. baz.ll (baz.o): @baz_imp has an indirect tail call to @local_alias, which
;;    is promoted and imported into baz.o as an available_externally definition
;;    via replaceAliasWithAliasee (retaining @local_aliasee's !guid metadata and
;;    recording the alias GUID in !thinlto_alias_guid).
;; 2. b.ll (b.o): @baz_same has an indirect tail call to @local_alias in the
;;    same module where @local_alias is a GlobalAlias.
;; 3. both.ll (both.o): @both_via_alias and @both_via_aliasee have indirect tail
;;    calls to @local_alias and @local_aliasee respectively, and both targets
;;    are imported into both.o as available_externally definitions.
;; 4. decl.ll (decl.o): @decl_via_alias (hot) imports @local_alias as an
;;    available_externally definition before @decl_via_aliasee (cold) imports
;;    @local_aliasee as a declaration, verifying that @local_alias.llvm.* (which
;;    retains @local_aliasee's !guid) does not shadow @local_aliasee.llvm.* in
;;    InstrProfSymtab.

; REQUIRES: asserts

; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: opt -thinlto-bc main.ll -o main.o
; RUN: opt -thinlto-bc baz.ll -o baz.o
; RUN: opt -thinlto-bc b.ll -o b.o
; RUN: opt -thinlto-bc both.ll -o both.o
; RUN: opt -thinlto-bc decl.ll -o decl.o

;; 1. In-process ThinLTO via llvm-lto2: verify .memprof.1 clones are created
;; and all cross-module symbol references are satisfied.
; RUN: llvm-lto2 run main.o baz.o b.o both.o decl.o \
; RUN:  -enable-memprof-context-disambiguation \
; RUN:  -enable-memprof-indirect-call-support=true \
; RUN:  -supports-hot-cold-new \
; RUN:  -r=main.o,main,plx \
; RUN:  -r=main.o,baz_imp, \
; RUN:  -r=main.o,baz_same, \
; RUN:  -r=main.o,both_via_alias, \
; RUN:  -r=main.o,both_via_aliasee, \
; RUN:  -r=main.o,decl_via_alias, \
; RUN:  -r=main.o,decl_via_aliasee, \
; RUN:  -r=main.o,leaf,plx \
; RUN:  -r=main.o,_Znam, \
; RUN:  -r=baz.o,baz_imp,plx \
; RUN:  -r=b.o,baz_same,plx \
; RUN:  -r=b.o,leaf, \
; RUN:  -r=both.o,both_via_alias,plx \
; RUN:  -r=both.o,both_via_aliasee,plx \
; RUN:  -r=decl.o,decl_via_alias,plx \
; RUN:  -r=decl.o,decl_via_aliasee,plx \
; RUN:  -thinlto-threads=1 \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes \
; RUN:  -pass-remarks=memprof-context-disambiguation \
; RUN:  -o inproc.out 2>&1 | FileCheck %s --check-prefix=REMARKS-INPROC
; RUN: llvm-nm inproc.out.1 | FileCheck %s --check-prefix=NM-MAIN
; RUN: llvm-nm inproc.out.2 | FileCheck %s --check-prefix=NM-BAZ
; RUN: llvm-nm inproc.out.3 | FileCheck %s --check-prefix=NM-B
; RUN: llvm-nm inproc.out.4 | FileCheck %s --check-prefix=NM-BOTH
; RUN: llvm-nm inproc.out.5 | FileCheck %s --check-prefix=NM-DECL

; REMARKS-INPROC-DAG: created clone baz_imp.memprof.1
; REMARKS-INPROC-DAG: created clone baz_same.memprof.1
; REMARKS-INPROC-DAG: created clone both_via_alias.memprof.1
; REMARKS-INPROC-DAG: created clone both_via_aliasee.memprof.1
; REMARKS-INPROC-DAG: created clone decl_via_alias.memprof.1
; REMARKS-INPROC-DAG: created clone decl_via_aliasee.memprof.1

; NM-MAIN-DAG: U baz_imp.memprof.1
; NM-MAIN-DAG: U baz_same.memprof.1
; NM-MAIN-DAG: U both_via_alias.memprof.1
; NM-MAIN-DAG: U both_via_aliasee.memprof.1
; NM-MAIN-DAG: U decl_via_alias.memprof.1
; NM-MAIN-DAG: U decl_via_aliasee.memprof.1

; NM-BAZ-DAG: T baz_imp{{$}}
; NM-BAZ-DAG: T baz_imp.memprof.1{{$}}
; NM-BAZ-DAG: U local_alias.llvm.{{[0-9]+}}.memprof.1{{$}}

; NM-B-DAG: T baz_same{{$}}
; NM-B-DAG: T baz_same.memprof.1{{$}}
; NM-B-DAG: {{[Tt]}} local_aliasee.llvm.{{[0-9]+}}.memprof.1{{$}}
; NM-B-DAG: {{[Tt]}} local_alias.llvm.{{[0-9]+}}.memprof.1{{$}}

; NM-BOTH-DAG: T both_via_alias{{$}}
; NM-BOTH-DAG: T both_via_alias.memprof.1{{$}}
; NM-BOTH-DAG: T both_via_aliasee{{$}}
; NM-BOTH-DAG: T both_via_aliasee.memprof.1{{$}}
; NM-BOTH-DAG: U local_alias.llvm.{{[0-9]+}}.memprof.1{{$}}
; NM-BOTH-DAG: U local_aliasee.llvm.{{[0-9]+}}{{$}}
; NM-BOTH-DAG: U leaf.memprof.1{{$}}

; NM-DECL-DAG: T decl_via_alias{{$}}
; NM-DECL-DAG: T decl_via_alias.memprof.1{{$}}
; NM-DECL-DAG: T decl_via_aliasee{{$}}
; NM-DECL-DAG: T decl_via_aliasee.memprof.1{{$}}
; NM-DECL-DAG: U local_alias.llvm.{{[0-9]+}}.memprof.1{{$}}
; NM-DECL-DAG: U local_aliasee.llvm.{{[0-9]+}}.memprof.1{{$}}

;; 2. Distributed ThinLTO thin link.
; RUN: llvm-lto2 run main.o baz.o b.o both.o decl.o \
; RUN:  -enable-memprof-context-disambiguation \
; RUN:  -enable-memprof-indirect-call-support=true \
; RUN:  -supports-hot-cold-new \
; RUN:  -thinlto-distributed-indexes \
; RUN:  -r=main.o,main,plx \
; RUN:  -r=main.o,baz_imp, \
; RUN:  -r=main.o,baz_same, \
; RUN:  -r=main.o,both_via_alias, \
; RUN:  -r=main.o,both_via_aliasee, \
; RUN:  -r=main.o,decl_via_alias, \
; RUN:  -r=main.o,decl_via_aliasee, \
; RUN:  -r=main.o,leaf,plx \
; RUN:  -r=main.o,_Znam, \
; RUN:  -r=baz.o,baz_imp,plx \
; RUN:  -r=b.o,baz_same,plx \
; RUN:  -r=b.o,leaf, \
; RUN:  -r=both.o,both_via_alias,plx \
; RUN:  -r=both.o,both_via_aliasee,plx \
; RUN:  -r=decl.o,decl_via_alias,plx \
; RUN:  -r=decl.o,decl_via_aliasee,plx \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes \
; RUN:  -stats -o distrib.out 2>&1 | FileCheck %s --check-prefix=STATS-LINK

; STATS-LINK: 12 memprof-context-disambiguation - Number of profiled callees found via tail calls
; STATS-LINK: 12 memprof-context-disambiguation - Number of tail calls in chains to profiled callees that are indirect (thin link)

;; Verify main.o backend references all six .memprof.1 clones.
; RUN: opt -import-all-index -passes=function-import,memprof-context-disambiguation \
; RUN:  -summary-file=main.o.thinlto.bc -memprof-import-summary=main.o.thinlto.bc \
; RUN:  -enable-import-metadata \
; RUN:  main.o -S -o distrib.main.ll
; RUN: FileCheck %s --check-prefix=IR-MAIN < distrib.main.ll

; IR-MAIN-DAG: call ptr @baz_imp.memprof.1(ptr null)
; IR-MAIN-DAG: call ptr @baz_same.memprof.1(ptr null)
; IR-MAIN-DAG: call ptr @both_via_alias.memprof.1(ptr null)
; IR-MAIN-DAG: call ptr @both_via_aliasee.memprof.1(ptr null)
; IR-MAIN-DAG: call ptr @decl_via_alias.memprof.1(ptr null)
; IR-MAIN-DAG: call ptr @decl_via_aliasee.memprof.1(ptr null)

;; Scenario 1: Distributed ThinLTO backend for baz.o (@local_alias promoted and
;; imported as an available_externally definition via replaceAliasWithAliasee).
;; Verify @local_alias.llvm.* retains @local_aliasee's !guid
;; (5693215144062012398) and records MD5("b.cc;local_alias")
;; (-592420616430196302) in !thinlto_alias_guid, is NOT cloned in baz.o, and
;; @baz_imp.memprof.1 IS created and promoted to call @local_alias.llvm.*.memprof.1.
; RUN: opt -import-all-index -passes=function-import,memprof-context-disambiguation \
; RUN:  -enable-memprof-indirect-call-support=true \
; RUN:  -summary-file=baz.o.thinlto.bc -memprof-import-summary=baz.o.thinlto.bc \
; RUN:  -enable-import-metadata \
; RUN:  -pass-remarks=memprof-context-disambiguation \
; RUN:  -stats baz.o -S -o distrib.baz.ll 2>&1 | FileCheck %s --check-prefix=REMARKS-BAZ \
; RUN:  --implicit-check-not="created clone local_alias"
; RUN: FileCheck %s --check-prefix=IR-BAZ < distrib.baz.ll

; REMARKS-BAZ: created clone baz_imp.memprof.1
; REMARKS-BAZ: call in clone baz_imp.memprof.1 promoted and assigned to call function clone local_alias.llvm.{{[0-9]+}}.memprof.1
; REMARKS-BAZ: 1 memprof-context-disambiguation - Number of indirect tail calls recorded for ICP for cloning during ThinLTO backend

; IR-BAZ-NOT: define {{.*}}@local_alias.llvm.{{[0-9]+}}.memprof.1
; IR-BAZ: define dso_local ptr @baz_imp(ptr %fp)
; IR-BAZ: define available_externally hidden ptr @local_alias.llvm.{{[0-9]+}}() !guid ![[BAZ_GUID:[0-9]+]] !thinlto_src_module !{{[0-9]+}} !thinlto_src_file !{{[0-9]+}} !thinlto_alias_guid ![[BAZ_ALIAS_GUID:[0-9]+]] {
; IR-BAZ: define dso_local ptr @baz_imp.memprof.1(ptr %fp)
; IR-BAZ:   call ptr @local_alias.llvm.{{[0-9]+}}.memprof.1()
; IR-BAZ-NOT: define {{.*}}@local_alias.llvm.{{[0-9]+}}.memprof.1
; IR-BAZ-DAG: ![[BAZ_GUID]] = !{i64 5693215144062012398}
; IR-BAZ-DAG: ![[BAZ_ALIAS_GUID]] = !{i64 -592420616430196302}

;; Scenario 2: Distributed ThinLTO backend for b.o (@local_alias is a
;; GlobalAlias in the same module as @baz_same). Verify @baz_same.memprof.1 and
;; both promoted target clones are created.
; RUN: opt -import-all-index -passes=function-import,memprof-context-disambiguation \
; RUN:  -enable-memprof-indirect-call-support=true \
; RUN:  -summary-file=b.o.thinlto.bc -memprof-import-summary=b.o.thinlto.bc \
; RUN:  -enable-import-metadata \
; RUN:  -pass-remarks=memprof-context-disambiguation \
; RUN:  -stats b.o -S -o distrib.b.ll 2>&1 | FileCheck %s --check-prefix=REMARKS-B
; RUN: FileCheck %s --check-prefix=IR-B < distrib.b.ll

; REMARKS-B-DAG: created clone local_aliasee.llvm.{{[0-9]+}}.memprof.1
; REMARKS-B-DAG: created clone baz_same.memprof.1
; REMARKS-B: 1 memprof-context-disambiguation - Number of indirect tail calls recorded for ICP for cloning during ThinLTO backend

; IR-B-DAG: define hidden ptr @local_aliasee.llvm.{{[0-9]+}}.memprof.1()
; IR-B-DAG: @local_alias.llvm.{{[0-9]+}}.memprof.1 = hidden alias ptr (), ptr @local_aliasee.llvm.{{[0-9]+}}.memprof.1
; IR-B-DAG: define dso_local ptr @baz_same.memprof.1(ptr %fp)

;; Scenario 3: Distributed ThinLTO backend for both.o (both @local_aliasee and
;; @local_alias imported as available_externally definitions).
; RUN: opt -import-all-index -passes=function-import,memprof-context-disambiguation \
; RUN:  -enable-memprof-indirect-call-support=true \
; RUN:  -summary-file=both.o.thinlto.bc -memprof-import-summary=both.o.thinlto.bc \
; RUN:  -enable-import-metadata \
; RUN:  -pass-remarks=memprof-context-disambiguation \
; RUN:  -stats both.o -S -o distrib.both.ll 2>&1 | FileCheck %s --check-prefix=REMARKS-BOTH
; RUN: FileCheck %s --check-prefix=IR-BOTH < distrib.both.ll

; REMARKS-BOTH-DAG: created clone both_via_alias.memprof.1
; REMARKS-BOTH-DAG: call in clone both_via_alias.memprof.1 promoted and assigned to call function clone local_alias.llvm.{{[0-9]+}}.memprof.1
; REMARKS-BOTH-DAG: created clone both_via_aliasee.memprof.1
; REMARKS-BOTH-DAG: call in clone both_via_aliasee.memprof.1 promoted and assigned to call function clone local_aliasee.llvm.{{[0-9]+}}.memprof.1
; REMARKS-BOTH: 2 memprof-context-disambiguation - Number of indirect tail calls recorded for ICP for cloning during ThinLTO backend

; IR-BOTH: define available_externally hidden ptr @local_aliasee.llvm.{{[0-9]+}}() !guid ![[BOTH_GUID:[0-9]+]]
; IR-BOTH: define available_externally hidden ptr @local_alias.llvm.{{[0-9]+}}() !guid ![[BOTH_GUID]] {{.*}}!thinlto_alias_guid ![[BOTH_ALIAS_GUID:[0-9]+]]
; IR-BOTH: define dso_local ptr @both_via_alias.memprof.1(ptr %fp)
; IR-BOTH:   call ptr @local_alias.llvm.{{[0-9]+}}.memprof.1()
; IR-BOTH: define dso_local ptr @both_via_aliasee.memprof.1(ptr %fp)
; IR-BOTH:   call ptr @local_aliasee.llvm.{{[0-9]+}}.memprof.1()
; IR-BOTH-DAG: ![[BOTH_GUID]] = !{i64 5693215144062012398}
; IR-BOTH-DAG: ![[BOTH_ALIAS_GUID]] = !{i64 -592420616430196302}

;; Scenario 4: Distributed ThinLTO backend for decl.o (@local_alias imported
;; first as an available_externally definition, @local_aliasee imported second
;; as a declaration). Verify @local_alias.llvm.* precedes @local_aliasee.llvm.*
;; in module order and does not shadow @local_aliasee.llvm.* in InstrProfSymtab.
; RUN: llvm-dis decl.o.thinlto.bc -o - | FileCheck %s --check-prefix=INDEX-DECL
; INDEX-DECL-DAG: gv: (guid: 17854323457279355314, summaries: (alias: ({{.*}}importType: definition
; INDEX-DECL-DAG: gv: (guid: 5693215144062012398, summaries: (function: ({{.*}}importType: declaration

; RUN: opt -import-all-index -passes=function-import,memprof-context-disambiguation \
; RUN:  -enable-memprof-indirect-call-support=true \
; RUN:  -summary-file=decl.o.thinlto.bc -memprof-import-summary=decl.o.thinlto.bc \
; RUN:  -enable-import-metadata \
; RUN:  -pass-remarks=memprof-context-disambiguation \
; RUN:  -stats decl.o -S -o distrib.decl.ll 2>&1 | FileCheck %s --check-prefix=REMARKS-DECL
; RUN: FileCheck %s --check-prefix=IR-DECL < distrib.decl.ll

; REMARKS-DECL-DAG: created clone decl_via_alias.memprof.1
; REMARKS-DECL-DAG: call in clone decl_via_alias.memprof.1 promoted and assigned to call function clone local_alias.llvm.{{[0-9]+}}.memprof.1
; REMARKS-DECL-DAG: created clone decl_via_aliasee.memprof.1
; REMARKS-DECL-DAG: call in clone decl_via_aliasee.memprof.1 promoted and assigned to call function clone local_aliasee.llvm.{{[0-9]+}}.memprof.1
; REMARKS-DECL: 2 memprof-context-disambiguation - Number of indirect tail calls recorded for ICP for cloning during ThinLTO backend

; IR-DECL: define available_externally hidden ptr @local_alias.llvm.{{[0-9]+}}() !guid ![[DECL_GUID:[0-9]+]] {{.*}}!thinlto_alias_guid ![[DECL_ALIAS_GUID:[0-9]+]]
; IR-DECL: declare !guid ![[DECL_GUID]] hidden ptr @local_aliasee.llvm.{{[0-9]+}}()
; IR-DECL: define dso_local ptr @decl_via_alias.memprof.1(ptr %fp)
; IR-DECL:   call ptr @local_alias.llvm.{{[0-9]+}}.memprof.1()
; IR-DECL: define dso_local ptr @decl_via_aliasee.memprof.1(ptr %fp)
; IR-DECL:   call ptr @local_aliasee.llvm.{{[0-9]+}}.memprof.1()
; IR-DECL-DAG: ![[DECL_GUID]] = !{i64 5693215144062012398}
; IR-DECL-DAG: ![[DECL_ALIAS_GUID]] = !{i64 -592420616430196302}

;--- main.ll
source_filename = "main.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main() {
entry:
  %c1 = call ptr @baz_imp(ptr null), !callsite !101
  %c2 = call ptr @baz_imp(ptr null), !callsite !102
  %c3 = call ptr @baz_same(ptr null), !callsite !103
  %c4 = call ptr @baz_same(ptr null), !callsite !104
  %c5 = call ptr @both_via_alias(ptr null), !callsite !105
  %c6 = call ptr @both_via_alias(ptr null), !callsite !106
  %c7 = call ptr @both_via_aliasee(ptr null), !callsite !107
  %c8 = call ptr @both_via_aliasee(ptr null), !callsite !108
  %c9 = call ptr @decl_via_alias(ptr null), !callsite !109
  %c10 = call ptr @decl_via_alias(ptr null), !callsite !110
  %c11 = call ptr @decl_via_aliasee(ptr null), !callsite !111
  %c12 = call ptr @decl_via_aliasee(ptr null), !callsite !112
  ret i32 0
}

declare ptr @baz_imp(ptr)
declare ptr @baz_same(ptr)
declare ptr @both_via_alias(ptr)
declare ptr @both_via_aliasee(ptr)
declare ptr @decl_via_alias(ptr)
declare ptr @decl_via_aliasee(ptr)

define ptr @leaf() #0 {
entry:
  %call = tail call ptr @_Znam(i64 10) #1, !memprof !1, !callsite !5
  ret ptr %call
}

declare ptr @_Znam(i64)

attributes #0 = { noinline }
attributes #1 = { builtin allocsize(0) }

!1 = !{!11, !12, !13, !14, !15, !16, !17, !18, !19, !20, !21, !22}
!5 = !{i64 3186456655321080972}
!11 = !{!31, !"notcold"}
!12 = !{!32, !"cold"}
!13 = !{!33, !"notcold"}
!14 = !{!34, !"cold"}
!15 = !{!35, !"notcold"}
!16 = !{!36, !"cold"}
!17 = !{!37, !"notcold"}
!18 = !{!38, !"cold"}
!19 = !{!39, !"notcold"}
!20 = !{!40, !"cold"}
!21 = !{!41, !"notcold"}
!22 = !{!42, !"cold"}
!31 = !{i64 3186456655321080972, i64 8632435727821051414}
!32 = !{i64 3186456655321080972, i64 -3421689549917153178}
!33 = !{i64 3186456655321080972, i64 1234567890123456789}
!34 = !{i64 3186456655321080972, i64 2345678901234567890}
!35 = !{i64 3186456655321080972, i64 3456789012345678901}
!36 = !{i64 3186456655321080972, i64 4567890123456789012}
!37 = !{i64 3186456655321080972, i64 5678901234567890123}
!38 = !{i64 3186456655321080972, i64 6789012345678901234}
!39 = !{i64 3186456655321080972, i64 7890123456789012345}
!40 = !{i64 3186456655321080972, i64 -7890123456789012345}
!41 = !{i64 3186456655321080972, i64 -6789012345678901234}
!42 = !{i64 3186456655321080972, i64 -5678901234567890123}
!101 = !{i64 8632435727821051414}
!102 = !{i64 -3421689549917153178}
!103 = !{i64 1234567890123456789}
!104 = !{i64 2345678901234567890}
!105 = !{i64 3456789012345678901}
!106 = !{i64 4567890123456789012}
!107 = !{i64 5678901234567890123}
!108 = !{i64 6789012345678901234}
!109 = !{i64 7890123456789012345}
!110 = !{i64 -7890123456789012345}
!111 = !{i64 -6789012345678901234}
!112 = !{i64 -5678901234567890123}

;--- baz.ll
source_filename = "baz.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define ptr @baz_imp(ptr %fp) #0 {
entry:
  %call = tail call ptr %fp(), !prof !0
  ret ptr %call
}

attributes #0 = { noinline }

;; VP metadata with unprefixed original name GUID MD5("local_alias") =
;; 13999805283412485828 (-4446938790297065788).
!0 = !{!"VP", i32 0, i64 4, i64 -4446938790297065788, i64 4}

;--- b.ll
source_filename = "b.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@local_alias = internal alias ptr (), ptr @local_aliasee

define internal ptr @local_aliasee() {
entry:
  %call = tail call ptr @leaf()
  ret ptr %call
}

define ptr @baz_same(ptr %fp) #0 {
entry:
  %call = tail call ptr %fp(), !prof !0
  ret ptr %call
}

declare ptr @leaf()

attributes #0 = { noinline }

;; VP metadata with unprefixed original name GUID MD5("local_alias") =
;; 13999805283412485828 (-4446938790297065788).
!0 = !{!"VP", i32 0, i64 4, i64 -4446938790297065788, i64 4}

;--- both.ll
source_filename = "both.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define ptr @both_via_alias(ptr %fp) #0 {
entry:
  %call = tail call ptr %fp(), !prof !0
  ret ptr %call
}

define ptr @both_via_aliasee(ptr %fp) #0 {
entry:
  %call = tail call ptr %fp(), !prof !1
  ret ptr %call
}

attributes #0 = { noinline }

;; MD5("local_alias") = -4446938790297065788.
!0 = !{!"VP", i32 0, i64 4, i64 -4446938790297065788, i64 4}
;; MD5("local_aliasee") = 5179408749141818126.
!1 = !{!"VP", i32 0, i64 4, i64 5179408749141818126, i64 4}

;--- decl.ll
source_filename = "decl.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define ptr @decl_via_alias(ptr %fp) #0 !prof !29 {
entry:
  %call = tail call ptr %fp(), !prof !30
  ret ptr %call
}

define ptr @decl_via_aliasee(ptr %fp) #0 !prof !29 {
entry:
  %call = tail call ptr %fp(), !prof !31
  ret ptr %call
}

attributes #0 = { noinline }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"ProfileSummary", !1}
!1 = !{!2, !3, !4, !5, !6, !7, !8, !9, !10, !11}
!2 = !{!"ProfileFormat", !"InstrProf"}
!3 = !{!"TotalCount", i64 100}
!4 = !{!"MaxCount", i64 70}
!5 = !{!"MaxInternalCount", i64 0}
!6 = !{!"MaxFunctionCount", i64 70}
!7 = !{!"NumCounts", i64 2}
!8 = !{!"NumFunctions", i64 2}
!9 = !{!"IsPartialProfile", i64 0}
!10 = !{!"PartialProfileRatio", double 0.000000e+00}
!11 = !{!"DetailedSummary", !12}
!12 = !{!13, !14, !15}
!13 = !{i32 10000, i64 70, i32 1}
!14 = !{i32 990000, i64 50, i32 1}
!15 = !{i32 999999, i64 40, i32 2}
!29 = !{!"function_entry_count", i64 1}
;; Hot count 70 (>= 50): imports @local_alias as definition.
!30 = !{!"VP", i32 0, i64 70, i64 -4446938790297065788, i64 70}
;; Cold count 30 (<= 40): imports @local_aliasee as declaration (after Mover.move).
!31 = !{!"VP", i32 0, i64 30, i64 5179408749141818126, i64 30}
