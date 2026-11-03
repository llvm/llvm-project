;; Test that distributed ThinLTO backends locate synthesized CallsiteInfo
;; records for missing indirect tail call frames when the profiled target in VP
;; metadata is a local (internal) alias whose VP candidate GUID is the
;; unprefixed original name GUID MD5("local_alias").
;;
;; Covers both distributed ThinLTO link-failure subcases where main.o calls
;; noinline callers on notcold+cold contexts (emitting U <caller>.memprof.1):
;; 1. baz.ll (baz.o): @baz_imp has an indirect tail call to @local_alias, which
;;    is promoted and imported into baz.o as an available_externally definition
;;    via replaceAliasWithAliasee (retaining @aliasee's !guid metadata).
;; 2. b.ll (b.o): @baz_same has an indirect tail call to @local_alias in the
;;    same module where @local_alias is a GlobalAlias.

; REQUIRES: asserts

; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: opt -thinlto-bc main.ll -o main.o
; RUN: opt -thinlto-bc baz.ll -o baz.o
; RUN: opt -thinlto-bc b.ll -o b.o

;; Distributed ThinLTO thin link.
; RUN: llvm-lto2 run main.o baz.o b.o \
; RUN:  -enable-memprof-context-disambiguation \
; RUN:  -supports-hot-cold-new \
; RUN:  -thinlto-distributed-indexes \
; RUN:  -r=main.o,main,plx \
; RUN:  -r=main.o,baz_imp, \
; RUN:  -r=main.o,baz_same, \
; RUN:  -r=main.o,leaf,plx \
; RUN:  -r=main.o,_Znam, \
; RUN:  -r=baz.o,baz_imp,plx \
; RUN:  -r=b.o,aliasee,plx \
; RUN:  -r=b.o,baz_same,plx \
; RUN:  -r=b.o,leaf, \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes \
; RUN:  -stats -o distrib.out 2>&1 | FileCheck %s --check-prefix=STATS-LINK

; STATS-LINK: 4 memprof-context-disambiguation - Number of profiled callees found via tail calls
; STATS-LINK: 4 memprof-context-disambiguation - Number of tail calls in chains to profiled callees that are indirect (thin link)

;; Verify main.o backend references both .memprof.1 clones.
; RUN: opt -import-all-index -passes=function-import,memprof-context-disambiguation \
; RUN:  -summary-file=main.o.thinlto.bc -memprof-import-summary=main.o.thinlto.bc \
; RUN:  -enable-import-metadata \
; RUN:  main.o -S -o distrib.main.ll
; RUN: FileCheck %s --check-prefix=IR-MAIN < distrib.main.ll

; IR-MAIN-DAG: call ptr @baz_imp.memprof.1(ptr null)
; IR-MAIN-DAG: call ptr @baz_same.memprof.1(ptr null)

;; Subcase 5a: Distributed ThinLTO backend for baz.o (@local_alias promoted and
;; imported as an available_externally definition via replaceAliasWithAliasee).
;; Verify @local_alias.llvm.* is imported as a definition retaining @aliasee's
;; !guid (-1039159065113703048) and recording MD5("b.cc;local_alias")
;; (-592420616430196302) in !thinlto_alias_guid, is NOT cloned in baz.o, and
;; @baz_imp.memprof.1 IS created so main.o's reference is satisfied.
; RUN: opt -import-all-index -passes=function-import,memprof-context-disambiguation \
; RUN:  -summary-file=baz.o.thinlto.bc -memprof-import-summary=baz.o.thinlto.bc \
; RUN:  -enable-import-metadata \
; RUN:  -pass-remarks=memprof-context-disambiguation \
; RUN:  -stats baz.o -S -o distrib.baz.ll 2>&1 | FileCheck %s --check-prefix=REMARKS-BAZ \
; RUN:  --implicit-check-not="created clone local_alias"
; RUN: FileCheck %s --check-prefix=IR-BAZ < distrib.baz.ll

; REMARKS-BAZ: created clone baz_imp.memprof.1
; REMARKS-BAZ: 1 memprof-context-disambiguation - Number of indirect tail calls recorded for ICP for cloning during ThinLTO backend

; IR-BAZ-NOT: define {{.*}}@local_alias.llvm.{{[0-9]+}}.memprof.1
; IR-BAZ: define dso_local ptr @baz_imp(ptr %fp)
; IR-BAZ: define available_externally hidden ptr @local_alias.llvm.{{[0-9]+}}() !guid ![[BAZ_GUID:[0-9]+]] !thinlto_src_module !{{[0-9]+}} !thinlto_src_file !{{[0-9]+}} !thinlto_alias_guid ![[BAZ_ALIAS_GUID:[0-9]+]] {
; IR-BAZ: define dso_local ptr @baz_imp.memprof.1(ptr %fp)
; IR-BAZ-NOT: define {{.*}}@local_alias.llvm.{{[0-9]+}}.memprof.1
; IR-BAZ-DAG: ![[BAZ_GUID]] = !{i64 -1039159065113703048}
; IR-BAZ-DAG: ![[BAZ_ALIAS_GUID]] = !{i64 -592420616430196302}

;; Subcase 5b: Distributed ThinLTO backend for b.o (@local_alias is a
;; GlobalAlias in the same module as @baz_same). Verify @baz_same.memprof.1 IS
;; created so main.o's reference is satisfied.
; RUN: opt -import-all-index -passes=function-import,memprof-context-disambiguation \
; RUN:  -summary-file=b.o.thinlto.bc -memprof-import-summary=b.o.thinlto.bc \
; RUN:  -enable-import-metadata \
; RUN:  -pass-remarks=memprof-context-disambiguation \
; RUN:  -stats b.o -S -o distrib.b.ll 2>&1 | FileCheck %s --check-prefix=REMARKS-B
; RUN: FileCheck %s --check-prefix=IR-B < distrib.b.ll

; REMARKS-B-DAG: created clone aliasee.memprof.1
; REMARKS-B-DAG: created clone baz_same.memprof.1
; REMARKS-B: 1 memprof-context-disambiguation - Number of indirect tail calls recorded for ICP for cloning during ThinLTO backend

; IR-B-DAG: @local_alias.llvm.{{[0-9]+}}.memprof.1 = hidden alias ptr (), ptr @aliasee.memprof.1
; IR-B-DAG: define dso_local ptr @baz_same.memprof.1(ptr %fp)

;--- main.ll
source_filename = "main.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main() {
entry:
  %c1 = call ptr @baz_imp(ptr null), !callsite !6
  %c2 = call ptr @baz_imp(ptr null), !callsite !7
  %c3 = call ptr @baz_same(ptr null), !callsite !8
  %c4 = call ptr @baz_same(ptr null), !callsite !9
  ret i32 0
}

declare ptr @baz_imp(ptr)
declare ptr @baz_same(ptr)

define ptr @leaf() #0 {
entry:
  %call = tail call ptr @_Znam(i64 10) #1, !memprof !1, !callsite !5
  ret ptr %call
}

declare ptr @_Znam(i64)

attributes #0 = { noinline }
attributes #1 = { builtin allocsize(0) }

!1 = !{!2, !4, !10, !12}
!2 = !{!3, !"notcold"}
!3 = !{i64 3186456655321080972, i64 8632435727821051414}
!4 = !{!14, !"cold"}
!14 = !{i64 3186456655321080972, i64 -3421689549917153178}
!10 = !{!11, !"notcold"}
!11 = !{i64 3186456655321080972, i64 1234567890123456789}
!12 = !{!13, !"cold"}
!13 = !{i64 3186456655321080972, i64 2345678901234567890}
!5 = !{i64 3186456655321080972}
!6 = !{i64 8632435727821051414}
!7 = !{i64 -3421689549917153178}
!8 = !{i64 1234567890123456789}
!9 = !{i64 2345678901234567890}

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

@local_alias = internal alias ptr (), ptr @aliasee

define ptr @aliasee() {
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
