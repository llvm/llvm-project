;; Test handling of indirect call promotion in ThinLTO backends with MemProf
;; context disambiguation enabled, when the profiled target is the aliasee of
;; an alias that was imported as a copy of the aliasee (see FunctionImport's
;; replaceAliasWithAliasee), and the aliasee itself was not imported. IRPGO
;; value profiles record the aliasee for calls through an alias, as the alias
;; has no profile data of its own, and the imported alias copy retains the
;; aliasee's !guid metadata.
;;
;; 1. @icp_regular: The indirect call is not on a MemProf context, so it is
;;    promoted by the regular PGOIndirectCallPromotion pass after
;;    MemProfContextDisambiguation. MemProfContextDisambiguation's own symtab
;;    keys imported alias copies by the alias GUID instead of the aliasee GUID,
;;    which must not affect the symtab used by the regular ICP pass, which
;;    should still find the alias copy via the aliasee GUID.
;; 2. @icp_caller: The indirect call is on MemProf contexts for which the
;;    aliasee is cloned, so MemProfContextDisambiguation must promote it to the
;;    alias copy to call the assigned clone (c_alias.memprof.1, created as an
;;    alias to c_aliasee.memprof.1 in c.o). The importing of a declaration of
;;    the cloned aliasee is disabled so that it is not in the module.

; REQUIRES: asserts

; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: opt -thinlto-bc main.ll -o main.o
; RUN: opt -thinlto-bc icp.ll -o icp.o
; RUN: opt -thinlto-bc c.ll -o c.o

;; In-process ThinLTO.
; RUN: llvm-lto2 run main.o icp.o c.o \
; RUN:  -enable-memprof-context-disambiguation \
; RUN:  -enable-memprof-indirect-call-support=true \
; RUN:  -memprof-import-cloned-indirect-targets=false \
; RUN:  -supports-hot-cold-new \
; RUN:  -r=main.o,main,plx \
; RUN:  -r=main.o,icp_caller, \
; RUN:  -r=icp.o,icp_caller,plx \
; RUN:  -r=icp.o,icp_regular,plx \
; RUN:  -r=icp.o,other,plx \
; RUN:  -r=icp.o,c_alias, \
; RUN:  -r=c.o,c_alias,plx \
; RUN:  -r=c.o,_Znam, \
; RUN:  -thinlto-threads=1 \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes \
; RUN:  -pass-remarks=memprof-context-disambiguation \
; RUN:  -o inproc.out 2>&1 | FileCheck %s --check-prefix=REMARKS-INPROC
; RUN: llvm-nm inproc.out.1 | FileCheck %s --check-prefix=NM-MAIN
; RUN: llvm-nm inproc.out.2 | FileCheck %s --check-prefix=NM-ICP
; RUN: llvm-nm inproc.out.3 | FileCheck %s --check-prefix=NM-C

; REMARKS-INPROC-DAG: created clone icp_caller.memprof.1
; REMARKS-INPROC-DAG: call in clone icp_caller.memprof.1 promoted and assigned to call function clone c_alias.memprof.1
; REMARKS-INPROC-DAG: created clone c_aliasee.memprof.1

; NM-MAIN-DAG: U icp_caller.memprof.1

; NM-ICP-DAG: T icp_caller{{$}}
; NM-ICP-DAG: T icp_caller.memprof.1{{$}}
; NM-ICP-DAG: U c_alias{{$}}
; NM-ICP-DAG: U c_alias.memprof.1{{$}}

; NM-C-DAG: T c_alias{{$}}
; NM-C-DAG: T c_alias.memprof.1{{$}}

;; Distributed ThinLTO.
; RUN: llvm-lto2 run main.o icp.o c.o \
; RUN:  -enable-memprof-context-disambiguation \
; RUN:  -enable-memprof-indirect-call-support=true \
; RUN:  -memprof-import-cloned-indirect-targets=false \
; RUN:  -supports-hot-cold-new \
; RUN:  -thinlto-distributed-indexes \
; RUN:  -r=main.o,main,plx \
; RUN:  -r=main.o,icp_caller, \
; RUN:  -r=icp.o,icp_caller,plx \
; RUN:  -r=icp.o,icp_regular,plx \
; RUN:  -r=icp.o,other,plx \
; RUN:  -r=icp.o,c_alias, \
; RUN:  -r=c.o,c_alias,plx \
; RUN:  -r=c.o,_Znam, \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes \
; RUN:  -o distrib.out

;; @c_alias is imported as a definition (hot direct calls), while @c_aliasee is
;; not imported (cold indirect call target, and declaration import disabled).
; RUN: llvm-dis icp.o.thinlto.bc -o - | FileCheck %s --check-prefix=INDEX
; INDEX: gv: (guid: 789939177336168348, summaries: (alias: ({{.*}}importType: definition
; INDEX-NOT: gv: (guid: 11222198776233219311, summaries: (function: ({{.*}}importType

; RUN: opt -import-all-index \
; RUN:  -passes=function-import,memprof-context-disambiguation,pgo-icall-prom \
; RUN:  -enable-memprof-indirect-call-support=true \
; RUN:  -summary-file=icp.o.thinlto.bc -memprof-import-summary=icp.o.thinlto.bc \
; RUN:  -enable-import-metadata \
; RUN:  -pass-remarks='memprof-context-disambiguation|pgo-icall-prom' \
; RUN:  icp.o -S -o distrib.icp.ll 2>&1 | FileCheck %s --check-prefix=REMARKS
; RUN: FileCheck %s --check-prefix=IR < distrib.icp.ll

; REMARKS-DAG: created clone icp_caller.memprof.1
; REMARKS-DAG: call in clone icp_caller.memprof.1 promoted and assigned to call function clone c_alias.memprof.1
; REMARKS-DAG: Promote indirect call to other with count 60 out of 100
; REMARKS-DAG: Promote indirect call to c_alias with count 40 out of 40

; IR-NOT: @c_aliasee
; IR: define dso_local ptr @icp_caller(ptr %fp)
; IR: icmp eq ptr %fp, @c_alias
; IR: call ptr @c_alias()
; IR: define dso_local ptr @icp_regular(ptr %fp)
; IR: icmp eq ptr %fp, @other
; IR: icmp eq ptr %fp, @c_alias
; IR: define available_externally dso_local ptr @c_alias() !guid ![[GUID:[0-9]+]] !thinlto_src_module !{{[0-9]+}} !thinlto_src_file !{{[0-9]+}} !thinlto_alias_guid ![[ALIAS_GUID:[0-9]+]] {
; IR: define dso_local ptr @icp_caller.memprof.1(ptr %fp)
; IR: icmp eq ptr %fp, @c_alias
; IR: call ptr @c_alias.memprof.1()
; IR: declare ptr @c_alias.memprof.1()
; IR-NOT: @c_aliasee
; IR-DAG: ![[GUID]] = !{i64 -7224545297476332305}
; IR-DAG: ![[ALIAS_GUID]] = !{i64 789939177336168348}

;--- main.ll
source_filename = "main.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main() {
entry:
  %c1 = call ptr @icp_caller(ptr null), !callsite !1
  %c2 = call ptr @icp_caller(ptr null), !callsite !2
  ret i32 0
}

declare ptr @icp_caller(ptr)

!1 = !{i64 8632435727821051414}
!2 = !{i64 -3421689549917153178}

;--- icp.ll
source_filename = "icp.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define ptr @icp_caller(ptr %fp) #0 !prof !13 {
entry:
  %d = call ptr @c_alias()
  %r = call ptr %fp(), !prof !18, !callsite !19
  ret ptr %r
}

define ptr @icp_regular(ptr %fp) #0 !prof !13 {
entry:
  %d = call ptr @c_alias()
  %r = call ptr %fp(), !prof !14
  ret ptr %r
}

define ptr @other() {
entry:
  ret ptr null
}

declare ptr @c_alias()

attributes #0 = { noinline }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"ProfileSummary", !1}
!1 = !{!2, !3, !4, !5, !6, !7, !8, !9, !10, !11}
!2 = !{!"ProfileFormat", !"InstrProf"}
!3 = !{!"TotalCount", i64 100}
!4 = !{!"MaxCount", i64 60}
!5 = !{!"MaxInternalCount", i64 0}
!6 = !{!"MaxFunctionCount", i64 60}
!7 = !{!"NumCounts", i64 2}
!8 = !{!"NumFunctions", i64 2}
!9 = !{!"IsPartialProfile", i64 0}
!10 = !{!"PartialProfileRatio", double 0.000000e+00}
!11 = !{!"DetailedSummary", !12}
!12 = !{!15, !16, !17}
;; Hot: count >= 60. Cold: count <= 40.
!15 = !{i32 10000, i64 60, i32 1}
!16 = !{i32 990000, i64 60, i32 1}
!17 = !{i32 999999, i64 40, i32 2}
!13 = !{!"function_entry_count", i64 60}
;; IRPGO VP metadata: @other (hot, count 60) and @c_aliasee's PGO name GUID
;; MD5("c.cc;c_aliasee") = 11222198776233219311 (cold, count 40), the latter
;; profiled via a call through @c_alias.
!14 = !{!"VP", i32 0, i64 100, i64 -4848550849129848967, i64 60, i64 -7224545297476332305, i64 40}
!18 = !{!"VP", i32 0, i64 40, i64 -7224545297476332305, i64 40}
!19 = !{i64 1111111111111111111}

;--- c.ll
source_filename = "c.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@c_alias = alias ptr (), ptr @c_aliasee

define internal ptr @c_aliasee() {
entry:
  %call = tail call ptr @_Znam(i64 10) #1, !memprof !1, !callsite !5
  ret ptr %call
}

declare ptr @_Znam(i64)

attributes #1 = { builtin allocsize(0) }

!1 = !{!2, !4}
!2 = !{!3, !"notcold"}
!3 = !{i64 3186456655321080972, i64 1111111111111111111, i64 8632435727821051414}
!4 = !{!6, !"cold"}
!6 = !{i64 3186456655321080972, i64 1111111111111111111, i64 -3421689549917153178}
!5 = !{i64 3186456655321080972}
