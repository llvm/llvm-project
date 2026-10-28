;; Test to make sure that missing frames in memprof profiles due to indirect tail
;; calls are identified via the profiled targets of the indirect call, and that
;; the ThinLTO backend promotes the indirect tail call so that it can call the
;; cloned target. A second profiled target of the indirect tail call (qux) is
;; not on any profiled context, so it has no synthesized callsite record, and
;; should be promoted to call the original target in every clone.

;; -stats requires asserts
; REQUIRES: asserts

; RUN: opt -thinlto-bc %s >%t.o

;; The summary edges for the profiled targets of the indirect tail call should be
;; marked as tail calls (as well as indirect).
; RUN: llvm-dis %t.o -o - | FileCheck %s --check-prefix=SUMMARY
; SUMMARY: gv: (name: "_Z3bazPFPvvE", {{.*}} calls: ((callee: ^{{[0-9]+}}, hotness: none, tail: 1, indirect: 1), (callee: ^{{[0-9]+}}, hotness: none, tail: 1, indirect: 1))

; RUN: llvm-lto2 run %t.o -enable-memprof-context-disambiguation \
; RUN:  -supports-hot-cold-new \
; RUN:  -r=%t.o,_Z3barv,plx \
; RUN:  -r=%t.o,_Z3quxv,plx \
; RUN:  -r=%t.o,_Z3bazPFPvvE,plx \
; RUN:  -r=%t.o,main,plx \
; RUN:  -r=%t.o,_Znam, \
; RUN:  -pass-remarks=memprof-context-disambiguation \
; RUN:  -stats -save-temps \
; RUN:  -o %t.out 2>&1 | FileCheck %s --check-prefixes=STATS,STATS-BE,REMARKS

; RUN: llvm-dis %t.out.1.4.opt.bc -o - | FileCheck %s --check-prefix=IR

;; Try again but with distributed ThinLTO
; RUN: llvm-lto2 run %t.o -enable-memprof-context-disambiguation \
; RUN:  -supports-hot-cold-new \
; RUN:  -thinlto-distributed-indexes \
; RUN:  -r=%t.o,_Z3barv,plx \
; RUN:  -r=%t.o,_Z3quxv,plx \
; RUN:  -r=%t.o,_Z3bazPFPvvE,plx \
; RUN:  -r=%t.o,main,plx \
; RUN:  -r=%t.o,_Znam, \
; RUN:  -stats \
; RUN:  -o %t2.out 2>&1 | FileCheck %s --check-prefix=STATS

;; Run ThinLTO backend
; RUN: opt -passes=memprof-context-disambiguation \
; RUN:  -memprof-import-summary=%t.o.thinlto.bc \
; RUN:  -pass-remarks=memprof-context-disambiguation \
; RUN:  -stats %t.o -S -o %t.be.ll 2>&1 | FileCheck %s --check-prefixes=STATS-BE,REMARKS
; RUN: FileCheck %s --check-prefix=IR < %t.be.ll

;; With the indirect tail call search disabled, the missing frame is not found,
;; and nothing is cloned.
; RUN: llvm-lto2 run %t.o -enable-memprof-context-disambiguation \
; RUN:  -supports-hot-cold-new \
; RUN:  -memprof-indirect-tail-call-search=false \
; RUN:  -r=%t.o,_Z3barv,plx \
; RUN:  -r=%t.o,_Z3quxv,plx \
; RUN:  -r=%t.o,_Z3bazPFPvvE,plx \
; RUN:  -r=%t.o,main,plx \
; RUN:  -r=%t.o,_Znam, \
; RUN:  -stats -save-temps \
; RUN:  -o %t3.out 2>&1 | FileCheck %s --check-prefix=STATS-OFF
; RUN: llvm-dis %t3.out.1.4.opt.bc -o - | FileCheck %s --check-prefix=IR-OFF

; REMARKS-DAG: call in clone _Z3bazPFPvvE promoted and assigned to call function clone _Z3barv
; REMARKS-DAG: call in clone _Z3bazPFPvvE.memprof.1 promoted and assigned to call function clone _Z3barv.memprof.1
; REMARKS-DAG: call in clone _Z3bazPFPvvE promoted and assigned to call function clone _Z3quxv
; REMARKS-DAG: call in clone _Z3bazPFPvvE.memprof.1 promoted and assigned to call function clone _Z3quxv

;; The missing frame is found once for each of the 2 profiled callers (main's
;; calls to baz).
; STATS: 2 memprof-context-disambiguation - Number of profiled callees found via tail calls
; STATS: 2 memprof-context-disambiguation - Number of tail calls in chains to profiled callees that are indirect (thin link)
; STATS-BE: 1 memprof-context-disambiguation - Number of indirect tail calls recorded for ICP for cloning during ThinLTO backend

; STATS-OFF-NOT: Number of profiled callees found via tail calls
; STATS-OFF-NOT: Number of indirect tail calls recorded for ICP
; IR-OFF-NOT: memprof.1

source_filename = "memprof-tailcall-indirect.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline
; IR-LABEL: @_Z3barv()
define ptr @_Z3barv() local_unnamed_addr #0 {
entry:
  ; IR: call {{.*}} @_Znam(i64 10) #[[NOTCOLD:[0-9]+]]
  %call = tail call ptr @_Znam(i64 10) #2, !memprof !0, !callsite !5
  ret ptr %call
}

; Function Attrs: nobuiltin allocsize(0)
declare ptr @_Znam(i64) #1

;; Another target of the indirect tail call, not on any profiled context.
; Function Attrs: noinline
define ptr @_Z3quxv() #0 {
entry:
  ;; Side effect to prevent the promoted calls to qux from being removed.
  call void asm sideeffect "", ""()
  ret ptr null
}

;; The indirect tail call to bar (via the function pointer) is missing from the
;; profiled contexts, which have main calling bar directly.
; Function Attrs: noinline
; IR-LABEL: @_Z3bazPFPvvE(
define ptr @_Z3bazPFPvvE(ptr %fp) #0 {
entry:
  ;; The original version is promoted to call the original bar and qux.
  ; IR: icmp eq ptr %fp, @_Z3barv
  ; IR: call {{.*}} @_Z3barv()
  ; IR: icmp eq ptr %fp, @_Z3quxv
  ; IR: call {{.*}} @_Z3quxv()
  %call = tail call ptr %fp(), !prof !8
  ret ptr %call
}

; Function Attrs: noinline
; IR-LABEL: @main()
define i32 @main() #0 {
  ;; The first call to baz is part of a not cold context, and should use the
  ;; original functions.
  ; IR: call {{.*}} @_Z3bazPFPvvE(
  %call = call ptr @_Z3bazPFPvvE(ptr @_Z3barv), !callsite !6
  ;; The second call to baz is part of a cold context, and should call the
  ;; cloned functions.
  ; IR: call {{.*}} @_Z3bazPFPvvE.memprof.1(
  %call1 = call ptr @_Z3bazPFPvvE(ptr @_Z3barv), !callsite !7
  ret i32 0
}

; IR-LABEL: @_Z3barv.memprof.1()
; IR: call {{.*}} @_Znam(i64 10) #[[COLD:[0-9]+]]
;; The clone is promoted to call the cloned bar, and the original qux.
; IR-LABEL: @_Z3bazPFPvvE.memprof.1(
; IR: icmp eq ptr %fp, @_Z3barv
; IR: call {{.*}} @_Z3barv.memprof.1()
; IR: icmp eq ptr %fp, @_Z3quxv
; IR: call {{.*}} @_Z3quxv()
; IR-NOT: @_Z3quxv.memprof

; IR: attributes #[[NOTCOLD]] = { builtin allocsize(0) "memprof"="notcold" }
; IR: attributes #[[COLD]] = { builtin allocsize(0) "memprof"="cold" }

attributes #0 = { noinline }
attributes #1 = { nobuiltin allocsize(0) }
attributes #2 = { builtin allocsize(0) }

!0 = !{!1, !3}
!1 = !{!2, !"notcold"}
!2 = !{i64 3186456655321080972, i64 8632435727821051414}
!3 = !{!4, !"cold"}
!4 = !{i64 3186456655321080972, i64 -3421689549917153178}
!5 = !{i64 3186456655321080972}
!6 = !{i64 8632435727821051414}
!7 = !{i64 -3421689549917153178}
;; VP metadata with the GUIDs of _Z3barv and _Z3quxv as the profiled targets.
!8 = !{!"VP", i32 0, i64 8, i64 -1069303473483922844, i64 4, i64 -8259963147212483945, i64 4}
