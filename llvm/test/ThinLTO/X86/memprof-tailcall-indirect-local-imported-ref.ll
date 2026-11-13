;; Test that the ThinLTO backend can perform MemProf ICP and cloning for an
;; indirect tail call whose profiled target is a local function in another
;; module that is imported as a declaration, when the destination module also
;; imports the definition of a global whose initializer references that local
;; function. In that case the IRMover creates the declaration of the (promoted)
;; local function while linking the global's definition, and the declaration
;; import must still attach the original GUID of the local function (as !guid
;; metadata) so that the backend can match the promoted declaration with the
;; profiled target and with the callsite record synthesized by the thin link.
;;
;; This mirrors memprof-tailcall-indirect-local.ll, except that the indirect
;; tail call loads its target from a constant global defined in the module of
;; the local function, which is imported into the caller's module.

;; -stats requires asserts
; REQUIRES: asserts

; RUN: split-file %s %t
; RUN: opt -thinlto-bc %t/main.ll -o %t/main.o
; RUN: opt -thinlto-bc %t/baz.ll -o %t/baz.o

;; In-process ThinLTO.
; RUN: llvm-lto2 run %t/main.o %t/baz.o -enable-memprof-context-disambiguation \
; RUN:  -supports-hot-cold-new -import-instr-limit=1 \
; RUN:  -r=%t/baz.o,_Z3bazv,plx \
; RUN:  -r=%t/baz.o,fptab, \
; RUN:  -r=%t/main.o,main,plx \
; RUN:  -r=%t/main.o,fptab,plx \
; RUN:  -r=%t/main.o,_Z3bazv, \
; RUN:  -r=%t/main.o,_Znam, \
; RUN:  -thinlto-threads=1 \
; RUN:  -pass-remarks=memprof-context-disambiguation \
; RUN:  -pass-remarks-missed=memprof-context-disambiguation \
; RUN:  -stats -save-temps \
; RUN:  -o %t.out 2>&1 | FileCheck %s --check-prefixes=STATS,STATS-BE,REMARKS \
; RUN:  --implicit-check-not="cannot promote"

; RUN: llvm-dis %t.out.2.4.opt.bc -o - | FileCheck %s --check-prefix=IR-INPROC

;; Distributed ThinLTO.
; RUN: llvm-lto2 run %t/main.o %t/baz.o -enable-memprof-context-disambiguation \
; RUN:  -supports-hot-cold-new -import-instr-limit=1 \
; RUN:  -thinlto-distributed-indexes \
; RUN:  -r=%t/baz.o,_Z3bazv,plx \
; RUN:  -r=%t/baz.o,fptab, \
; RUN:  -r=%t/main.o,main,plx \
; RUN:  -r=%t/main.o,fptab,plx \
; RUN:  -r=%t/main.o,_Z3bazv, \
; RUN:  -r=%t/main.o,_Znam, \
; RUN:  -stats \
; RUN:  -o %t2.out 2>&1 | FileCheck %s --check-prefix=STATS

;; Run ThinLTO backend for baz.o. The definition of fptab is imported, which
;; creates the declaration of the promoted bar via the IRMover before the
;; declaration import runs.
; RUN: opt -import-all-index -passes=function-import,memprof-context-disambiguation \
; RUN:  -summary-file=%t/baz.o.thinlto.bc -memprof-import-summary=%t/baz.o.thinlto.bc \
; RUN:  -enable-import-metadata \
; RUN:  -pass-remarks=memprof-context-disambiguation \
; RUN:  -pass-remarks-missed=memprof-context-disambiguation \
; RUN:  -stats %t/baz.o -S -o %t.be.ll 2>&1 | FileCheck %s --check-prefixes=STATS-BE,REMARKS \
; RUN:  --implicit-check-not="cannot promote"
; RUN: FileCheck %s --check-prefix=IR-BE < %t.be.ll

; REMARKS-DAG: call in clone _Z3bazv promoted and assigned to call function clone _ZL3barv.llvm.{{[0-9]+}}{{$}}
; REMARKS-DAG: call in clone _Z3bazv.memprof.1 promoted and assigned to call function clone _ZL3barv.llvm.{{[0-9]+}}.memprof.1

;; The missing frame is found once for each of the 2 profiled callers (main's
;; calls to baz).
; STATS: 2 memprof-context-disambiguation - Number of profiled callees found via tail calls
; STATS: 2 memprof-context-disambiguation - Number of tail calls in chains to profiled callees that are indirect (thin link)
;; The (low) import instruction limit prevents importing the definition of bar
;; (and of baz into main's module), so only baz's indirect tail call is recorded.
; STATS-BE: 1 memprof-context-disambiguation - Number of indirect tail calls recorded for ICP for cloning during ThinLTO backend

;; In-process, the later optimization pipeline folds the load from the imported
;; constant fptab, leaving direct calls to the promoted bar and its clone.
; IR-INPROC: define {{.*}} @_Z3bazv()
; IR-INPROC:   call ptr @[[BAR:_ZL3barv.llvm.[0-9]+]]()
; IR-INPROC: define {{.*}} @_Z3bazv.memprof.1()
; IR-INPROC:   call ptr @[[BAR]].memprof.1()

;; The imported definition of fptab references the promoted bar.
; IR-BE: @fptab = available_externally {{.*}}constant ptr @[[BAR:_ZL3barv.llvm.[0-9]+]]
;; Both versions of baz compare against the original (promoted) bar.
; IR-BE: define {{.*}} @_Z3bazv(
; IR-BE:   icmp eq ptr %fp, @[[BAR]]{{$}}
;; The declaration of the promoted bar carries the original GUID
;; (MD5("main.cc;_ZL3barv")).
; IR-BE: declare !guid ![[GUID:[0-9]+]] {{.*}}@[[BAR]]()
; IR-BE: define {{.*}} @_Z3bazv.memprof.1(
; IR-BE:   icmp eq ptr %fp, @[[BAR]]{{$}}
; IR-BE: ![[GUID]] = !{i64 6179810419695740674}

;--- baz.ll
source_filename = "baz.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@fptab = external constant ptr

;; The indirect tail call to bar (via the function pointer loaded from fptab) is
;; missing from the profiled contexts, which have main calling bar directly.
define ptr @_Z3bazv() {
entry:
  %fp = load ptr, ptr @fptab
  %call = tail call ptr %fp(), !prof !0
  ret ptr %call
}

;; VP metadata with a single candidate, the original name GUID of the local
;; function _ZL3barv (i.e. MD5("_ZL3barv"), without the "main.cc;" prefix).
!0 = !{!"VP", i32 0, i64 4, i64 -4486107812812573614, i64 4}

;--- main.ll
source_filename = "main.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

;; Constant global referencing the local function, imported into baz's module.
@fptab = dso_local constant ptr @_ZL3barv

define i32 @main() {
entry:
  ;; The first call to baz is part of a not cold context, and the second is
  ;; part of a cold context.
  %call = call ptr @_Z3bazv(), !callsite !6
  %call1 = call ptr @_Z3bazv(), !callsite !7
  ret i32 0
}

declare ptr @_Z3bazv()

;; Local function only called indirectly.
define internal ptr @_ZL3barv() {
entry:
  %call = tail call ptr @_Znam(i64 10) #0, !memprof !1, !callsite !5
  ret ptr %call
}

declare ptr @_Znam(i64)

attributes #0 = { builtin allocsize(0) }

!1 = !{!2, !4}
!2 = !{!3, !"notcold"}
!3 = !{i64 3186456655321080972, i64 8632435727821051414}
!4 = !{!8, !"cold"}
!8 = !{i64 3186456655321080972, i64 -3421689549917153178}
!5 = !{i64 3186456655321080972}
!6 = !{i64 8632435727821051414}
!7 = !{i64 -3421689549917153178}
