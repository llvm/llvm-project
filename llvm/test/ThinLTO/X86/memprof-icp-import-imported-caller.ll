;; Test that the thin link imports a declaration of a cross-module indirect call
;; target that MemProf cloning requires to be promoted, into a module that the
;; caller containing the indirect call is imported into (and that doesn't
;; define the target). The ThinLTO backend clones and performs ICP on imported
;; functions as well, which requires the target to be available in the
;; importing module.
;;
;; The contexts are main -> baz -> foo -(indirect)-> _ZN1A8PushSlowEv, where each
;; function is defined in a different module, except for _ZN1A8PushSlowEv which
;; is defined in main's module. foo is imported into baz's module. The target
;; is marked noinline so that its definition is never imported.

; REQUIRES: asserts

; RUN: split-file %s %t
; RUN: opt -thinlto-bc %t/main.ll -o %t/main.o
; RUN: opt -thinlto-bc %t/foo.ll -o %t/foo.o
; RUN: opt -thinlto-bc %t/baz.ll -o %t/baz.o

; RUN: llvm-lto2 run %t/main.o %t/foo.o %t/baz.o \
; RUN:  -enable-memprof-context-disambiguation \
; RUN:	-enable-memprof-indirect-call-support=true \
; RUN:  -supports-hot-cold-new \
; RUN:  -r=%t/foo.o,_Z3fooPv,plx \
; RUN:  -r=%t/baz.o,_Z3bazv,plx \
; RUN:  -r=%t/baz.o,_Z3fooPv, \
; RUN:  -r=%t/main.o,main,plx \
; RUN:  -r=%t/main.o,_Z3bazv, \
; RUN:  -r=%t/main.o,_ZN1A8PushSlowEv,pl \
; RUN:  -r=%t/main.o,_Znwm, \
; RUN:	-thinlto-threads=1 \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes -stats \
; RUN:  -pass-remarks=memprof-context-disambiguation \
; RUN:  -pass-remarks-missed=memprof-context-disambiguation \
; RUN:  -save-temps \
; RUN:  -o %t.out 2>&1 | FileCheck %s --check-prefixes=REMARKS,STATS \
; RUN:	--implicit-check-not="cannot promote"
;; Check the IR after importing for baz's module, as the call to foo in baz
;; (and the inlined indirect call) is subsequently optimized away.
; RUN: llvm-dis %t.out.3.3.import.bc -o - | FileCheck %s --check-prefix=IR-BAZ
; RUN: llvm-nm %t.out.1 | FileCheck %s --check-prefix=EXPORTED

;; Without importing the target's declaration, the indirect call in the copies
;; of foo in foo's and baz's modules can't be promoted.
; RUN: llvm-lto2 run %t/main.o %t/foo.o %t/baz.o \
; RUN:  -enable-memprof-context-disambiguation \
; RUN:	-enable-memprof-indirect-call-support=true \
; RUN:  -supports-hot-cold-new \
; RUN:  -memprof-import-cloned-indirect-targets=false \
; RUN:  -r=%t/foo.o,_Z3fooPv,plx \
; RUN:  -r=%t/baz.o,_Z3bazv,plx \
; RUN:  -r=%t/baz.o,_Z3fooPv, \
; RUN:  -r=%t/main.o,main,plx \
; RUN:  -r=%t/main.o,_Z3bazv, \
; RUN:  -r=%t/main.o,_ZN1A8PushSlowEv,pl \
; RUN:  -r=%t/main.o,_Znwm, \
; RUN:	-thinlto-threads=1 \
; RUN:  -pass-remarks-missed=memprof-context-disambiguation \
; RUN:  -o %t.none.out 2>&1 | FileCheck %s --check-prefix=REMARKS-NONE
; REMARKS-NONE-COUNT-2: Memprof cannot promote indirect call: target with md5sum 17285164559125343003 not found

;; Distributed ThinLTO: the declaration import is recorded in baz's index.
; RUN: llvm-lto2 run %t/main.o %t/foo.o %t/baz.o \
; RUN:  -enable-memprof-context-disambiguation \
; RUN:	-enable-memprof-indirect-call-support=true \
; RUN:  -supports-hot-cold-new \
; RUN:  -r=%t/foo.o,_Z3fooPv,plx \
; RUN:  -r=%t/baz.o,_Z3bazv,plx \
; RUN:  -r=%t/baz.o,_Z3fooPv, \
; RUN:  -r=%t/main.o,main,plx \
; RUN:  -r=%t/main.o,_Z3bazv, \
; RUN:  -r=%t/main.o,_ZN1A8PushSlowEv,pl \
; RUN:  -r=%t/main.o,_Znwm, \
; RUN:  -thinlto-distributed-indexes \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes \
; RUN:  -o %t.distrib.out
; RUN: llvm-dis %t/baz.o.thinlto.bc -o - | FileCheck %s --check-prefix=DISTRIB-INDEX
;; Run ThinLTO backend for baz.o.
; RUN: opt -import-all-index -passes=function-import,memprof-context-disambiguation \
; RUN:  -enable-memprof-indirect-call-support=true \
; RUN:  -summary-file=%t/baz.o.thinlto.bc -memprof-import-summary=%t/baz.o.thinlto.bc \
; RUN:  -enable-import-metadata \
; RUN:  -pass-remarks=memprof-context-disambiguation \
; RUN:  -pass-remarks-missed=memprof-context-disambiguation \
; RUN:  %t/baz.o -S -o %t.distrib.baz.ll 2>&1 | FileCheck %s --check-prefix=REMARKS-BAZ \
; RUN:	--implicit-check-not="cannot promote"
; RUN: FileCheck %s --check-prefix=IR-BAZ < %t.distrib.baz.ll

;; The indirect call in foo is promoted in each module containing a copy of foo:
;; main's module, which defines the target and imports foo (via baz), foo's
;; module, and baz's module, which imports foo.
; REMARKS-COUNT-3: call in clone _Z3fooPv.memprof.1 promoted and assigned to call function clone _ZN1A8PushSlowEv.memprof.1

;; The declaration is imported into foo's module, which contains the indirect
;; call, and into baz's module, which foo is imported into.
; STATS: 2 function-import - Number of function declarations thin link decided to import because the callee is required in the caller's module (e.g. for MemProf ICP)

;; The indirect call in the copy of foo imported into baz's module is promoted.
; REMARKS-BAZ: call in clone _Z3fooPv.memprof.1 promoted and assigned to call function clone _ZN1A8PushSlowEv.memprof.1

; IR-BAZ: declare !guid !{{[0-9]+}} {{.*}}i32 @_ZN1A8PushSlowEv(ptr, i32)

; EXPORTED: T _ZN1A8PushSlowEv{{$}}

; DISTRIB-INDEX: gv: (guid: 17285164559125343003, summaries: (function: (module: ^{{[0-9]+}}, flags: ({{.*}}importType: declaration

;--- foo.ll
source_filename = "foo.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @_Z3fooPv(ptr %b) {
entry:
  %0 = load ptr, ptr %b, align 8
  %call = tail call i32 %0(ptr null, i32 0), !prof !1, !callsite !2
  ret i32 %call
}

;; VP metadata with a single candidate _ZN1A8PushSlowEv with a count of 1 (as
;; synthesized from the MemProf profile).
!1 = !{!"VP", i32 0, i64 1, i64 -1161579514584208613, i64 1}
!2 = !{i64 -2101080423462424381}

;--- baz.ll
source_filename = "baz.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @_Z3bazv() {
entry:
  %call = call i32 @_Z3fooPv(ptr null), !callsite !1
  ret i32 %call
}

declare i32 @_Z3fooPv(ptr)

!1 = !{i64 4374214591823871002}

;--- main.ll
source_filename = "main.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main() {
entry:
  %call2 = call i32 @_Z3bazv(), !callsite !30
  %call4 = call i32 @_Z3bazv(), !callsite !31
  ret i32 0
}

declare i32 @_Z3bazv()

define i32 @_ZN1A8PushSlowEv(ptr %this, i32 %s) #1 {
entry:
  %call = tail call ptr @_Znwm(i64 noundef 4) #0, !memprof !33, !callsite !38
  ret i32 %s
}

declare ptr @_Znwm(i64)

attributes #0 = { builtin allocsize(0) }
attributes #1 = { noinline }

!30 = !{i64 -6490791336773930154}
!31 = !{i64 5188446645037944434}
!33 = !{!34, !36}
!34 = !{!35, !"notcold"}
!35 = !{i64 -852997907418798798, i64 -2101080423462424381, i64 4374214591823871002, i64 -6490791336773930154}
!36 = !{!37, !"cold"}
!37 = !{i64 -852997907418798798, i64 -2101080423462424381, i64 4374214591823871002, i64 5188446645037944434}
!38 = !{i64 -852997907418798798}
