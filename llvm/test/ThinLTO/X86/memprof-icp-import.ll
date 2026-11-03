;; Test that after the thin link MemProf cloning, the cross-module indirect call
;; targets of callsites assigned to call a cloned callee are made available in
;; the caller's module, so that the ThinLTO backend can perform MemProf ICP. If
;; the target's definition is not otherwise imported, a declaration is imported.
;; Here the target of the indirect call is not declared in the module containing
;; the indirect call, which is common for virtual calls through a base class. It
;; is also not visible outside the LTO unit, so it would be internalized if not
;; exported.

; REQUIRES: asserts

; RUN: split-file %s %t
; RUN: opt -thinlto-bc %t/main.ll -o %t/main.o
; RUN: opt -thinlto-bc %t/foo.ll -o %t/foo.o

;; The indirect call edge is not hot in the per-module summary, but is marked
;; as being from an indirect call.
; RUN: llvm-dis %t/foo.o -o - | FileCheck %s --check-prefix=SUMMARY
; SUMMARY: gv: (name: "_Z3fooPv", {{.*}} calls: ((callee: ^{{[0-9]+}}, hotness: none, tail: 1, indirect: 1))

;; The direct call edge from main to foo is not marked indirect.
; RUN: llvm-dis %t/main.o -o - | FileCheck %s --check-prefix=SUMMARY-MAIN
; SUMMARY-MAIN: gv: (name: "main", {{.*}} calls: ((callee: ^{{[0-9]+}})){{[,)]}}

;; Use a small import instruction limit such that the definition of
;; _ZN1A8PushSlowEv is not imported. By default a declaration of
;; _ZN1A8PushSlowEv is imported instead.
; RUN: llvm-lto2 run %t/main.o %t/foo.o -enable-memprof-context-disambiguation \
; RUN:	-enable-memprof-indirect-call-support=true \
; RUN:	-import-instr-limit=1 \
; RUN:  -supports-hot-cold-new \
; RUN:  -r=%t/foo.o,_Z3fooPv,plx \
; RUN:  -r=%t/main.o,main,plx \
; RUN:  -r=%t/main.o,_Z3fooPv, \
; RUN:  -r=%t/main.o,_ZN1A8PushSlowEv,pl \
; RUN:  -r=%t/main.o,_Znwm, \
; RUN:	-thinlto-threads=1 \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes -stats \
; RUN:  -pass-remarks=memprof-context-disambiguation \
; RUN:  -pass-remarks-missed=memprof-context-disambiguation \
; RUN:  -save-temps \
; RUN:  -o %t.decl.out 2>&1 | FileCheck %s --check-prefixes=REMARKS,STATS \
; RUN:	--implicit-check-not="cannot promote"
; RUN: llvm-dis %t.decl.out.2.4.opt.bc -o - | FileCheck %s --check-prefix=IR-DECL
;; The target was exported and therefore not internalized.
; RUN: llvm-nm %t.decl.out.1 | FileCheck %s --check-prefix=EXPORTED

; REMARKS: created clone _Z3fooPv.memprof.1
; REMARKS: call in clone _Z3fooPv promoted and assigned to call function clone _ZN1A8PushSlowEv{{$}}
; REMARKS: call in clone _Z3fooPv.memprof.1 promoted and assigned to call function clone _ZN1A8PushSlowEv.memprof.1

; STATS: 1 function-import - Number of function declarations imported in backend
; STATS: 1 function-import - Number of function declarations thin link decided to import because the callee is required in the caller's module (e.g. for MemProf ICP)
;; Both the indirect call in foo and the direct call from main to foo are
;; assigned to call a cloned callee, but only the indirect call
;; edge from foo to _ZN1A8PushSlowEv is updated, since it is the only one
;; marked as an indirect call edge in the summary.
; STATS: 2 memprof-context-disambiguation - Number of summary callsites assigned to call a cloned callee (thin link)
; STATS: 1 memprof-context-disambiguation - Number of summary indirect call edges to cloned callees marked as requiring a declaration import (thin link)
; STATS: 1 memprof-context-disambiguation - Number of summary callsites assigned to call a cloned callee with an indirect call edge (thin link)

; IR-DECL: define {{.*}} @_Z3fooPv(
; IR-DECL:   icmp eq ptr %0, @_ZN1A8PushSlowEv
; IR-DECL: declare !guid !{{[0-9]+}} {{.*}}i32 @_ZN1A8PushSlowEv(ptr, i32)
; IR-DECL: declare {{.*}} @_ZN1A8PushSlowEv.memprof.1(

; EXPORTED: T _ZN1A8PushSlowEv{{$}}

;; Distributed ThinLTO: the declaration import is recorded in foo's index.
; RUN: llvm-lto2 run %t/main.o %t/foo.o -enable-memprof-context-disambiguation \
; RUN:	-enable-memprof-indirect-call-support=true \
; RUN:	-import-instr-limit=1 \
; RUN:  -supports-hot-cold-new \
; RUN:  -r=%t/foo.o,_Z3fooPv,plx \
; RUN:  -r=%t/main.o,main,plx \
; RUN:  -r=%t/main.o,_Z3fooPv, \
; RUN:  -r=%t/main.o,_ZN1A8PushSlowEv,pl \
; RUN:  -r=%t/main.o,_Znwm, \
; RUN:  -thinlto-distributed-indexes \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes \
; RUN:  -o %t.distrib.out
; RUN: llvm-dis %t/foo.o.thinlto.bc -o - | FileCheck %s --check-prefix=DISTRIB-INDEX
; DISTRIB-INDEX: gv: (guid: 17285164559125343003, summaries: (function: (module: ^{{[0-9]+}}, flags: ({{.*}}importType: declaration
;; Run ThinLTO backend for foo.o.
; RUN: opt -import-all-index -passes=function-import,memprof-context-disambiguation \
; RUN:  -enable-memprof-indirect-call-support=true \
; RUN:  -summary-file=%t/foo.o.thinlto.bc -memprof-import-summary=%t/foo.o.thinlto.bc \
; RUN:  -pass-remarks=memprof-context-disambiguation \
; RUN:  -pass-remarks-missed=memprof-context-disambiguation \
; RUN:  %t/foo.o -S -o %t.distrib.foo.ll 2>&1 | FileCheck %s --check-prefix=REMARKS-DISTRIB \
; RUN:	--implicit-check-not="cannot promote"
; RUN: FileCheck %s --check-prefix=IR-DECL < %t.distrib.foo.ll
; REMARKS-DISTRIB: created clone _Z3fooPv.memprof.1
; REMARKS-DISTRIB: call in clone _Z3fooPv promoted and assigned to call function clone _ZN1A8PushSlowEv{{$}}
; REMARKS-DISTRIB: call in clone _Z3fooPv.memprof.1 promoted and assigned to call function clone _ZN1A8PushSlowEv.memprof.1

;; Without importing the target we can't promote, and the target is not
;; exported.
; RUN: llvm-lto2 run %t/main.o %t/foo.o -enable-memprof-context-disambiguation \
; RUN:	-enable-memprof-indirect-call-support=true \
; RUN:	-import-instr-limit=1 \
; RUN:  -supports-hot-cold-new \
; RUN:  -r=%t/foo.o,_Z3fooPv,plx \
; RUN:  -r=%t/main.o,main,plx \
; RUN:  -r=%t/main.o,_Z3fooPv, \
; RUN:  -r=%t/main.o,_ZN1A8PushSlowEv,pl \
; RUN:  -r=%t/main.o,_Znwm, \
; RUN:  -memprof-import-cloned-indirect-targets=false \
; RUN:	-thinlto-threads=1 \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes \
; RUN:  -pass-remarks=memprof-context-disambiguation \
; RUN:  -pass-remarks-missed=memprof-context-disambiguation \
; RUN:  -o %t.none.out 2>&1 | FileCheck %s --check-prefix=REMARKS-NONE
; RUN: llvm-nm %t.none.out.1 | FileCheck %s --check-prefix=NOT-EXPORTED

; REMARKS-NONE: Memprof cannot promote indirect call: target with md5sum 17285164559125343003 not found
;; It was internalized and then removed as dead.
; NOT-EXPORTED-NOT: _ZN1A8PushSlowEv
; NOT-EXPORTED: T main

;; With -memprof-require-definition-for-promotion, MemProf ICP can't promote to
;; a declaration, so the declaration is not imported, and the target is not
;; exported.
; RUN: llvm-lto2 run %t/main.o %t/foo.o -enable-memprof-context-disambiguation \
; RUN:	-enable-memprof-indirect-call-support=true \
; RUN:	-import-instr-limit=1 \
; RUN:  -supports-hot-cold-new \
; RUN:  -r=%t/foo.o,_Z3fooPv,plx \
; RUN:  -r=%t/main.o,main,plx \
; RUN:  -r=%t/main.o,_Z3fooPv, \
; RUN:  -r=%t/main.o,_ZN1A8PushSlowEv,pl \
; RUN:  -r=%t/main.o,_Znwm, \
; RUN:  -memprof-require-definition-for-promotion \
; RUN:	-thinlto-threads=1 \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes -stats \
; RUN:  -pass-remarks-missed=memprof-context-disambiguation \
; RUN:  -o %t.reqdef.out 2>&1 | FileCheck %s --check-prefix=REQDEF
; RUN: llvm-nm %t.reqdef.out.1 | FileCheck %s --check-prefix=NOT-EXPORTED

; REQDEF: Memprof cannot promote indirect call: target with md5sum 17285164559125343003 not found
;; The stats are printed after the remarks.
; REQDEF-NOT: Number of function declarations
; REQDEF-NOT: requiring a declaration import

;--- foo.ll
source_filename = "foo.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

;; Note there is no declaration of the indirect call target _ZN1A8PushSlowEv.
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

;--- main.ll
source_filename = "main.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main() {
entry:
  %call2 = call i32 @_Z3fooPv(ptr null), !callsite !30
  %call4 = call i32 @_Z3fooPv(ptr null), !callsite !31
  ret i32 0
}

declare i32 @_Z3fooPv(ptr)

define i32 @_ZN1A8PushSlowEv(ptr %this, i32 %s) {
entry:
  %call = tail call ptr @_Znwm(i64 noundef 4) #0, !memprof !33, !callsite !38
  ret i32 %s
}

declare ptr @_Znwm(i64)

attributes #0 = { builtin allocsize(0) }

!30 = !{i64 -6490791336773930154}
!31 = !{i64 5188446645037944434}
!33 = !{!34, !36}
!34 = !{!35, !"notcold"}
!35 = !{i64 -852997907418798798, i64 -2101080423462424381, i64 -6490791336773930154}
!36 = !{!37, !"cold"}
!37 = !{i64 -852997907418798798, i64 -2101080423462424381, i64 5188446645037944434}
!38 = !{i64 -852997907418798798}
