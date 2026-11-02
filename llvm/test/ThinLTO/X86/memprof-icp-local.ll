;; Test that MemProf cloning and ICP work for an indirect call to a target with
;; local linkage, when the VP metadata contains the GUID of the target's
;; original name without the source file prefix (as is the case when the VP
;; metadata is synthesized from the MemProf profile's callee GUIDs, or comes
;; from a SamplePGO profile). The thin link should translate the original name
;; GUID to the actual GUID of the local target, both for the call edges and for
;; the synthesized callsite records, and the ThinLTO backend should then be
;; able to locate the (promoted) target function for ICP.

; REQUIRES: asserts

; RUN: split-file %s %t
; RUN: opt -thinlto-bc %t/main.ll -o %t/main.o
; RUN: opt -thinlto-bc %t/foo.ll -o %t/foo.o

;; In-process ThinLTO.
; RUN: llvm-lto2 run %t/main.o %t/foo.o -enable-memprof-context-disambiguation \
; RUN:	-enable-memprof-indirect-call-support=true \
; RUN:  -supports-hot-cold-new \
; RUN:  -r=%t/foo.o,_Z3fooPv,plx \
; RUN:  -r=%t/main.o,main,plx \
; RUN:  -r=%t/main.o,_Z3fooPv, \
; RUN:  -r=%t/main.o,_Znwm, \
; RUN:	-thinlto-threads=1 \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes \
; RUN:  -pass-remarks=memprof-context-disambiguation \
; RUN:  -pass-remarks-missed=memprof-context-disambiguation \
; RUN:  -save-temps \
; RUN:  -o %t.out 2>&1 | FileCheck %s --check-prefix=REMARKS \
; RUN:	--implicit-check-not="cannot promote"

; RUN: llvm-dis %t.out.2.4.opt.bc -o - | FileCheck %s --check-prefix=IR

;; Distributed ThinLTO.
; RUN: llvm-lto2 run %t/main.o %t/foo.o -enable-memprof-context-disambiguation \
; RUN:	-enable-memprof-indirect-call-support=true \
; RUN:  -supports-hot-cold-new \
; RUN:  -thinlto-distributed-indexes \
; RUN:  -r=%t/foo.o,_Z3fooPv,plx \
; RUN:  -r=%t/main.o,main,plx \
; RUN:  -r=%t/main.o,_Z3fooPv, \
; RUN:  -r=%t/main.o,_Znwm, \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes \
; RUN:  -o %t.out

;; Run ThinLTO backend for foo.o.
; RUN: opt -import-all-index -passes=function-import,memprof-context-disambiguation \
; RUN:  -enable-memprof-indirect-call-support=true \
; RUN:  -summary-file=%t/foo.o.thinlto.bc -memprof-import-summary=%t/foo.o.thinlto.bc \
; RUN:  -enable-import-metadata \
; RUN:  -pass-remarks=memprof-context-disambiguation \
; RUN:  -pass-remarks-missed=memprof-context-disambiguation \
; RUN:  %t/foo.o -S 2>&1 | FileCheck %s --check-prefix=REMARKS-FOO \
; RUN:	--check-prefix=IR --implicit-check-not="cannot promote"

;; The local bar is promoted (and renamed) since it is imported into foo's
;; module for ICP.
; REMARKS-DAG: created clone _ZL3barPvj.llvm.{{[0-9]+}}.memprof.1
; REMARKS-DAG: call in clone main assigned to call function clone _Z3fooPv.memprof.1
; REMARKS-FOO: created clone _Z3fooPv.memprof.1
; REMARKS-FOO: call in clone _Z3fooPv promoted and assigned to call function clone _ZL3barPvj.llvm.{{[0-9]+}}{{$}}
; REMARKS-FOO: call in clone _Z3fooPv.memprof.1 promoted and assigned to call function clone _ZL3barPvj.llvm.{{[0-9]+}}.memprof.1
; REMARKS: created clone _Z3fooPv.memprof.1
; REMARKS: call in clone _Z3fooPv promoted and assigned to call function clone _ZL3barPvj.llvm.{{[0-9]+}}{{$}}
; REMARKS: call in clone _Z3fooPv.memprof.1 promoted and assigned to call function clone _ZL3barPvj.llvm.{{[0-9]+}}.memprof.1

; IR: define {{.*}} @_Z3fooPv(
; IR:   icmp eq ptr %0, @[[BAR:_ZL3barPvj.llvm.[0-9]+]]
; IR: define {{.*}} @_Z3fooPv.memprof.1(
;; We should still compare against the original version of bar (the call is
;; updated to the cloned version as checked in the remarks above).
; IR:   icmp eq ptr %0, @[[BAR]]{{$}}

;; Test the case where the local bar is too large to be imported into foo's
;; module (simulated with -import-instr-limit=0). A declaration of bar is
;; imported into foo's module instead, which requires bar to be exported from
;; main's module, and therefore promoted (and renamed).
; RUN: llvm-lto2 run %t/main.o %t/foo.o -enable-memprof-context-disambiguation \
; RUN:	-enable-memprof-indirect-call-support=true \
; RUN:  -supports-hot-cold-new -import-instr-limit=0 \
; RUN:  -r=%t/foo.o,_Z3fooPv,plx \
; RUN:  -r=%t/main.o,main,plx \
; RUN:  -r=%t/main.o,_Z3fooPv, \
; RUN:  -r=%t/main.o,_Znwm, \
; RUN:	-thinlto-threads=1 \
; RUN:  -memprof-verify-ccg -memprof-verify-nodes \
; RUN:  -pass-remarks=memprof-context-disambiguation \
; RUN:  -pass-remarks-missed=memprof-context-disambiguation \
; RUN:  -save-temps \
; RUN:  -o %t.decl.out 2>&1 | FileCheck %s --check-prefix=REMARKS-DECL \
; RUN:	--implicit-check-not="cannot promote"
; RUN: llvm-dis %t.decl.out.2.3.import.bc -o - | FileCheck %s --check-prefix=IR-DECL
; RUN: llvm-nm %t.decl.out.1 | FileCheck %s --check-prefix=NM-DECL

; REMARKS-DECL-DAG: created clone _ZL3barPvj.llvm.{{[0-9]+}}.memprof.1
; REMARKS-DECL-DAG: call in clone _Z3fooPv promoted and assigned to call function clone _ZL3barPvj.llvm.{{[0-9]+}}{{$}}
; REMARKS-DECL-DAG: call in clone _Z3fooPv.memprof.1 promoted and assigned to call function clone _ZL3barPvj.llvm.{{[0-9]+}}.memprof.1

;; The declaration of the promoted bar imported into foo's module.
; IR-DECL: declare !guid !{{[0-9]+}} {{.*}}i32 @_ZL3barPvj.llvm.{{[0-9]+}}(ptr, i32)

;; The promoted bar and its clone are defined in main's module.
; NM-DECL-DAG: T _ZL3barPvj.llvm.{{[0-9]+}}{{$}}
; NM-DECL-DAG: T _ZL3barPvj.llvm.{{[0-9]+}}.memprof.1

;; Without importing the declaration we can't promote, and bar is not promoted
;; or exported, and is subsequently removed as dead.
; RUN: llvm-lto2 run %t/main.o %t/foo.o -enable-memprof-context-disambiguation \
; RUN:	-enable-memprof-indirect-call-support=true \
; RUN:  -supports-hot-cold-new -import-instr-limit=0 \
; RUN:  -memprof-import-cloned-indirect-targets=false \
; RUN:  -r=%t/foo.o,_Z3fooPv,plx \
; RUN:  -r=%t/main.o,main,plx \
; RUN:  -r=%t/main.o,_Z3fooPv, \
; RUN:  -r=%t/main.o,_Znwm, \
; RUN:	-thinlto-threads=1 \
; RUN:  -pass-remarks-missed=memprof-context-disambiguation \
; RUN:  -o %t.nodecl.out 2>&1 | FileCheck %s --check-prefix=REMARKS-NODECL
; RUN: llvm-nm %t.nodecl.out.1 | FileCheck %s --check-prefix=NM-NODECL

; REMARKS-NODECL: Memprof cannot promote indirect call: target with md5sum 13717123755983020101 not found
; NM-NODECL-NOT: _ZL3barPvj
; NM-NODECL: T main

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

;; VP metadata with a single candidate, the original name GUID of the local
;; function _ZL3barPvj (i.e. MD5("_ZL3barPvj"), without the "main.cc;" prefix).
!1 = !{!"VP", i32 0, i64 1, i64 -4729620317726531515, i64 1}
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

;; Local function only called indirectly.
define internal i32 @_ZL3barPvj(ptr %this, i32 %s) {
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
