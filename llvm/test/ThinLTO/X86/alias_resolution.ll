; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/alias_resolution.ll -o %t2.bc
; RUN: llvm-lto -thinlto-action=thinlink -o %t.index.bc %t1.bc %t2.bc
; RUN: llvm-lto -thinlto-action=promote -thinlto-index %t.index.bc %t2.bc -o - | llvm-dis -o - | FileCheck %s --check-prefix=PROMOTE_MOD2 --check-prefix=NOTPROMOTED
; RUN: llvm-lto -thinlto-action=promote -thinlto-index %t.index.bc %t1.bc -o - | llvm-dis -o - | FileCheck %s --check-prefix=PROMOTE_MOD1 --check-prefix=NOTPROMOTED

; There is no importing going on with this IR, but let's check the ODR resolution for compile time

; NOTPROMOTED: @linkonceODRfuncAlias = alias void (...), bitcast (void ()* @linkonceODRfunc{{.*}} to void (...)*)
; NOTPROMOTED: @linkonceODRfuncWeakAlias = weak alias void (...), bitcast (void ()* @linkonceODRfunc{{.*}} to void (...)*)
; PROMOTE_MOD1: @linkonceODRfuncLinkonceAlias = weak alias void (...), bitcast (void ()* @linkonceODRfunc{{.*}} to void (...)*)
; PROMOTE_MOD2: @linkonceODRfuncLinkonceAlias = linkonce alias void (...), bitcast (void ()* @linkonceODRfunc{{.*}} to void (...)*)
; PROMOTE_MOD1: @linkonceODRfuncWeakODRAlias = weak_odr alias void (...), bitcast (void ()* @linkonceODRfunc.mod1 to void (...)*)
; PROMOTE_MOD2: @linkonceODRfuncWeakODRAlias = weak_odr alias void (...), bitcast (void ()* @linkonceODRfunc to void (...)*)
; PROMOTE_MOD1: @linkonceODRfuncLinkonceODRAlias = weak_odr alias void (...), bitcast (void ()* @linkonceODRfunc.mod1 to void (...)*)
; PROMOTE_MOD2: @linkonceODRfuncLinkonceODRAlias = linkonce_odr alias void (...), bitcast (void ()* @linkonceODRfunc to void (...)*)

; NOTPROMOTED: @weakODRfuncAlias = alias void (...), bitcast (void ()* @weakODRfunc{{.*}} to void (...)*)
; NOTPROMOTED: @weakODRfuncWeakAlias = weak alias void (...), bitcast (void ()* @weakODRfunc{{.*}} to void (...)*)
; PROMOTE_MOD1: @weakODRfuncLinkonceAlias = weak alias void (...), bitcast (void ()* @weakODRfunc{{.*}} to void (...)*)
; PROMOTE_MOD2: @weakODRfuncLinkonceAlias = linkonce alias void (...), bitcast (void ()* @weakODRfunc{{.*}} to void (...)*)
; PROMOTE_MOD1: @weakODRfuncWeakODRAlias = weak_odr alias void (...), bitcast (void ()* @weakODRfunc.mod1 to void (...)*)
; PROMOTE_MOD2: @weakODRfuncWeakODRAlias = weak_odr alias void (...), bitcast (void ()* @weakODRfunc to void (...)*)
; PROMOTE_MOD1: @weakODRfuncLinkonceODRAlias = weak_odr alias void (...), bitcast (void ()* @weakODRfunc.mod1 to void (...)*)
; PROMOTE_MOD2: @weakODRfuncLinkonceODRAlias = linkonce_odr alias void (...), bitcast (void ()* @weakODRfunc to void (...)*)

; NOTPROMOTED: @linkoncefuncAlias = alias void (...), bitcast (void ()* @linkoncefunc{{.*}} to void (...)*)
; NOTPROMOTED: @linkoncefuncWeakAlias = weak alias void (...), bitcast (void ()* @linkoncefunc{{.*}} to void (...)*)
; PROMOTE_MOD1: @linkoncefuncLinkonceAlias = weak alias void (...), bitcast (void ()* @linkoncefunc{{.*}} to void (...)*)
; PROMOTE_MOD2: @linkoncefuncLinkonceAlias = linkonce alias void (...), bitcast (void ()* @linkoncefunc{{.*}} to void (...)*)
; PROMOTE_MOD1: @linkoncefuncWeakODRAlias = weak_odr alias void (...), bitcast (void ()* @linkoncefunc.mod1 to void (...)*)
; PROMOTE_MOD2: @linkoncefuncWeakODRAlias = weak_odr alias void (...), bitcast (void ()* @linkoncefunc to void (...)*)
; PROMOTE_MOD1: @linkoncefuncLinkonceODRAlias = weak_odr alias void (...), bitcast (void ()* @linkoncefunc.mod1 to void (...)*)
; PROMOTE_MOD2: @linkoncefuncLinkonceODRAlias = linkonce_odr alias void (...), bitcast (void ()* @linkoncefunc to void (...)*)

; NOTPROMOTED: @weakfuncAlias = alias void (...), bitcast (void ()* @weakfunc{{.*}} to void (...)*)
; NOTPROMOTED: @weakfuncWeakAlias = weak alias void (...), bitcast (void ()* @weakfunc{{.*}} to void (...)*)
; PROMOTE_MOD1: @weakfuncLinkonceAlias = weak alias void (...), bitcast (void ()* @weakfunc{{.*}} to void (...)*)
; PROMOTE_MOD2: @weakfuncLinkonceAlias = linkonce alias void (...), bitcast (void ()* @weakfunc{{.*}} to void (...)*)
; FIXME: The "resolution" should turn one of these to linkonce_odr
; PROMOTE_MOD1: @weakfuncWeakODRAlias = weak_odr alias void (...), bitcast (void ()* @weakfunc.mod1 to void (...)*)
; PROMOTE_MOD2: @weakfuncWeakODRAlias = weak_odr alias void (...), bitcast (void ()* @weakfunc to void (...)*)
; PROMOTE_MOD1: @weakfuncLinkonceODRAlias = weak_odr alias void (...), bitcast (void ()* @weakfunc.mod1 to void (...)*)
; PROMOTE_MOD2: @weakfuncLinkonceODRAlias = linkonce_odr alias void (...), bitcast (void ()* @weakfunc to void (...)*)

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

@linkonceODRfuncAlias = alias void (...), bitcast (void ()* @linkonceODRfunc.mod1 to void (...)*)
@linkonceODRfuncWeakAlias = weak alias void (...), bitcast (void ()* @linkonceODRfunc.mod1 to void (...)*)
@linkonceODRfuncLinkonceAlias = linkonce alias void (...), bitcast (void ()* @linkonceODRfunc.mod1 to void (...)*)
@linkonceODRfuncWeakODRAlias = weak_odr alias void (...), bitcast (void ()* @linkonceODRfunc.mod1 to void (...)*)
@linkonceODRfuncLinkonceODRAlias = linkonce_odr alias void (...), bitcast (void ()* @linkonceODRfunc.mod1 to void (...)*)
define linkonce_odr void @linkonceODRfunc.mod1() {
entry:
  ret void
}

@weakODRfuncAlias = alias void (...), bitcast (void ()* @weakODRfunc.mod1 to void (...)*)
@weakODRfuncWeakAlias = weak alias void (...), bitcast (void ()* @weakODRfunc.mod1 to void (...)*)
@weakODRfuncLinkonceAlias = linkonce alias void (...), bitcast (void ()* @weakODRfunc.mod1 to void (...)*)
@weakODRfuncWeakODRAlias = weak_odr alias void (...), bitcast (void ()* @weakODRfunc.mod1 to void (...)*)
@weakODRfuncLinkonceODRAlias = linkonce_odr alias void (...), bitcast (void ()* @weakODRfunc.mod1 to void (...)*)
define weak_odr void @weakODRfunc.mod1() {
entry:
  ret void
}

@linkoncefuncAlias = alias void (...), bitcast (void ()* @linkoncefunc.mod1 to void (...)*)
@linkoncefuncWeakAlias = weak alias void (...), bitcast (void ()* @linkoncefunc.mod1 to void (...)*)
@linkoncefuncLinkonceAlias = linkonce alias void (...), bitcast (void ()* @linkoncefunc.mod1 to void (...)*)
@linkoncefuncWeakODRAlias = weak_odr alias void (...), bitcast (void ()* @linkoncefunc.mod1 to void (...)*)
@linkoncefuncLinkonceODRAlias = linkonce_odr alias void (...), bitcast (void ()* @linkoncefunc.mod1 to void (...)*)
define linkonce void @linkoncefunc.mod1() {
entry:
  ret void
}

@weakfuncAlias = alias void (...), bitcast (void ()* @weakfunc.mod1 to void (...)*)
@weakfuncWeakAlias = weak alias void (...), bitcast (void ()* @weakfunc.mod1 to void (...)*)
@weakfuncLinkonceAlias = linkonce alias void (...), bitcast (void ()* @weakfunc.mod1 to void (...)*)
@weakfuncWeakODRAlias = weak_odr alias void (...), bitcast (void ()* @weakfunc.mod1 to void (...)*)
@weakfuncLinkonceODRAlias = linkonce_odr alias void (...), bitcast (void ()* @weakfunc.mod1 to void (...)*)
define weak void @weakfunc.mod1() {
entry:
  ret void
}

