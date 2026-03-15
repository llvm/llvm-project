; RUN: opt  -module-summary %s -o %t1.bc
; RUN: opt  -module-summary %p/Inputs/alias_resolution.ll -o %t2.bc
; RUN: llvm-lto  -thinlto-action=thinlink -o %t.index.bc %t1.bc %t2.bc
; RUN: llvm-lto  -thinlto-action=promote -thinlto-index %t.index.bc %t2.bc -o - | llvm-dis  -o - | FileCheck %s --check-prefix=PROMOTE_MOD2 --check-prefix=NOTPROMOTED
; RUN: llvm-lto  -thinlto-action=promote -thinlto-index %t.index.bc %t1.bc -o - | llvm-dis  -o - | FileCheck %s --check-prefix=PROMOTE_MOD1 --check-prefix=NOTPROMOTED

; There is no importing going on with this IR, but let's check the ODR resolution for compile time

; NOTPROMOTED: @linkonceODRfuncAlias = alias void (...), ptr @linkonceODRfunc{{.*}}
; NOTPROMOTED: @linkonceODRfuncWeakAlias = weak alias void (...), ptr @linkonceODRfunc{{.*}}
; PROMOTE_MOD1: @linkonceODRfuncLinkonceAlias = weak alias void (...), ptr @linkonceODRfunc{{.*}}
; PROMOTE_MOD2: @linkonceODRfuncLinkonceAlias = linkonce alias void (...), ptr @linkonceODRfunc{{.*}}
; PROMOTE_MOD1: @linkonceODRfuncWeakODRAlias = weak_odr alias void (...), ptr @linkonceODRfunc.mod1
; PROMOTE_MOD2: @linkonceODRfuncWeakODRAlias = weak_odr alias void (...), ptr @linkonceODRfunc
; PROMOTE_MOD1: @linkonceODRfuncLinkonceODRAlias = weak_odr alias void (...), ptr @linkonceODRfunc.mod1
; PROMOTE_MOD2: @linkonceODRfuncLinkonceODRAlias = linkonce_odr alias void (...), ptr @linkonceODRfunc

; NOTPROMOTED: @weakODRfuncAlias = alias void (...), ptr @weakODRfunc{{.*}}
; NOTPROMOTED: @weakODRfuncWeakAlias = weak alias void (...), ptr @weakODRfunc{{.*}}
; PROMOTE_MOD1: @weakODRfuncLinkonceAlias = weak alias void (...), ptr @weakODRfunc{{.*}}
; PROMOTE_MOD2: @weakODRfuncLinkonceAlias = linkonce alias void (...), ptr @weakODRfunc{{.*}}
; PROMOTE_MOD1: @weakODRfuncWeakODRAlias = weak_odr alias void (...), ptr @weakODRfunc.mod1
; PROMOTE_MOD2: @weakODRfuncWeakODRAlias = weak_odr alias void (...), ptr @weakODRfunc
; PROMOTE_MOD1: @weakODRfuncLinkonceODRAlias = weak_odr alias void (...), ptr @weakODRfunc.mod1
; PROMOTE_MOD2: @weakODRfuncLinkonceODRAlias = linkonce_odr alias void (...), ptr @weakODRfunc

; NOTPROMOTED: @linkoncefuncAlias = alias void (...), ptr @linkoncefunc{{.*}}
; NOTPROMOTED: @linkoncefuncWeakAlias = weak alias void (...), ptr @linkoncefunc{{.*}}
; PROMOTE_MOD1: @linkoncefuncLinkonceAlias = weak alias void (...), ptr @linkoncefunc{{.*}}
; PROMOTE_MOD2: @linkoncefuncLinkonceAlias = linkonce alias void (...), ptr @linkoncefunc{{.*}}
; PROMOTE_MOD1: @linkoncefuncWeakODRAlias = weak_odr alias void (...), ptr @linkoncefunc.mod1
; PROMOTE_MOD2: @linkoncefuncWeakODRAlias = weak_odr alias void (...), ptr @linkoncefunc
; PROMOTE_MOD1: @linkoncefuncLinkonceODRAlias = weak_odr alias void (...), ptr @linkoncefunc.mod1
; PROMOTE_MOD2: @linkoncefuncLinkonceODRAlias = linkonce_odr alias void (...), ptr @linkoncefunc

; NOTPROMOTED: @weakfuncAlias = alias void (...), ptr @weakfunc{{.*}}
; NOTPROMOTED: @weakfuncWeakAlias = weak alias void (...), ptr @weakfunc{{.*}}
; PROMOTE_MOD1: @weakfuncLinkonceAlias = weak alias void (...), ptr @weakfunc{{.*}}
; PROMOTE_MOD2: @weakfuncLinkonceAlias = linkonce alias void (...), ptr @weakfunc{{.*}}
; FIXME: The "resolution" should turn one of these to linkonce_odr
; PROMOTE_MOD1: @weakfuncWeakODRAlias = weak_odr alias void (...), ptr @weakfunc.mod1
; PROMOTE_MOD2: @weakfuncWeakODRAlias = weak_odr alias void (...), ptr @weakfunc
; PROMOTE_MOD1: @weakfuncLinkonceODRAlias = weak_odr alias void (...), ptr @weakfunc.mod1
; PROMOTE_MOD2: @weakfuncLinkonceODRAlias = linkonce_odr alias void (...), ptr @weakfunc

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

@linkonceODRfuncAlias = alias void (...), ptr @linkonceODRfunc.mod1
@linkonceODRfuncWeakAlias = weak alias void (...), ptr @linkonceODRfunc.mod1
@linkonceODRfuncLinkonceAlias = linkonce alias void (...), ptr @linkonceODRfunc.mod1
@linkonceODRfuncWeakODRAlias = weak_odr alias void (...), ptr @linkonceODRfunc.mod1
@linkonceODRfuncLinkonceODRAlias = linkonce_odr alias void (...), ptr @linkonceODRfunc.mod1
define linkonce_odr void @linkonceODRfunc.mod1() {
entry:
  ret void
}

@weakODRfuncAlias = alias void (...), ptr @weakODRfunc.mod1
@weakODRfuncWeakAlias = weak alias void (...), ptr @weakODRfunc.mod1
@weakODRfuncLinkonceAlias = linkonce alias void (...), ptr @weakODRfunc.mod1
@weakODRfuncWeakODRAlias = weak_odr alias void (...), ptr @weakODRfunc.mod1
@weakODRfuncLinkonceODRAlias = linkonce_odr alias void (...), ptr @weakODRfunc.mod1
define weak_odr void @weakODRfunc.mod1() {
entry:
  ret void
}

@linkoncefuncAlias = alias void (...), ptr @linkoncefunc.mod1
@linkoncefuncWeakAlias = weak alias void (...), ptr @linkoncefunc.mod1
@linkoncefuncLinkonceAlias = linkonce alias void (...), ptr @linkoncefunc.mod1
@linkoncefuncWeakODRAlias = weak_odr alias void (...), ptr @linkoncefunc.mod1
@linkoncefuncLinkonceODRAlias = linkonce_odr alias void (...), ptr @linkoncefunc.mod1
define linkonce void @linkoncefunc.mod1() {
entry:
  ret void
}

@weakfuncAlias = alias void (...), ptr @weakfunc.mod1
@weakfuncWeakAlias = weak alias void (...), ptr @weakfunc.mod1
@weakfuncLinkonceAlias = linkonce alias void (...), ptr @weakfunc.mod1
@weakfuncWeakODRAlias = weak_odr alias void (...), ptr @weakfunc.mod1
@weakfuncLinkonceODRAlias = linkonce_odr alias void (...), ptr @weakfunc.mod1
define weak void @weakfunc.mod1() {
entry:
  ret void
}

