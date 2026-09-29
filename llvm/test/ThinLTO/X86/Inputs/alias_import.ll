target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

@globalfuncAlias = alias void (...), ptr @globalfunc
@globalfuncWeakAlias = weak alias void (...), ptr @globalfunc
@globalfuncLinkonceAlias = linkonce alias void (...), ptr @globalfunc
@globalfuncWeakODRAlias = weak_odr alias void (...), ptr @globalfunc
@globalfuncLinkonceODRAlias = linkonce_odr alias void (...), ptr @globalfunc
define hidden void @globalfunc() {
entry:
  ret void
}

@internalfuncAlias = alias void (...), ptr @internalfunc
@internalfuncWeakAlias = weak alias void (...), ptr @internalfunc
@internalfuncLinkonceAlias = linkonce alias void (...), ptr @internalfunc
@internalfuncWeakODRAlias = weak_odr alias void (...), ptr @internalfunc
@internalfuncLinkonceODRAlias = linkonce_odr alias void (...), ptr @internalfunc
define internal void @internalfunc() {
entry:
  ret void
}

@linkonceODRfuncAlias = alias void (...), ptr @linkonceODRfunc
@linkonceODRfuncWeakAlias = weak alias void (...), ptr @linkonceODRfunc
@linkonceODRfuncLinkonceAlias = linkonce alias void (...), ptr @linkonceODRfunc
@linkonceODRfuncWeakODRAlias = weak_odr alias void (...), ptr @linkonceODRfunc
@linkonceODRfuncLinkonceODRAlias = linkonce_odr alias void (...), ptr @linkonceODRfunc
define linkonce_odr void @linkonceODRfunc() {
entry:
  ret void
}

@weakODRfuncAlias = alias void (...), ptr @weakODRfunc
@weakODRfuncWeakAlias = weak alias void (...), ptr @weakODRfunc
@weakODRfuncLinkonceAlias = linkonce alias void (...), ptr @weakODRfunc
@weakODRfuncWeakODRAlias = weak_odr alias void (...), ptr @weakODRfunc
@weakODRfuncLinkonceODRAlias = linkonce_odr alias void (...), ptr @weakODRfunc
define weak_odr void @weakODRfunc() {
entry:
  ret void
}

@linkoncefuncAlias = alias void (...), ptr @linkoncefunc
@linkoncefuncWeakAlias = weak alias void (...), ptr @linkoncefunc
@linkoncefuncLinkonceAlias = linkonce alias void (...), ptr @linkoncefunc
@linkoncefuncWeakODRAlias = weak_odr alias void (...), ptr @linkoncefunc
@linkoncefuncLinkonceODRAlias = linkonce_odr alias void (...), ptr @linkoncefunc
define linkonce void @linkoncefunc() {
entry:
  ret void
}

@weakfuncAlias = alias void (...), ptr @weakfunc
@weakfuncWeakAlias = weak alias void (...), ptr @weakfunc
@weakfuncLinkonceAlias = linkonce alias void (...), ptr @weakfunc
@weakfuncWeakODRAlias = weak_odr alias void (...), ptr @weakfunc
@weakfuncLinkonceODRAlias = linkonce_odr alias void (...), ptr @weakfunc
define weak void @weakfunc() {
entry:
  ret void
}

