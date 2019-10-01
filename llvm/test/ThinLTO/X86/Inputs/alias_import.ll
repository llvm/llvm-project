target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

@globalfuncAlias = alias void (...), bitcast (void ()* @globalfunc to void (...)*)
@globalfuncWeakAlias = weak alias void (...), bitcast (void ()* @globalfunc to void (...)*)
@globalfuncLinkonceAlias = linkonce alias void (...), bitcast (void ()* @globalfunc to void (...)*)
@globalfuncWeakODRAlias = weak_odr alias void (...), bitcast (void ()* @globalfunc to void (...)*)
@globalfuncLinkonceODRAlias = linkonce_odr alias void (...), bitcast (void ()* @globalfunc to void (...)*)
define hidden void @globalfunc() {
entry:
  ret void
}

@internalfuncAlias = alias void (...), bitcast (void ()* @internalfunc to void (...)*)
@internalfuncWeakAlias = weak alias void (...), bitcast (void ()* @internalfunc to void (...)*)
@internalfuncLinkonceAlias = linkonce alias void (...), bitcast (void ()* @internalfunc to void (...)*)
@internalfuncWeakODRAlias = weak_odr alias void (...), bitcast (void ()* @internalfunc to void (...)*)
@internalfuncLinkonceODRAlias = linkonce_odr alias void (...), bitcast (void ()* @internalfunc to void (...)*)
define internal void @internalfunc() {
entry:
  ret void
}

@linkonceODRfuncAlias = alias void (...), bitcast (void ()* @linkonceODRfunc to void (...)*)
@linkonceODRfuncWeakAlias = weak alias void (...), bitcast (void ()* @linkonceODRfunc to void (...)*)
@linkonceODRfuncLinkonceAlias = linkonce alias void (...), bitcast (void ()* @linkonceODRfunc to void (...)*)
@linkonceODRfuncWeakODRAlias = weak_odr alias void (...), bitcast (void ()* @linkonceODRfunc to void (...)*)
@linkonceODRfuncLinkonceODRAlias = linkonce_odr alias void (...), bitcast (void ()* @linkonceODRfunc to void (...)*)
define linkonce_odr void @linkonceODRfunc() {
entry:
  ret void
}

@weakODRfuncAlias = alias void (...), bitcast (void ()* @weakODRfunc to void (...)*)
@weakODRfuncWeakAlias = weak alias void (...), bitcast (void ()* @weakODRfunc to void (...)*)
@weakODRfuncLinkonceAlias = linkonce alias void (...), bitcast (void ()* @weakODRfunc to void (...)*)
@weakODRfuncWeakODRAlias = weak_odr alias void (...), bitcast (void ()* @weakODRfunc to void (...)*)
@weakODRfuncLinkonceODRAlias = linkonce_odr alias void (...), bitcast (void ()* @weakODRfunc to void (...)*)
define weak_odr void @weakODRfunc() {
entry:
  ret void
}

@linkoncefuncAlias = alias void (...), bitcast (void ()* @linkoncefunc to void (...)*)
@linkoncefuncWeakAlias = weak alias void (...), bitcast (void ()* @linkoncefunc to void (...)*)
@linkoncefuncLinkonceAlias = linkonce alias void (...), bitcast (void ()* @linkoncefunc to void (...)*)
@linkoncefuncWeakODRAlias = weak_odr alias void (...), bitcast (void ()* @linkoncefunc to void (...)*)
@linkoncefuncLinkonceODRAlias = linkonce_odr alias void (...), bitcast (void ()* @linkoncefunc to void (...)*)
define linkonce void @linkoncefunc() {
entry:
  ret void
}

@weakfuncAlias = alias void (...), bitcast (void ()* @weakfunc to void (...)*)
@weakfuncWeakAlias = weak alias void (...), bitcast (void ()* @weakfunc to void (...)*)
@weakfuncLinkonceAlias = linkonce alias void (...), bitcast (void ()* @weakfunc to void (...)*)
@weakfuncWeakODRAlias = weak_odr alias void (...), bitcast (void ()* @weakfunc to void (...)*)
@weakfuncLinkonceODRAlias = linkonce_odr alias void (...), bitcast (void ()* @weakfunc to void (...)*)
define weak void @weakfunc() {
entry:
  ret void
}

