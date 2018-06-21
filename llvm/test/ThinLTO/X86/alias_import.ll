; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/alias_import.ll -o %t2.bc
; RUN: llvm-lto -thinlto-action=thinlink -o %t.index.bc %t1.bc %t2.bc
; RUN: llvm-lto -thinlto-action=promote -thinlto-index %t.index.bc %t2.bc -o - | llvm-dis -o - | FileCheck %s --check-prefix=PROMOTE
; RUN: llvm-lto -thinlto-action=import -thinlto-index %t.index.bc %t1.bc -o - | llvm-dis -o - | FileCheck %s --check-prefix=IMPORT

; Alias can't point to "available_externally", so they are implemented by
; importing the alias as an available_externally definition copied from the
; aliasee's body.
; PROMOTE-DAG: @globalfuncAlias = alias void (...), bitcast (void ()* @globalfunc to void (...)*)
; PROMOTE-DAG: @globalfuncWeakAlias = weak alias void (...), bitcast (void ()* @globalfunc to void (...)*)
; PROMOTE-DAG: @globalfuncLinkonceAlias = weak alias void (...), bitcast (void ()* @globalfunc to void (...)*)
; PROMOTE-DAG: @globalfuncWeakODRAlias = weak_odr alias void (...), bitcast (void ()* @globalfunc to void (...)*)
; PROMOTE-DAG: @globalfuncLinkonceODRAlias = weak_odr alias void (...), bitcast (void ()* @globalfunc to void (...)*)
; PROMOTE-DAG: @internalfuncAlias = alias void (...), bitcast (void ()* @internalfunc to void (...)*)
; PROMOTE-DAG: @internalfuncWeakAlias = weak alias void (...), bitcast (void ()* @internalfunc to void (...)*)
; PROMOTE-DAG: @internalfuncLinkonceAlias = weak alias void (...), bitcast (void ()* @internalfunc to void (...)*)
; PROMOTE-DAG: @internalfuncWeakODRAlias = weak_odr alias void (...), bitcast (void ()* @internalfunc to void (...)*)
; PROMOTE-DAG: @internalfuncLinkonceODRAlias = weak_odr alias void (...), bitcast (void ()* @internalfunc to void (...)*)
; PROMOTE-DAG: @linkoncefuncAlias = alias void (...), bitcast (void ()* @linkoncefunc to void (...)*)
; PROMOTE-DAG: @linkoncefuncWeakAlias = weak alias void (...), bitcast (void ()* @linkoncefunc to void (...)*)
; PROMOTE-DAG: @linkoncefuncLinkonceAlias = weak alias void (...), bitcast (void ()* @linkoncefunc to void (...)*)
; PROMOTE-DAG: @linkoncefuncWeakODRAlias = weak_odr alias void (...), bitcast (void ()* @linkoncefunc to void (...)*)
; PROMOTE-DAG: @linkoncefuncLinkonceODRAlias = weak_odr alias void (...), bitcast (void ()* @linkoncefunc to void (...)*)
; PROMOTE-DAG: @weakfuncAlias = alias void (...), bitcast (void ()* @weakfunc to void (...)*)
; PROMOTE-DAG: @weakfuncWeakAlias = weak alias void (...), bitcast (void ()* @weakfunc to void (...)*)
; PROMOTE-DAG: @weakfuncLinkonceAlias = weak alias void (...), bitcast (void ()* @weakfunc to void (...)*)
; PROMOTE-DAG: @weakfuncWeakODRAlias = weak_odr alias void (...), bitcast (void ()* @weakfunc to void (...)*)
; PROMOTE-DAG: @weakfuncLinkonceODRAlias = weak_odr alias void (...), bitcast (void ()* @weakfunc to void (...)*)
; PROMOTE-DAG: @weakODRfuncAlias = alias void (...), bitcast (void ()* @weakODRfunc to void (...)*)
; PROMOTE-DAG: @weakODRfuncWeakAlias = weak alias void (...), bitcast (void ()* @weakODRfunc to void (...)*)
; PROMOTE-DAG: @weakODRfuncLinkonceAlias = weak alias void (...), bitcast (void ()* @weakODRfunc to void (...)*)
; PROMOTE-DAG: @weakODRfuncWeakODRAlias = weak_odr alias void (...), bitcast (void ()* @weakODRfunc to void (...)*)
; PROMOTE-DAG: @weakODRfuncLinkonceODRAlias = weak_odr alias void (...), bitcast (void ()* @weakODRfunc to void (...)*)
; PROMOTE-DAG: @linkonceODRfuncAlias = alias void (...), bitcast (void ()* @linkonceODRfunc to void (...)*)
; PROMOTE-DAG: @linkonceODRfuncWeakAlias = weak alias void (...), bitcast (void ()* @linkonceODRfunc to void (...)*)
; PROMOTE-DAG: @linkonceODRfuncWeakODRAlias = weak_odr alias void (...), bitcast (void ()* @linkonceODRfunc to void (...)*)
; PROMOTE-DAG: @linkonceODRfuncLinkonceAlias = weak alias void (...), bitcast (void ()* @linkonceODRfunc to void (...)*)
; PROMOTE-DAG: @linkonceODRfuncLinkonceODRAlias = weak_odr alias void (...), bitcast (void ()* @linkonceODRfunc to void (...)*)

; PROMOTE-DAG: define void @globalfunc()
; PROMOTE-DAG: define internal void @internalfunc()
; PROMOTE-DAG: define weak_odr void @linkonceODRfunc()
; PROMOTE-DAG: define weak_odr void @weakODRfunc()
; PROMOTE-DAG: define weak void @linkoncefunc()
; PROMOTE-DAG: define weak void @weakfunc()

; On the import side now, verify that aliases are imported unless they
; are preemptible (non-ODR weak/linkonce).
; IMPORT-DAG: declare void @linkonceODRfuncWeakAlias
; IMPORT-DAG: declare void @linkonceODRfuncLinkonceAlias
; IMPORT-DAG: define available_externally void @linkonceODRfuncAlias
; IMPORT-DAG: define available_externally void @linkonceODRfuncWeakODRAlias
; IMPORT-DAG: define available_externally void @linkonceODRfuncLinkonceODRAlias
; IMPORT-DAG: define available_externally void @globalfuncAlias()
; IMPORT-DAG: declare void @globalfuncWeakAlias()
; IMPORT-DAG: declare void @globalfuncLinkonceAlias()
; IMPORT-DAG: define available_externally void @globalfuncWeakODRAlias()
; IMPORT-DAG: define available_externally void @globalfuncLinkonceODRAlias()
; IMPORT-DAG: define available_externally void @internalfuncAlias()
; IMPORT-DAG: declare void @internalfuncWeakAlias()
; IMPORT-DAG: declare void @internalfuncLinkonceAlias()
; IMPORT-DAG: define available_externally void @internalfuncWeakODRAlias()
; IMPORT-DAG: define available_externally void @internalfuncLinkonceODRAlias()
; IMPORT-DAG: define available_externally void @weakODRfuncAlias()
; IMPORT-DAG: declare void @weakODRfuncWeakAlias()
; IMPORT-DAG: declare void @weakODRfuncLinkonceAlias()
; IMPORT-DAG: define available_externally void @weakODRfuncWeakODRAlias()
; IMPORT-DAG: define available_externally void @weakODRfuncLinkonceODRAlias()
; IMPORT-DAG: define available_externally void @linkoncefuncAlias()
; IMPORT-DAG: declare void @linkoncefuncWeakAlias()
; IMPORT-DAG: declare void @linkoncefuncLinkonceAlias()
; IMPORT-DAG: define available_externally void @linkoncefuncWeakODRAlias()
; IMPORT-DAG: define available_externally void @linkoncefuncLinkonceODRAlias()
; IMPORT-DAG: define available_externally void @weakfuncAlias()
; IMPORT-DAG: declare void @weakfuncWeakAlias()
; IMPORT-DAG: declare void @weakfuncLinkonceAlias()
; IMPORT-DAG: define available_externally void @weakfuncWeakODRAlias()
; IMPORT-DAG: define available_externally void @weakfuncLinkonceODRAlias()
; IMPORT-DAG: define available_externally void @linkonceODRfuncAlias()
; IMPORT-DAG: declare void @linkonceODRfuncWeakAlias()
; IMPORT-DAG: define available_externally void @linkonceODRfuncWeakODRAlias()
; IMPORT-DAG: declare void @linkonceODRfuncLinkonceAlias()
; IMPORT-DAG: define available_externally void @linkonceODRfuncLinkonceODRAlias()

define i32 @main() #0 {
entry:
  call void @globalfuncAlias()
  call void @globalfuncWeakAlias()
  call void @globalfuncLinkonceAlias()
  call void @globalfuncWeakODRAlias()
  call void @globalfuncLinkonceODRAlias()

  call void @internalfuncAlias()
  call void @internalfuncWeakAlias()
  call void @internalfuncLinkonceAlias()
  call void @internalfuncWeakODRAlias()
  call void @internalfuncLinkonceODRAlias()
  call void @linkonceODRfuncAlias()
  call void @linkonceODRfuncWeakAlias()
  call void @linkonceODRfuncLinkonceAlias()
  call void @linkonceODRfuncWeakODRAlias()
  call void @linkonceODRfuncLinkonceODRAlias()

  call void @weakODRfuncAlias()
  call void @weakODRfuncWeakAlias()
  call void @weakODRfuncLinkonceAlias()
  call void @weakODRfuncWeakODRAlias()
  call void @weakODRfuncLinkonceODRAlias()

  call void @linkoncefuncAlias()
  call void @linkoncefuncWeakAlias()
  call void @linkoncefuncLinkonceAlias()
  call void @linkoncefuncWeakODRAlias()
  call void @linkoncefuncLinkonceODRAlias()

  call void @weakfuncAlias()
  call void @weakfuncWeakAlias()
  call void @weakfuncLinkonceAlias()
  call void @weakfuncWeakODRAlias()
  call void @weakfuncLinkonceODRAlias()

  ret i32 0
}


declare void @globalfuncAlias()
declare void @globalfuncWeakAlias()
declare void @globalfuncLinkonceAlias()
declare void @globalfuncWeakODRAlias()
declare void @globalfuncLinkonceODRAlias()

declare void @internalfuncAlias()
declare void @internalfuncWeakAlias()
declare void @internalfuncLinkonceAlias()
declare void @internalfuncWeakODRAlias()
declare void @internalfuncLinkonceODRAlias()

declare void @linkonceODRfuncAlias()
declare void @linkonceODRfuncWeakAlias()
declare void @linkonceODRfuncLinkonceAlias()
declare void @linkonceODRfuncWeakODRAlias()
declare void @linkonceODRfuncLinkonceODRAlias()

declare void @weakODRfuncAlias()
declare void @weakODRfuncWeakAlias()
declare void @weakODRfuncLinkonceAlias()
declare void @weakODRfuncWeakODRAlias()
declare void @weakODRfuncLinkonceODRAlias()

declare void @linkoncefuncAlias()
declare void @linkoncefuncWeakAlias()
declare void @linkoncefuncLinkonceAlias()
declare void @linkoncefuncWeakODRAlias()
declare void @linkoncefuncLinkonceODRAlias()

declare void @weakfuncAlias()
declare void @weakfuncWeakAlias()
declare void @weakfuncLinkonceAlias()
declare void @weakfuncWeakODRAlias()
declare void @weakfuncLinkonceODRAlias()


