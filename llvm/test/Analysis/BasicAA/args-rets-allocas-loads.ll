; RUN: opt -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output < %s 2>&1 | FileCheck  %s

declare void @callee(ptr %callee_arg)
declare void @nocap_callee(ptr nocapture %nocap_callee_arg)

declare ptr @normal_returner()
declare noalias ptr @noalias_returner()

define void @caller_a(ptr %arg_a0,
                      ptr %arg_a1,
                      ptr noalias %noalias_arg_a0,
                      ptr noalias %noalias_arg_a1,
                      ptr %indirect_a0,
                      ptr %indirect_a1) {
  %escape_alloca_a0 = alloca double
  %escape_alloca_a1 = alloca double
  %noescape_alloca_a0 = alloca double
  %noescape_alloca_a1 = alloca double

  %normal_ret_a0 = call ptr @normal_returner()
  %normal_ret_a1 = call ptr @normal_returner()
  %noalias_ret_a0 = call ptr @noalias_returner()
  %noalias_ret_a1 = call ptr @noalias_returner()

  %loaded_a0 = load ptr, ptr %indirect_a0
  %loaded_a1 = load ptr, ptr %indirect_a1

  call void @callee(ptr %escape_alloca_a0)
  call void @callee(ptr %escape_alloca_a1)
  call void @nocap_callee(ptr %noescape_alloca_a0)
  call void @nocap_callee(ptr %noescape_alloca_a1)

  store double 0.0, ptr %loaded_a0
  store double 0.0, ptr %loaded_a1
  store double 0.0, ptr %arg_a0
  store double 0.0, ptr %arg_a1
  store double 0.0, ptr %noalias_arg_a0
  store double 0.0, ptr %noalias_arg_a1
  store double 0.0, ptr %escape_alloca_a0
  store double 0.0, ptr %escape_alloca_a1
  store double 0.0, ptr %noescape_alloca_a0
  store double 0.0, ptr %noescape_alloca_a1
  store double 0.0, ptr %normal_ret_a0
  store double 0.0, ptr %normal_ret_a1
  store double 0.0, ptr %noalias_ret_a0
  store double 0.0, ptr %noalias_ret_a1
  ret void
}

; CHECK: Function: caller_a: 16 pointers, 8 call sites
; CHECK: MayAlias:	ptr* %indirect_a0, ptr* %indirect_a1
; CHECK: MayAlias:	ptr* %indirect_a0, double* %loaded_a0
; CHECK: MayAlias:	ptr* %indirect_a1, double* %loaded_a0
; CHECK: MayAlias:	ptr* %indirect_a0, double* %loaded_a1
; CHECK: MayAlias:	ptr* %indirect_a1, double* %loaded_a1
; CHECK: MayAlias:	double* %loaded_a0, double* %loaded_a1
; CHECK: MayAlias:	double* %arg_a0, ptr* %indirect_a0
; CHECK: MayAlias:	double* %arg_a0, ptr* %indirect_a1
; CHECK: MayAlias:	double* %arg_a0, double* %loaded_a0
; CHECK: MayAlias:	double* %arg_a0, double* %loaded_a1
; CHECK: MayAlias:	double* %arg_a1, ptr* %indirect_a0
; CHECK: MayAlias:	double* %arg_a1, ptr* %indirect_a1
; CHECK: MayAlias:	double* %arg_a1, double* %loaded_a0
; CHECK: MayAlias:	double* %arg_a1, double* %loaded_a1
; CHECK: MayAlias:	double* %arg_a0, double* %arg_a1
; CHECK: NoAlias:	ptr* %indirect_a0, double* %noalias_arg_a0
; CHECK: NoAlias:	ptr* %indirect_a1, double* %noalias_arg_a0
; CHECK: NoAlias:	double* %loaded_a0, double* %noalias_arg_a0
; CHECK: NoAlias:	double* %loaded_a1, double* %noalias_arg_a0
; CHECK: NoAlias:	double* %arg_a0, double* %noalias_arg_a0
; CHECK: NoAlias:	double* %arg_a1, double* %noalias_arg_a0
; CHECK: NoAlias:	ptr* %indirect_a0, double* %noalias_arg_a1
; CHECK: NoAlias:	ptr* %indirect_a1, double* %noalias_arg_a1
; CHECK: NoAlias:	double* %loaded_a0, double* %noalias_arg_a1
; CHECK: NoAlias:	double* %loaded_a1, double* %noalias_arg_a1
; CHECK: NoAlias:	double* %arg_a0, double* %noalias_arg_a1
; CHECK: NoAlias:	double* %arg_a1, double* %noalias_arg_a1
; CHECK: NoAlias:	double* %noalias_arg_a0, double* %noalias_arg_a1
; CHECK: NoAlias:	double* %escape_alloca_a0, ptr* %indirect_a0
; CHECK: NoAlias:	double* %escape_alloca_a0, ptr* %indirect_a1
; CHECK: MayAlias:	double* %escape_alloca_a0, double* %loaded_a0
; CHECK: MayAlias:	double* %escape_alloca_a0, double* %loaded_a1
; CHECK: NoAlias:	double* %arg_a0, double* %escape_alloca_a0
; CHECK: NoAlias:	double* %arg_a1, double* %escape_alloca_a0
; CHECK: NoAlias:	double* %escape_alloca_a0, double* %noalias_arg_a0
; CHECK: NoAlias:	double* %escape_alloca_a0, double* %noalias_arg_a1
; CHECK: NoAlias:	double* %escape_alloca_a1, ptr* %indirect_a0
; CHECK: NoAlias:	double* %escape_alloca_a1, ptr* %indirect_a1
; CHECK: MayAlias:	double* %escape_alloca_a1, double* %loaded_a0
; CHECK: MayAlias:	double* %escape_alloca_a1, double* %loaded_a1
; CHECK: NoAlias:	double* %arg_a0, double* %escape_alloca_a1
; CHECK: NoAlias:	double* %arg_a1, double* %escape_alloca_a1
; CHECK: NoAlias:	double* %escape_alloca_a1, double* %noalias_arg_a0
; CHECK: NoAlias:	double* %escape_alloca_a1, double* %noalias_arg_a1
; CHECK: NoAlias:	double* %escape_alloca_a0, double* %escape_alloca_a1
; CHECK: NoAlias:	ptr* %indirect_a0, double* %noescape_alloca_a0
; CHECK: NoAlias:	ptr* %indirect_a1, double* %noescape_alloca_a0
; CHECK: NoAlias:	double* %loaded_a0, double* %noescape_alloca_a0
; CHECK: NoAlias:	double* %loaded_a1, double* %noescape_alloca_a0
; CHECK: NoAlias:	double* %arg_a0, double* %noescape_alloca_a0
; CHECK: NoAlias:	double* %arg_a1, double* %noescape_alloca_a0
; CHECK: NoAlias:	double* %noalias_arg_a0, double* %noescape_alloca_a0
; CHECK: NoAlias:	double* %noalias_arg_a1, double* %noescape_alloca_a0
; CHECK: NoAlias:	double* %escape_alloca_a0, double* %noescape_alloca_a0
; CHECK: NoAlias:	double* %escape_alloca_a1, double* %noescape_alloca_a0
; CHECK: NoAlias:	ptr* %indirect_a0, double* %noescape_alloca_a1
; CHECK: NoAlias:	ptr* %indirect_a1, double* %noescape_alloca_a1
; CHECK: NoAlias:	double* %loaded_a0, double* %noescape_alloca_a1
; CHECK: NoAlias:	double* %loaded_a1, double* %noescape_alloca_a1
; CHECK: NoAlias:	double* %arg_a0, double* %noescape_alloca_a1
; CHECK: NoAlias:	double* %arg_a1, double* %noescape_alloca_a1
; CHECK: NoAlias:	double* %noalias_arg_a0, double* %noescape_alloca_a1
; CHECK: NoAlias:	double* %noalias_arg_a1, double* %noescape_alloca_a1
; CHECK: NoAlias:	double* %escape_alloca_a0, double* %noescape_alloca_a1
; CHECK: NoAlias:	double* %escape_alloca_a1, double* %noescape_alloca_a1
; CHECK: NoAlias:	double* %noescape_alloca_a0, double* %noescape_alloca_a1
; CHECK: MayAlias:	ptr* %indirect_a0, double* %normal_ret_a0
; CHECK: MayAlias:	ptr* %indirect_a1, double* %normal_ret_a0
; CHECK: MayAlias:	double* %loaded_a0, double* %normal_ret_a0
; CHECK: MayAlias:	double* %loaded_a1, double* %normal_ret_a0
; CHECK: MayAlias:	double* %arg_a0, double* %normal_ret_a0
; CHECK: MayAlias:	double* %arg_a1, double* %normal_ret_a0
; CHECK: NoAlias:	double* %noalias_arg_a0, double* %normal_ret_a0
; CHECK: NoAlias:	double* %noalias_arg_a1, double* %normal_ret_a0
; CHECK: MayAlias:	double* %escape_alloca_a0, double* %normal_ret_a0
; CHECK: MayAlias:	double* %escape_alloca_a1, double* %normal_ret_a0
; CHECK: NoAlias:	double* %noescape_alloca_a0, double* %normal_ret_a0
; CHECK: NoAlias:	double* %noescape_alloca_a1, double* %normal_ret_a0
; CHECK: MayAlias:	ptr* %indirect_a0, double* %normal_ret_a1
; CHECK: MayAlias:	ptr* %indirect_a1, double* %normal_ret_a1
; CHECK: MayAlias:	double* %loaded_a0, double* %normal_ret_a1
; CHECK: MayAlias:	double* %loaded_a1, double* %normal_ret_a1
; CHECK: MayAlias:	double* %arg_a0, double* %normal_ret_a1
; CHECK: MayAlias:	double* %arg_a1, double* %normal_ret_a1
; CHECK: NoAlias:	double* %noalias_arg_a0, double* %normal_ret_a1
; CHECK: NoAlias:	double* %noalias_arg_a1, double* %normal_ret_a1
; CHECK: MayAlias:	double* %escape_alloca_a0, double* %normal_ret_a1
; CHECK: MayAlias:	double* %escape_alloca_a1, double* %normal_ret_a1
; CHECK: NoAlias:	double* %noescape_alloca_a0, double* %normal_ret_a1
; CHECK: NoAlias:	double* %noescape_alloca_a1, double* %normal_ret_a1
; CHECK: MayAlias:	double* %normal_ret_a0, double* %normal_ret_a1
; CHECK: NoAlias:	ptr* %indirect_a0, double* %noalias_ret_a0
; CHECK: NoAlias:	ptr* %indirect_a1, double* %noalias_ret_a0
; CHECK: NoAlias:	double* %loaded_a0, double* %noalias_ret_a0
; CHECK: NoAlias:	double* %loaded_a1, double* %noalias_ret_a0
; CHECK: NoAlias:	double* %arg_a0, double* %noalias_ret_a0
; CHECK: NoAlias:	double* %arg_a1, double* %noalias_ret_a0
; CHECK: NoAlias:	double* %noalias_arg_a0, double* %noalias_ret_a0
; CHECK: NoAlias:	double* %noalias_arg_a1, double* %noalias_ret_a0
; CHECK: NoAlias:	double* %escape_alloca_a0, double* %noalias_ret_a0
; CHECK: NoAlias:	double* %escape_alloca_a1, double* %noalias_ret_a0
; CHECK: NoAlias:	double* %noalias_ret_a0, double* %noescape_alloca_a0
; CHECK: NoAlias:	double* %noalias_ret_a0, double* %noescape_alloca_a1
; CHECK: NoAlias:	double* %noalias_ret_a0, double* %normal_ret_a0
; CHECK: NoAlias:	double* %noalias_ret_a0, double* %normal_ret_a1
; CHECK: NoAlias:	ptr* %indirect_a0, double* %noalias_ret_a1
; CHECK: NoAlias:	ptr* %indirect_a1, double* %noalias_ret_a1
; CHECK: NoAlias:	double* %loaded_a0, double* %noalias_ret_a1
; CHECK: NoAlias:	double* %loaded_a1, double* %noalias_ret_a1
; CHECK: NoAlias:	double* %arg_a0, double* %noalias_ret_a1
; CHECK: NoAlias:	double* %arg_a1, double* %noalias_ret_a1
; CHECK: NoAlias:	double* %noalias_arg_a0, double* %noalias_ret_a1
; CHECK: NoAlias:	double* %noalias_arg_a1, double* %noalias_ret_a1
; CHECK: NoAlias:	double* %escape_alloca_a0, double* %noalias_ret_a1
; CHECK: NoAlias:	double* %escape_alloca_a1, double* %noalias_ret_a1
; CHECK: NoAlias:	double* %noalias_ret_a1, double* %noescape_alloca_a0
; CHECK: NoAlias:	double* %noalias_ret_a1, double* %noescape_alloca_a1
; CHECK: NoAlias:	double* %noalias_ret_a1, double* %normal_ret_a0
; CHECK: NoAlias:	double* %noalias_ret_a1, double* %normal_ret_a1
; CHECK: NoAlias:	double* %noalias_ret_a0, double* %noalias_ret_a1
; CHECK: Both ModRef:  Ptr: ptr* %indirect_a0	<->  %normal_ret_a0 = call ptr @normal_returner()
; CHECK: Both ModRef:  Ptr: ptr* %indirect_a1	<->  %normal_ret_a0 = call ptr @normal_returner()
; CHECK: Both ModRef:  Ptr: double* %loaded_a0	<->  %normal_ret_a0 = call ptr @normal_returner()
; CHECK: Both ModRef:  Ptr: double* %loaded_a1	<->  %normal_ret_a0 = call ptr @normal_returner()
; CHECK: Both ModRef:  Ptr: double* %arg_a0	<->  %normal_ret_a0 = call ptr @normal_returner()
; CHECK: Both ModRef:  Ptr: double* %arg_a1	<->  %normal_ret_a0 = call ptr @normal_returner()
; CHECK: NoModRef:  Ptr: double* %noalias_arg_a0	<->  %normal_ret_a0 = call ptr @normal_returner()
; CHECK: NoModRef:  Ptr: double* %noalias_arg_a1	<->  %normal_ret_a0 = call ptr @normal_returner()
; CHECK: Both ModRef:  Ptr: double* %escape_alloca_a0	<->  %normal_ret_a0 = call ptr @normal_returner()
; CHECK: Both ModRef:  Ptr: double* %escape_alloca_a1	<->  %normal_ret_a0 = call ptr @normal_returner()
; CHECK: NoModRef:  Ptr: double* %noescape_alloca_a0	<->  %normal_ret_a0 = call ptr @normal_returner()
; CHECK: NoModRef:  Ptr: double* %noescape_alloca_a1	<->  %normal_ret_a0 = call ptr @normal_returner()
; CHECK: Both ModRef:  Ptr: double* %normal_ret_a0	<->  %normal_ret_a0 = call ptr @normal_returner()
; CHECK: Both ModRef:  Ptr: double* %normal_ret_a1	<->  %normal_ret_a0 = call ptr @normal_returner()
; CHECK: NoModRef:  Ptr: double* %noalias_ret_a0	<->  %normal_ret_a0 = call ptr @normal_returner()
; CHECK: NoModRef:  Ptr: double* %noalias_ret_a1	<->  %normal_ret_a0 = call ptr @normal_returner()
; CHECK: Both ModRef:  Ptr: ptr* %indirect_a0	<->  %normal_ret_a1 = call ptr @normal_returner()
; CHECK: Both ModRef:  Ptr: ptr* %indirect_a1	<->  %normal_ret_a1 = call ptr @normal_returner()
; CHECK: Both ModRef:  Ptr: double* %loaded_a0	<->  %normal_ret_a1 = call ptr @normal_returner()
; CHECK: Both ModRef:  Ptr: double* %loaded_a1	<->  %normal_ret_a1 = call ptr @normal_returner()
; CHECK: Both ModRef:  Ptr: double* %arg_a0	<->  %normal_ret_a1 = call ptr @normal_returner()
; CHECK: Both ModRef:  Ptr: double* %arg_a1	<->  %normal_ret_a1 = call ptr @normal_returner()
; CHECK: NoModRef:  Ptr: double* %noalias_arg_a0	<->  %normal_ret_a1 = call ptr @normal_returner()
; CHECK: NoModRef:  Ptr: double* %noalias_arg_a1	<->  %normal_ret_a1 = call ptr @normal_returner()
; CHECK: Both ModRef:  Ptr: double* %escape_alloca_a0	<->  %normal_ret_a1 = call ptr @normal_returner()
; CHECK: Both ModRef:  Ptr: double* %escape_alloca_a1	<->  %normal_ret_a1 = call ptr @normal_returner()
; CHECK: NoModRef:  Ptr: double* %noescape_alloca_a0	<->  %normal_ret_a1 = call ptr @normal_returner()
; CHECK: NoModRef:  Ptr: double* %noescape_alloca_a1	<->  %normal_ret_a1 = call ptr @normal_returner()
; CHECK: Both ModRef:  Ptr: double* %normal_ret_a0	<->  %normal_ret_a1 = call ptr @normal_returner()
; CHECK: Both ModRef:  Ptr: double* %normal_ret_a1	<->  %normal_ret_a1 = call ptr @normal_returner()
; CHECK: NoModRef:  Ptr: double* %noalias_ret_a0	<->  %normal_ret_a1 = call ptr @normal_returner()
; CHECK: NoModRef:  Ptr: double* %noalias_ret_a1	<->  %normal_ret_a1 = call ptr @normal_returner()
; CHECK: Both ModRef:  Ptr: ptr* %indirect_a0	<->  %noalias_ret_a0 = call ptr @noalias_returner()
; CHECK: Both ModRef:  Ptr: ptr* %indirect_a1	<->  %noalias_ret_a0 = call ptr @noalias_returner()
; CHECK: Both ModRef:  Ptr: double* %loaded_a0	<->  %noalias_ret_a0 = call ptr @noalias_returner()
; CHECK: Both ModRef:  Ptr: double* %loaded_a1	<->  %noalias_ret_a0 = call ptr @noalias_returner()
; CHECK: Both ModRef:  Ptr: double* %arg_a0	<->  %noalias_ret_a0 = call ptr @noalias_returner()
; CHECK: Both ModRef:  Ptr: double* %arg_a1	<->  %noalias_ret_a0 = call ptr @noalias_returner()
; CHECK: NoModRef:  Ptr: double* %noalias_arg_a0	<->  %noalias_ret_a0 = call ptr @noalias_returner()
; CHECK: NoModRef:  Ptr: double* %noalias_arg_a1	<->  %noalias_ret_a0 = call ptr @noalias_returner()
; CHECK: Both ModRef:  Ptr: double* %escape_alloca_a0	<->  %noalias_ret_a0 = call ptr @noalias_returner()
; CHECK: Both ModRef:  Ptr: double* %escape_alloca_a1	<->  %noalias_ret_a0 = call ptr @noalias_returner()
; CHECK: NoModRef:  Ptr: double* %noescape_alloca_a0	<->  %noalias_ret_a0 = call ptr @noalias_returner()
; CHECK: NoModRef:  Ptr: double* %noescape_alloca_a1	<->  %noalias_ret_a0 = call ptr @noalias_returner()
; CHECK: Both ModRef:  Ptr: double* %normal_ret_a0	<->  %noalias_ret_a0 = call ptr @noalias_returner()
; CHECK: Both ModRef:  Ptr: double* %normal_ret_a1	<->  %noalias_ret_a0 = call ptr @noalias_returner()
; CHECK: Both ModRef:  Ptr: double* %noalias_ret_a0	<->  %noalias_ret_a0 = call ptr @noalias_returner()
; CHECK: NoModRef:  Ptr: double* %noalias_ret_a1	<->  %noalias_ret_a0 = call ptr @noalias_returner()
; CHECK: Both ModRef:  Ptr: ptr* %indirect_a0	<->  %noalias_ret_a1 = call ptr @noalias_returner()
; CHECK: Both ModRef:  Ptr: ptr* %indirect_a1	<->  %noalias_ret_a1 = call ptr @noalias_returner()
; CHECK: Both ModRef:  Ptr: double* %loaded_a0	<->  %noalias_ret_a1 = call ptr @noalias_returner()
; CHECK: Both ModRef:  Ptr: double* %loaded_a1	<->  %noalias_ret_a1 = call ptr @noalias_returner()
; CHECK: Both ModRef:  Ptr: double* %arg_a0	<->  %noalias_ret_a1 = call ptr @noalias_returner()
; CHECK: Both ModRef:  Ptr: double* %arg_a1	<->  %noalias_ret_a1 = call ptr @noalias_returner()
; CHECK: NoModRef:  Ptr: double* %noalias_arg_a0	<->  %noalias_ret_a1 = call ptr @noalias_returner()
; CHECK: NoModRef:  Ptr: double* %noalias_arg_a1	<->  %noalias_ret_a1 = call ptr @noalias_returner()
; CHECK: Both ModRef:  Ptr: double* %escape_alloca_a0	<->  %noalias_ret_a1 = call ptr @noalias_returner()
; CHECK: Both ModRef:  Ptr: double* %escape_alloca_a1	<->  %noalias_ret_a1 = call ptr @noalias_returner()
; CHECK: NoModRef:  Ptr: double* %noescape_alloca_a0	<->  %noalias_ret_a1 = call ptr @noalias_returner()
; CHECK: NoModRef:  Ptr: double* %noescape_alloca_a1	<->  %noalias_ret_a1 = call ptr @noalias_returner()
; CHECK: Both ModRef:  Ptr: double* %normal_ret_a0	<->  %noalias_ret_a1 = call ptr @noalias_returner()
; CHECK: Both ModRef:  Ptr: double* %normal_ret_a1	<->  %noalias_ret_a1 = call ptr @noalias_returner()
; CHECK: NoModRef:  Ptr: double* %noalias_ret_a0	<->  %noalias_ret_a1 = call ptr @noalias_returner()
; CHECK: Both ModRef:  Ptr: double* %noalias_ret_a1	<->  %noalias_ret_a1 = call ptr @noalias_returner()
; CHECK: Both ModRef:  Ptr: ptr* %indirect_a0	<->  call void @callee(ptr %escape_alloca_a0)
; CHECK: Both ModRef:  Ptr: ptr* %indirect_a1	<->  call void @callee(ptr %escape_alloca_a0)
; CHECK: Both ModRef:  Ptr: double* %loaded_a0	<->  call void @callee(ptr %escape_alloca_a0)
; CHECK: Both ModRef:  Ptr: double* %loaded_a1	<->  call void @callee(ptr %escape_alloca_a0)
; CHECK: Both ModRef:  Ptr: double* %arg_a0	<->  call void @callee(ptr %escape_alloca_a0)
; CHECK: Both ModRef:  Ptr: double* %arg_a1	<->  call void @callee(ptr %escape_alloca_a0)
; CHECK: NoModRef:  Ptr: double* %noalias_arg_a0	<->  call void @callee(ptr %escape_alloca_a0)
; CHECK: NoModRef:  Ptr: double* %noalias_arg_a1	<->  call void @callee(ptr %escape_alloca_a0)
; CHECK: Both ModRef:  Ptr: double* %escape_alloca_a0	<->  call void @callee(ptr %escape_alloca_a0)
; CHECK: Both ModRef:  Ptr: double* %escape_alloca_a1	<->  call void @callee(ptr %escape_alloca_a0)
; CHECK: NoModRef:  Ptr: double* %noescape_alloca_a0	<->  call void @callee(ptr %escape_alloca_a0)
; CHECK: NoModRef:  Ptr: double* %noescape_alloca_a1	<->  call void @callee(ptr %escape_alloca_a0)
; CHECK: Both ModRef:  Ptr: double* %normal_ret_a0	<->  call void @callee(ptr %escape_alloca_a0)
; CHECK: Both ModRef:  Ptr: double* %normal_ret_a1	<->  call void @callee(ptr %escape_alloca_a0)
; CHECK: NoModRef:  Ptr: double* %noalias_ret_a0	<->  call void @callee(ptr %escape_alloca_a0)
; CHECK: NoModRef:  Ptr: double* %noalias_ret_a1	<->  call void @callee(ptr %escape_alloca_a0)
; CHECK: Both ModRef:  Ptr: ptr* %indirect_a0	<->  call void @callee(ptr %escape_alloca_a1)
; CHECK: Both ModRef:  Ptr: ptr* %indirect_a1	<->  call void @callee(ptr %escape_alloca_a1)
; CHECK: Both ModRef:  Ptr: double* %loaded_a0	<->  call void @callee(ptr %escape_alloca_a1)
; CHECK: Both ModRef:  Ptr: double* %loaded_a1	<->  call void @callee(ptr %escape_alloca_a1)
; CHECK: Both ModRef:  Ptr: double* %arg_a0	<->  call void @callee(ptr %escape_alloca_a1)
; CHECK: Both ModRef:  Ptr: double* %arg_a1	<->  call void @callee(ptr %escape_alloca_a1)
; CHECK: NoModRef:  Ptr: double* %noalias_arg_a0	<->  call void @callee(ptr %escape_alloca_a1)
; CHECK: NoModRef:  Ptr: double* %noalias_arg_a1	<->  call void @callee(ptr %escape_alloca_a1)
; CHECK: Both ModRef:  Ptr: double* %escape_alloca_a0	<->  call void @callee(ptr %escape_alloca_a1)
; CHECK: Both ModRef:  Ptr: double* %escape_alloca_a1	<->  call void @callee(ptr %escape_alloca_a1)
; CHECK: NoModRef:  Ptr: double* %noescape_alloca_a0	<->  call void @callee(ptr %escape_alloca_a1)
; CHECK: NoModRef:  Ptr: double* %noescape_alloca_a1	<->  call void @callee(ptr %escape_alloca_a1)
; CHECK: Both ModRef:  Ptr: double* %normal_ret_a0	<->  call void @callee(ptr %escape_alloca_a1)
; CHECK: Both ModRef:  Ptr: double* %normal_ret_a1	<->  call void @callee(ptr %escape_alloca_a1)
; CHECK: NoModRef:  Ptr: double* %noalias_ret_a0	<->  call void @callee(ptr %escape_alloca_a1)
; CHECK: NoModRef:  Ptr: double* %noalias_ret_a1	<->  call void @callee(ptr %escape_alloca_a1)
; CHECK: Both ModRef:  Ptr: ptr* %indirect_a0	<->  call void @nocap_callee(ptr %noescape_alloca_a0)
; CHECK: Both ModRef:  Ptr: ptr* %indirect_a1	<->  call void @nocap_callee(ptr %noescape_alloca_a0)
; CHECK: Both ModRef:  Ptr: double* %loaded_a0	<->  call void @nocap_callee(ptr %noescape_alloca_a0)
; CHECK: Both ModRef:  Ptr: double* %loaded_a1	<->  call void @nocap_callee(ptr %noescape_alloca_a0)
; CHECK: Both ModRef:  Ptr: double* %arg_a0	<->  call void @nocap_callee(ptr %noescape_alloca_a0)
; CHECK: Both ModRef:  Ptr: double* %arg_a1	<->  call void @nocap_callee(ptr %noescape_alloca_a0)
; CHECK: NoModRef:  Ptr: double* %noalias_arg_a0	<->  call void @nocap_callee(ptr %noescape_alloca_a0)
; CHECK: NoModRef:  Ptr: double* %noalias_arg_a1	<->  call void @nocap_callee(ptr %noescape_alloca_a0)
; CHECK: Both ModRef:  Ptr: double* %escape_alloca_a0	<->  call void @nocap_callee(ptr %noescape_alloca_a0)
; CHECK: Both ModRef:  Ptr: double* %escape_alloca_a1	<->  call void @nocap_callee(ptr %noescape_alloca_a0)
; CHECK: Both ModRef:  Ptr: double* %noescape_alloca_a0	<->  call void @nocap_callee(ptr %noescape_alloca_a0)
; CHECK: NoModRef:  Ptr: double* %noescape_alloca_a1	<->  call void @nocap_callee(ptr %noescape_alloca_a0)
; CHECK: Both ModRef:  Ptr: double* %normal_ret_a0	<->  call void @nocap_callee(ptr %noescape_alloca_a0)
; CHECK: Both ModRef:  Ptr: double* %normal_ret_a1	<->  call void @nocap_callee(ptr %noescape_alloca_a0)
; CHECK: NoModRef:  Ptr: double* %noalias_ret_a0	<->  call void @nocap_callee(ptr %noescape_alloca_a0)
; CHECK: NoModRef:  Ptr: double* %noalias_ret_a1	<->  call void @nocap_callee(ptr %noescape_alloca_a0)
; CHECK: Both ModRef:  Ptr: ptr* %indirect_a0	<->  call void @nocap_callee(ptr %noescape_alloca_a1)
; CHECK: Both ModRef:  Ptr: ptr* %indirect_a1	<->  call void @nocap_callee(ptr %noescape_alloca_a1)
; CHECK: Both ModRef:  Ptr: double* %loaded_a0	<->  call void @nocap_callee(ptr %noescape_alloca_a1)
; CHECK: Both ModRef:  Ptr: double* %loaded_a1	<->  call void @nocap_callee(ptr %noescape_alloca_a1)
; CHECK: Both ModRef:  Ptr: double* %arg_a0	<->  call void @nocap_callee(ptr %noescape_alloca_a1)
; CHECK: Both ModRef:  Ptr: double* %arg_a1	<->  call void @nocap_callee(ptr %noescape_alloca_a1)
; CHECK: NoModRef:  Ptr: double* %noalias_arg_a0	<->  call void @nocap_callee(ptr %noescape_alloca_a1)
; CHECK: NoModRef:  Ptr: double* %noalias_arg_a1	<->  call void @nocap_callee(ptr %noescape_alloca_a1)
; CHECK: Both ModRef:  Ptr: double* %escape_alloca_a0	<->  call void @nocap_callee(ptr %noescape_alloca_a1)
; CHECK: Both ModRef:  Ptr: double* %escape_alloca_a1	<->  call void @nocap_callee(ptr %noescape_alloca_a1)
; CHECK: NoModRef:  Ptr: double* %noescape_alloca_a0	<->  call void @nocap_callee(ptr %noescape_alloca_a1)
; CHECK: Both ModRef:  Ptr: double* %noescape_alloca_a1	<->  call void @nocap_callee(ptr %noescape_alloca_a1)
; CHECK: Both ModRef:  Ptr: double* %normal_ret_a0	<->  call void @nocap_callee(ptr %noescape_alloca_a1)
; CHECK: Both ModRef:  Ptr: double* %normal_ret_a1	<->  call void @nocap_callee(ptr %noescape_alloca_a1)
; CHECK: NoModRef:  Ptr: double* %noalias_ret_a0	<->  call void @nocap_callee(ptr %noescape_alloca_a1)
; CHECK: NoModRef:  Ptr: double* %noalias_ret_a1	<->  call void @nocap_callee(ptr %noescape_alloca_a1)
; CHECK: Both ModRef:   %normal_ret_a0 = call ptr @normal_returner() <->   %normal_ret_a1 = call ptr @normal_returner()
; CHECK: Both ModRef:   %normal_ret_a0 = call ptr @normal_returner() <->   %noalias_ret_a0 = call ptr @noalias_returner()
; CHECK: Both ModRef:   %normal_ret_a0 = call ptr @normal_returner() <->   %noalias_ret_a1 = call ptr @noalias_returner()
; CHECK: Both ModRef:   %normal_ret_a0 = call ptr @normal_returner() <->   call void @callee(ptr %escape_alloca_a0)
; CHECK: Both ModRef:   %normal_ret_a0 = call ptr @normal_returner() <->   call void @callee(ptr %escape_alloca_a1)
; CHECK: Both ModRef:   %normal_ret_a0 = call ptr @normal_returner() <->   call void @nocap_callee(ptr %noescape_alloca_a0)
; CHECK: Both ModRef:   %normal_ret_a0 = call ptr @normal_returner() <->   call void @nocap_callee(ptr %noescape_alloca_a1)
; CHECK: Both ModRef:   %normal_ret_a1 = call ptr @normal_returner() <->   %normal_ret_a0 = call ptr @normal_returner()
; CHECK: Both ModRef:   %normal_ret_a1 = call ptr @normal_returner() <->   %noalias_ret_a0 = call ptr @noalias_returner()
; CHECK: Both ModRef:   %normal_ret_a1 = call ptr @normal_returner() <->   %noalias_ret_a1 = call ptr @noalias_returner()
; CHECK: Both ModRef:   %normal_ret_a1 = call ptr @normal_returner() <->   call void @callee(ptr %escape_alloca_a0)
; CHECK: Both ModRef:   %normal_ret_a1 = call ptr @normal_returner() <->   call void @callee(ptr %escape_alloca_a1)
; CHECK: Both ModRef:   %normal_ret_a1 = call ptr @normal_returner() <->   call void @nocap_callee(ptr %noescape_alloca_a0)
; CHECK: Both ModRef:   %normal_ret_a1 = call ptr @normal_returner() <->   call void @nocap_callee(ptr %noescape_alloca_a1)
; CHECK: Both ModRef:   %noalias_ret_a0 = call ptr @noalias_returner() <->   %normal_ret_a0 = call ptr @normal_returner()
; CHECK: Both ModRef:   %noalias_ret_a0 = call ptr @noalias_returner() <->   %normal_ret_a1 = call ptr @normal_returner()
; CHECK: Both ModRef:   %noalias_ret_a0 = call ptr @noalias_returner() <->   %noalias_ret_a1 = call ptr @noalias_returner()
; CHECK: Both ModRef:   %noalias_ret_a0 = call ptr @noalias_returner() <->   call void @callee(ptr %escape_alloca_a0)
; CHECK: Both ModRef:   %noalias_ret_a0 = call ptr @noalias_returner() <->   call void @callee(ptr %escape_alloca_a1)
; CHECK: Both ModRef:   %noalias_ret_a0 = call ptr @noalias_returner() <->   call void @nocap_callee(ptr %noescape_alloca_a0)
; CHECK: Both ModRef:   %noalias_ret_a0 = call ptr @noalias_returner() <->   call void @nocap_callee(ptr %noescape_alloca_a1)
; CHECK: Both ModRef:   %noalias_ret_a1 = call ptr @noalias_returner() <->   %normal_ret_a0 = call ptr @normal_returner()
; CHECK: Both ModRef:   %noalias_ret_a1 = call ptr @noalias_returner() <->   %normal_ret_a1 = call ptr @normal_returner()
; CHECK: Both ModRef:   %noalias_ret_a1 = call ptr @noalias_returner() <->   %noalias_ret_a0 = call ptr @noalias_returner()
; CHECK: Both ModRef:   %noalias_ret_a1 = call ptr @noalias_returner() <->   call void @callee(ptr %escape_alloca_a0)
; CHECK: Both ModRef:   %noalias_ret_a1 = call ptr @noalias_returner() <->   call void @callee(ptr %escape_alloca_a1)
; CHECK: Both ModRef:   %noalias_ret_a1 = call ptr @noalias_returner() <->   call void @nocap_callee(ptr %noescape_alloca_a0)
; CHECK: Both ModRef:   %noalias_ret_a1 = call ptr @noalias_returner() <->   call void @nocap_callee(ptr %noescape_alloca_a1)
; CHECK: Both ModRef:   call void @callee(ptr %escape_alloca_a0) <->   %normal_ret_a0 = call ptr @normal_returner()
; CHECK: Both ModRef:   call void @callee(ptr %escape_alloca_a0) <->   %normal_ret_a1 = call ptr @normal_returner()
; CHECK: Both ModRef:   call void @callee(ptr %escape_alloca_a0) <->   %noalias_ret_a0 = call ptr @noalias_returner()
; CHECK: Both ModRef:   call void @callee(ptr %escape_alloca_a0) <->   %noalias_ret_a1 = call ptr @noalias_returner()
; CHECK: Both ModRef:   call void @callee(ptr %escape_alloca_a0) <->   call void @callee(ptr %escape_alloca_a1)
; CHECK: Both ModRef:   call void @callee(ptr %escape_alloca_a0) <->   call void @nocap_callee(ptr %noescape_alloca_a0)
; CHECK: Both ModRef:   call void @callee(ptr %escape_alloca_a0) <->   call void @nocap_callee(ptr %noescape_alloca_a1)
; CHECK: Both ModRef:   call void @callee(ptr %escape_alloca_a1) <->   %normal_ret_a0 = call ptr @normal_returner()
; CHECK: Both ModRef:   call void @callee(ptr %escape_alloca_a1) <->   %normal_ret_a1 = call ptr @normal_returner()
; CHECK: Both ModRef:   call void @callee(ptr %escape_alloca_a1) <->   %noalias_ret_a0 = call ptr @noalias_returner()
; CHECK: Both ModRef:   call void @callee(ptr %escape_alloca_a1) <->   %noalias_ret_a1 = call ptr @noalias_returner()
; CHECK: Both ModRef:   call void @callee(ptr %escape_alloca_a1) <->   call void @callee(ptr %escape_alloca_a0)
; CHECK: Both ModRef:   call void @callee(ptr %escape_alloca_a1) <->   call void @nocap_callee(ptr %noescape_alloca_a0)
; CHECK: Both ModRef:   call void @callee(ptr %escape_alloca_a1) <->   call void @nocap_callee(ptr %noescape_alloca_a1)
; CHECK: Both ModRef:   call void @nocap_callee(ptr %noescape_alloca_a0) <->   %normal_ret_a0 = call ptr @normal_returner()
; CHECK: Both ModRef:   call void @nocap_callee(ptr %noescape_alloca_a0) <->   %normal_ret_a1 = call ptr @normal_returner()
; CHECK: Both ModRef:   call void @nocap_callee(ptr %noescape_alloca_a0) <->   %noalias_ret_a0 = call ptr @noalias_returner()
; CHECK: Both ModRef:   call void @nocap_callee(ptr %noescape_alloca_a0) <->   %noalias_ret_a1 = call ptr @noalias_returner()
; CHECK: Both ModRef:   call void @nocap_callee(ptr %noescape_alloca_a0) <->   call void @callee(ptr %escape_alloca_a0)
; CHECK: Both ModRef:   call void @nocap_callee(ptr %noescape_alloca_a0) <->   call void @callee(ptr %escape_alloca_a1)
; CHECK: Both ModRef:   call void @nocap_callee(ptr %noescape_alloca_a0) <->   call void @nocap_callee(ptr %noescape_alloca_a1)
; CHECK: Both ModRef:   call void @nocap_callee(ptr %noescape_alloca_a1) <->   %normal_ret_a0 = call ptr @normal_returner()
; CHECK: Both ModRef:   call void @nocap_callee(ptr %noescape_alloca_a1) <->   %normal_ret_a1 = call ptr @normal_returner()
; CHECK: Both ModRef:   call void @nocap_callee(ptr %noescape_alloca_a1) <->   %noalias_ret_a0 = call ptr @noalias_returner()
; CHECK: Both ModRef:   call void @nocap_callee(ptr %noescape_alloca_a1) <->   %noalias_ret_a1 = call ptr @noalias_returner()
; CHECK: Both ModRef:   call void @nocap_callee(ptr %noescape_alloca_a1) <->   call void @callee(ptr %escape_alloca_a0)
; CHECK: Both ModRef:   call void @nocap_callee(ptr %noescape_alloca_a1) <->   call void @callee(ptr %escape_alloca_a1)
; CHECK: Both ModRef:   call void @nocap_callee(ptr %noescape_alloca_a1) <->   call void @nocap_callee(ptr %noescape_alloca_a0)
; CHECK: ===== Alias Analysis Evaluator Report =====
; CHECK-NEXT:   120 Total Alias Queries Performed
; CHECK-NEXT:   84 no alias responses (70.0%)
; CHECK-NEXT:   36 may alias responses (30.0%)
; CHECK-NEXT:   0 partial alias responses (0.0%)
; CHECK-NEXT:   0 must alias responses (0.0%)
; CHECK-NEXT:   Alias Analysis Evaluator Pointer Alias Summary: 70%/30%/0%
; CHECK-NEXT:   184 Total ModRef Queries Performed
; CHECK-NEXT:   44 no mod/ref responses (23.9%)
; CHECK-NEXT:   0 mod responses (0.0%)
; CHECK-NEXT:   0 ref responses (0.0%)
; CHECK-NEXT:   140 mod & ref responses (76.0%)
; CHECK-NEXT:   Alias Analysis Evaluator Mod/Ref Summary: 23%/0%/0%/76%

