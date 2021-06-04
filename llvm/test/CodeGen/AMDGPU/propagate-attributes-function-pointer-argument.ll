; This is a regression test for a bug in the AMDGPU Propagate Attributes pass
; where a call instruction's callee could be replaced with a function pointer
; passed to the original call instruction as an argument.
;
; Example:
; `call void @f(void ()* @g)`
; could become
; `call void @g(void ()* @g.1)`
; which is invalid IR.

; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -amdgpu-propagate-attributes-late %s | FileCheck %s

; CHECK-LABEL: define amdgpu_kernel void @thiswasabug() #0
; CHECK-NOT: call void @g(void ()* @g.1)
; CHECK-DAG: call void @f(void ()* @g.1)
; CHECK-DAG: call void @g()
define amdgpu_kernel void @thiswasabug() #0 {
    ; no replacement, but @g should be renamed to @g.1
    call void @f(void ()* @g)

    ; this should call the clone, which takes the name @g
    call void @g()
    ret void
}

define private void @f(void ()* nocapture %0) #0 {
    ret void
}

; In order to expose this bug, it is necessary that `g` have one of the
; propagated attributes, so that a clone and substitution would take place if g
; were actually the function being called.
; CHECK-DAG: define private void @g.1() #0
; CHECK-DAG: define internal void @g() #1
define private void @g() #1 {
    ret void
}

attributes #0 = { noinline }
attributes #1 = { noinline "amdgpu-waves-per-eu"="1,10" }

; CHECK: attributes #0 = { noinline }
; CHECK-NEXT: attributes #1 = { noinline "target-features"="+enable-ds128,+enable-prt-strict-null,+flat-address-space,+flat-for-global,+load-store-opt,+promote-alloca,+trap-handler,+unaligned-access-mode,-wavefrontsize16,-wavefrontsize32,+wavefrontsize64" }
