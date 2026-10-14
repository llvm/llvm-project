; RUN: opt %s -passes='print<uniformity>' -disable-output 2>&1 | FileCheck %s
;
; Exercises NVPTX TTI: generic-space loads are not hard-coded as divergence
; sources; uniformity must follow the (cast) pointers feeding the load.
;
; Uniform case: one branch uses a pointer in entry/param space (101), the other
; loads a pointer from a global (addrspace 1). Each is addrspacecast to
; generic (0); the phi merges generic pointers only, then a single generic load.

target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Global slot in global AS holding a pointer to int in global AS.
@g1 = external addrspace(1) global i32 addrspace(1)*

define ptx_kernel i32 @uniform_generic_ptr_phi_load(i32 addrspace(101)* %p, i32 %n) {
; CHECK-LABEL: for function 'uniform_generic_ptr_phi_load'
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %then, label %else
; CHECK-NOT: DIVERGENT: %cmp =
; CHECK-NOT: DIVERGENT: br i1 %cmp,
then:
  %gen_param = addrspacecast i32 addrspace(101)* %p to i32 addrspace(0)*
  br label %merge
else:
  %pg = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(1)* @g1, align 8
  %gen_global = addrspacecast i32 addrspace(1)* %pg to i32 addrspace(0)*
  br label %merge
merge:
  %ptr = phi i32 addrspace(0)* [ %gen_param, %then ], [ %gen_global, %else ]
  %v = load i32, i32 addrspace(0)* %ptr, align 4
; CHECK-NOT: DIVERGENT: %ptr =
; CHECK-NOT: DIVERGENT: %v =
  ret i32 %v
}

; Divergent generic pointer: address depends on tid, so the generic load is divergent.
define ptx_kernel i32 @divergent_generic_load(i32 addrspace(1)* %base) {
; CHECK-LABEL: for function 'divergent_generic_load'
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %p = getelementptr inbounds i32, i32 addrspace(1)* %base, i32 %tid
  %g = addrspacecast i32 addrspace(1)* %p to i32 addrspace(0)*
  %v = load i32, i32 addrspace(0)* %g, align 4
; CHECK: DIVERGENT: %v =
  ret i32 %v
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
