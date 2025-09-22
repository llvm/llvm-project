; RUN: not llc -O0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -filetype=null < %s 2>&1 | FileCheck --check-prefix=CHECK %s
; CHECK-NOT: error: <unknown>:0:0: local memory (98304) exceeds limit (65536) in function 'k2'

@gA = internal addrspace(3) global [32768 x i8] undef, align 4
@gB = internal addrspace(3) global [32768 x i8] undef, align 4
@gC = internal addrspace(3) global [32768 x i8] undef, align 4

; ---- Helpers ----

define internal void @helperA() inlinehint {
entry:
  %p = getelementptr [32768 x i8], ptr addrspace(3) @gA, i32 0, i32 0
  store i8 1, ptr addrspace(3) %p
  ret void
}

define internal void @helperB() inlinehint {
entry:
  %p = getelementptr [32768 x i8], ptr addrspace(3) @gB, i32 0, i32 0
  store i8 2, ptr addrspace(3) %p
  ret void
}

define internal void @helperC() inlinehint {
entry:
  %p = getelementptr [32768 x i8], ptr addrspace(3) @gC, i32 0, i32 0
  store i8 3, ptr addrspace(3) %p
  ret void
}

; ---------------------------------------------------------------------------
; Dispatch: takes an index and calls the appropriate helper.
; If dispatch is NOT inlined, a backend lowering pass that conservatively
; examines call targets may think all helpers (and thus all globals) are
; potentially referenced by every kernel that calls dispatch.
; ---------------------------------------------------------------------------

define void @dispatch(i32 %idx) inlinehint {
entry:
  %cmp1 = icmp eq i32 %idx, 1
  br i1 %cmp1, label %case1, label %check2

check2:
  %cmp2 = icmp eq i32 %idx, 2
  br i1 %cmp2, label %case2, label %check3

check3:
  %cmp3 = icmp eq i32 %idx, 3
  br i1 %cmp3, label %case3, label %default

case1:
  call void @helperA()
  br label %done

case2:
  call void @helperB()
  br label %done

case3:
  call void @helperC()
  br label %done

default:
  ; fallthrough: call helperA to have a default behaviour
  call void @helperA()
  br label %done

done:
  ret void
}

; ---- Kernels ----

define amdgpu_kernel void @k0() {
entry:
  call void @dispatch(i32 1)
  call void @dispatch(i32 2)
  ret void
}

define amdgpu_kernel void @k1() {
entry:
  call void @dispatch(i32 2)
  call void @dispatch(i32 1)
  ret void
}

define amdgpu_kernel void @k2() {
entry:
  call void @helperC()
  ret void
}
