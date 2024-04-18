; RUN: llvm-split -o %t %s -j 3 -mtarget amdgcn-amd-amdhsa
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK0 %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=CHECK1 %s
; RUN: llvm-dis -o - %t2 | FileCheck --check-prefix=CHECK2 %s

; We have 4 kernels:
;   - Each kernel has an internal helper
;   - @A and @B's helpers does an indirect call.
;
; We default to putting A/B in P0, alongside a copy
; of all helpers. The other kernels can still go into
; separate partitions.

; CHECK0-NOT: define
; CHECK0: define internal void @HelperA
; CHECK0: define internal void @HelperB
; CHECK0: define internal void @HelperC
; CHECK0: define internal void @HelperD
; CHECK0: define amdgpu_kernel void @A
; CHECK0: define amdgpu_kernel void @B
; CHECK0-NOT: define

; CHECK1-NOT: define
; CHECK1: define internal void @HelperD
; CHECK1: define amdgpu_kernel void @D
; CHECK1-NOT: define

; CHECK2-NOT: define
; CHECK2: define internal void @HelperC
; CHECK2: define amdgpu_kernel void @C
; CHECK2-NOT: define

define internal void @HelperA(ptr %call) {
    call void %call()
    ret void
}

define internal void @HelperB(ptr %call) {
    call void %call()
    ret void
}

define internal void @HelperC() {
    ret void
}

define internal void @HelperD() {
    ret void
}

define amdgpu_kernel void @A(ptr %call) {
    call void @HelperA(ptr %call)
    ret void
}

define amdgpu_kernel void @B(ptr %call) {
    call void @HelperB(ptr %call)
    ret void
}

define amdgpu_kernel void @C() {
    call void @HelperC()
    ret void
}

define amdgpu_kernel void @D() {
    call void @HelperD()
    ret void
}
