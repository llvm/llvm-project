; Half denormals are flushed where the denormal mode for half asks for it
; (PTX .ftz on f16: denormal_fpenv(preservesign) for every type): the GPU
; keeps them, so half arithmetic reads and writes through
; flush(x) = |x| < 2^-14 ? copysign(0, x) : x. A float-only mode leaves half
; alone.
; RUN: llc -mtriple=air %s -o - | FileCheck %s

declare <2 x half> @llvm.minimumnum.v2f16(<2 x half>, <2 x half>)

; CHECK-LABEL: define void @ftz
; CHECK: [[A:%.*]] = load <2 x half>
; CHECK: [[B:%.*]] = load <2 x half>
; CHECK-DAG: [[AA:%.*]] = call <2 x half> @air.fabs.v2f16(<2 x half> [[A]])
; CHECK-DAG: [[AS:%.*]] = fcmp olt <2 x half> [[AA]], splat (half 6.103520e-05)
; CHECK-DAG: [[AZ:%.*]] = call <2 x half> @llvm.copysign.v2f16(<2 x half> zeroinitializer, <2 x half> [[A]])
; CHECK-DAG: [[AF:%.*]] = select <2 x i1> [[AS]], <2 x half> [[AZ]], <2 x half> [[A]]
; CHECK: [[M:%.*]] = call <2 x half> @air.fmin.v2f16(<2 x half> [[AF]],
; CHECK: [[MA:%.*]] = call <2 x half> @air.fabs.v2f16(<2 x half> [[M]])
; CHECK: fcmp olt <2 x half> [[MA]], splat (half 6.103520e-05)
; CHECK-LABEL: define void @fadd_ftz
; CHECK: fadd half
; CHECK-NEXT: call half @air.fabs.f16
define void @ftz(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %o) #0 {
  %x = load <2 x half>, ptr addrspace(1) %a
  %y = load <2 x half>, ptr addrspace(1) %b
  %m = call <2 x half> @llvm.minimumnum.v2f16(<2 x half> %x, <2 x half> %y)
  store <2 x half> %m, ptr addrspace(1) %o
  ret void
}

define void @fadd_ftz(ptr addrspace(1) %a, ptr addrspace(1) %o) #0 {
  %x = load half, ptr addrspace(1) %a
  %s = fadd half %x, %x
  store half %s, ptr addrspace(1) %o
  ret void
}

; CHECK-LABEL: define void @float_only
; CHECK-NOT: fabs
; CHECK: ret void
define void @float_only(ptr addrspace(1) %a, ptr addrspace(1) %o) #1 {
  %x = load half, ptr addrspace(1) %a
  %s = fadd half %x, %x
  store half %s, ptr addrspace(1) %o
  ret void
}

attributes #0 = { denormal_fpenv(preservesign) "air-kernel" }
attributes #1 = { denormal_fpenv(float: preservesign) "air-kernel" }
