; RUN: opt -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -passes=amdgpu-lds-buffering -S %s | FileCheck %s

; Check LDS global creation
; CHECK: @ldsbuf_test.ldsbuf = internal unnamed_addr addrspace(3) global

; CHECK-LABEL: @ldsbuf_test(
; Ensure the original direct global load is gone before the first memcpy-in
; CHECK-NOT: load <4 x i32>, ptr addrspace(1) %p
; LDS slot computation GEP on the function-scoped LDS global
; CHECK: %[[SLOT:[^ ]+]] = getelementptr inbounds {{.*}}, ptr addrspace(3) @ldsbuf_test.ldsbuf, i32 0, i32 %
; memcpy global -> LDS at the load site
; CHECK: call void @llvm.memcpy.p3.p1.i64(ptr addrspace(3){{.*}}%[[SLOT]], ptr addrspace(1){{.*}}%p, i64 16, i1 false)
; Ensure the original direct global store is gone between the two memcpys
; CHECK-NOT: store <4 x i32>
; memcpy LDS -> global at the original store site
; CHECK: call void @llvm.memcpy.p1.p3.i64(ptr addrspace(1){{.*}}%p, ptr addrspace(3){{.*}}%[[SLOT]], i64 16, i1 false)

; Minimal kernel with a single load from AS(1) used once by a
; store back to the same pointer. Should be buffered via LDS.

define amdgpu_kernel void @ldsbuf_test(ptr addrspace(1) %p) #0 {
entry:
  %ld = load <4 x i32>, ptr addrspace(1) %p, align 16
  store <4 x i32> %ld, ptr addrspace(1) %p, align 16
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,256" "target-cpu"="gfx950" "uniform-work-group-size"="true" }
