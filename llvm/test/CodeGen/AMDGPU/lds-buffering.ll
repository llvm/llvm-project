; RUN: opt -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -passes='amdgpu-lds-buffering<max-bytes=64>' -S %s | FileCheck %s

%state = type { <4 x i32>, <4 x i32> }

;===---------------------------------------------------------------------===//
; Positive cases
;===---------------------------------------------------------------------===//

; CHECK: @ldsbuf_test.ldsbuf = internal unnamed_addr addrspace(3) global
; CHECK: @ldsbuf_complex.ldsbuf = internal unnamed_addr addrspace(3) global
; CHECK: @ldsbuf_rocrand_preserved_subsequence.ldsbuf = internal unnamed_addr addrspace(3) global

; Basic positive case: a single global load whose only use is a store back to
; the same pointer. Include an intervening (potentially-aliasing) global store
; to keep the value live and avoid trivially folding away.
; CHECK-LABEL: @ldsbuf_test(
; CHECK: %[[SLOT_BASIC:[^ ]+]] = getelementptr inbounds {{.*}}, ptr addrspace(3) @ldsbuf_test.ldsbuf, i32 0, i32 %
; CHECK: call void @llvm.memcpy.p3.p1.i64(ptr addrspace(3){{.*}}%[[SLOT_BASIC]], ptr addrspace(1){{.*}}%p, i64 16, i1 false)
; CHECK: call void @llvm.memcpy.p1.p3.i64(ptr addrspace(1){{.*}}%p, ptr addrspace(3){{.*}}%[[SLOT_BASIC]], i64 16, i1 false)
define amdgpu_kernel void @ldsbuf_test(ptr addrspace(1) %p, ptr addrspace(1) %q) #0 {
entry:
  %ld = load <4 x i32>, ptr addrspace(1) %p, align 16
  store i32 0, ptr addrspace(1) %q, align 4
  store <4 x i32> %ld, ptr addrspace(1) %p, align 16
  ret void
}

; "Complexity" positive case: keep the loaded value live across control flow
; and additional global memory operations, then store back to the same pointer.
; CHECK-LABEL: @ldsbuf_complex(
; CHECK: %[[SLOT_COMPLEX:[^ ]+]] = getelementptr inbounds {{.*}}, ptr addrspace(3) @ldsbuf_complex.ldsbuf, i32 0, i32 %
; CHECK: call void @llvm.memcpy.p3.p1.i64(ptr addrspace(3){{.*}}%[[SLOT_COMPLEX]], ptr addrspace(1){{.*}}%p, i64 16, i1 false)
; CHECK: call void @llvm.memcpy.p1.p3.i64(ptr addrspace(1){{.*}}%p, ptr addrspace(3){{.*}}%[[SLOT_COMPLEX]], i64 16, i1 false)
define amdgpu_kernel void @ldsbuf_complex(ptr addrspace(1) %p, ptr addrspace(1) %q, i1 %c) #0 {
entry:
  %ld = load <4 x i32>, ptr addrspace(1) %p, align 16
  br i1 %c, label %then, label %else

then:
  store i32 1, ptr addrspace(1) %q, align 4
  br label %merge

else:
  store i32 2, ptr addrspace(1) %q, align 4
  br label %merge

merge:
  store <4 x i32> %ld, ptr addrspace(1) %p, align 16
  ret void
}

; rocRand-inspired positive case: model a per-thread RNG state where part of
; the state (a preserved 16B "subsequence" field) is loaded and stored back
; unchanged, but kept live across a non-trivial loop body.
; CHECK-LABEL: @ldsbuf_rocrand_preserved_subsequence(
; CHECK: %[[TAILPTR:[^ ]+]] = getelementptr inbounds %state, ptr addrspace(1) %state_ptr, i32 0, i32 1
; CHECK: %[[SLOT_TAIL:[^ ]+]] = getelementptr inbounds {{.*}}, ptr addrspace(3) @ldsbuf_rocrand_preserved_subsequence.ldsbuf, i32 0, i32 %
; CHECK: call void @llvm.memcpy.p3.p1.i64(ptr addrspace(3){{.*}}%[[SLOT_TAIL]], ptr addrspace(1){{.*}}%[[TAILPTR]], i64 16, i1 false)
; CHECK: call void @llvm.memcpy.p1.p3.i64(ptr addrspace(1){{.*}}%[[TAILPTR]], ptr addrspace(3){{.*}}%[[SLOT_TAIL]], i64 16, i1 false)
define amdgpu_kernel void @ldsbuf_rocrand_preserved_subsequence(ptr addrspace(1) %state_ptr,
                                                                ptr addrspace(1) %out) #0 {
entry:
  %tailptr = getelementptr inbounds %state, ptr addrspace(1) %state_ptr, i32 0, i32 1
  %tail = load <4 x i32>, ptr addrspace(1) %tailptr, align 16

  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %acc = phi i32 [ 1, %entry ], [ %acc.next, %loop ]

  %shl = shl i32 %acc, 6
  %xor = xor i32 %shl, %acc
  %shr = lshr i32 %xor, 13
  %acc.next = xor i32 %shr, 1234567

  %idx = zext i32 %i to i64
  %outp = getelementptr inbounds float, ptr addrspace(1) %out, i64 %idx
  %f = uitofp i32 %acc.next to float
  store float %f, ptr addrspace(1) %outp, align 4

  %i.next = add nuw i32 %i, 1
  %done = icmp eq i32 %i.next, 8
  br i1 %done, label %exit, label %loop

exit:
  store <4 x i32> %tail, ptr addrspace(1) %tailptr, align 16
  ret void
}

;===---------------------------------------------------------------------===//
; Negative coverage: patterns that must NOT be transformed.
;===---------------------------------------------------------------------===//

; Negative: atomic operations are excluded (must not transform).
; CHECK-LABEL: @ldsbuf_atomic(
; CHECK: load atomic <4 x i32>, ptr addrspace(1) %p unordered, align 16
; CHECK: store atomic <4 x i32> {{.*}}, ptr addrspace(1) %p unordered, align 16
define amdgpu_kernel void @ldsbuf_atomic(ptr addrspace(1) %p) #0 {
entry:
  %ld = load atomic <4 x i32>, ptr addrspace(1) %p unordered, align 16
  store atomic <4 x i32> %ld, ptr addrspace(1) %p unordered, align 16
  ret void
}

; Negative: volatile operations are excluded (must not transform).
; CHECK-LABEL: @ldsbuf_volatile(
; CHECK: load volatile <4 x i32>, ptr addrspace(1) %p, align 16
; CHECK: store volatile <4 x i32> {{.*}}, ptr addrspace(1) %p, align 16
define amdgpu_kernel void @ldsbuf_volatile(ptr addrspace(1) %p) #0 {
entry:
  %ld = load volatile <4 x i32>, ptr addrspace(1) %p, align 16
  store volatile <4 x i32> %ld, ptr addrspace(1) %p, align 16
  ret void
}

; Negative: alignment requirement (min 16) must be met.
; CHECK-LABEL: @ldsbuf_misaligned(
; CHECK: load <4 x i32>, ptr addrspace(1) %p, align 8
; CHECK: store <4 x i32> {{.*}}, ptr addrspace(1) %p, align 8
define amdgpu_kernel void @ldsbuf_misaligned(ptr addrspace(1) %p) #0 {
entry:
  %ld = load <4 x i32>, ptr addrspace(1) %p, align 8
  store <4 x i32> %ld, ptr addrspace(1) %p, align 8
  ret void
}

; Negative: size limit (max-bytes=64) must be respected.
; CHECK-LABEL: @ldsbuf_too_large(
; CHECK: load <20 x i32>, ptr addrspace(1) %p, align 16
; CHECK: store <20 x i32> {{.*}}, ptr addrspace(1) %p, align 16
define amdgpu_kernel void @ldsbuf_too_large(ptr addrspace(1) %p) #0 {
entry:
  %ld = load <20 x i32>, ptr addrspace(1) %p, align 16
  store <20 x i32> %ld, ptr addrspace(1) %p, align 16
  ret void
}

; Negative: non-global pointers are excluded (only addrspace(1) is supported).
; CHECK-LABEL: @ldsbuf_non_global_ptr(
; CHECK: load <4 x i32>, ptr %p, align 16
; CHECK: store <4 x i32> {{.*}}, ptr %p, align 16
define amdgpu_kernel void @ldsbuf_non_global_ptr(ptr %p) #0 {
entry:
  %ld = load <4 x i32>, ptr %p, align 16
  store <4 x i32> %ld, ptr %p, align 16
  ret void
}

; Negative: the load must have exactly one use (the final store).
; CHECK-LABEL: @ldsbuf_multiple_uses(
; CHECK: %ld = load <4 x i32>, ptr addrspace(1) %p, align 16
; CHECK: extractelement <4 x i32> %ld, i32 0
; CHECK: store <4 x i32> %ld, ptr addrspace(1) %p, align 16
define amdgpu_kernel void @ldsbuf_multiple_uses(ptr addrspace(1) %p, ptr addrspace(1) %q) #0 {
entry:
  %ld = load <4 x i32>, ptr addrspace(1) %p, align 16
  %e = extractelement <4 x i32> %ld, i32 0
  store i32 %e, ptr addrspace(1) %q, align 4
  store <4 x i32> %ld, ptr addrspace(1) %p, align 16
  ret void
}

; Negative: budget rejection. With large pre-existing LDS usage and large
; workgroup size, per-thread slots should be rejected by the LDS budget check.
@ldsbuf_budget_reject.big = internal addrspace(3) global [8192 x i32] zeroinitializer, align 16

; CHECK-LABEL: @ldsbuf_budget_reject(
; CHECK: load <16 x i32>, ptr addrspace(1) %p, align 16
; CHECK: store <16 x i32> {{.*}}, ptr addrspace(1) %p, align 16
define amdgpu_kernel void @ldsbuf_budget_reject(ptr addrspace(1) %p) #1 {
entry:
  %x = load i32, ptr addrspace(3) getelementptr inbounds ([8192 x i32], ptr addrspace(3) @ldsbuf_budget_reject.big, i32 0, i32 0), align 16
  call void @llvm.donothing()
  %ld = load <16 x i32>, ptr addrspace(1) %p, align 16
  store <16 x i32> %ld, ptr addrspace(1) %p, align 16
  ret void
}

declare void @llvm.donothing()

attributes #0 = { "amdgpu-flat-work-group-size"="1,256" "uniform-work-group-size"="true" }
attributes #1 = { "amdgpu-flat-work-group-size"="1024,1024" "uniform-work-group-size"="true" }
