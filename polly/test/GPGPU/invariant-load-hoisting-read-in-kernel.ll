; RUN: opt %loadPolly -disable-output -polly-acc-dump-kernel-ir \
; RUN: -polly-codegen-ppcg -polly-scops \
; RUN: -polly-invariant-load-hoisting < %s | FileCheck %s

; REQUIRES: pollyacc

; Verify that invariant loads used in a kernel statement are correctly forwarded
; as subtree value to the GPU kernel.

; CHECK:  define ptx_kernel void @FUNC_foo_SCOP_0_KERNEL_0({{.*}} float %polly.access.p.load)
; CHECK:   store float %polly.access.p.load, float* %indvar2f.phiops

define void @foo(float* %A, float* %p) {
entry:
  br label %loop

loop:
  %indvar = phi i64 [0, %entry], [%indvar.next, %loop]
  %indvar.next = add i64 %indvar, 1
  %invariant = load float, float* %p
  %ptr = getelementptr float, float* %A, i64 %indvar
  store float 42.0, float* %ptr
  %cmp = icmp sle i64 %indvar, 1024
  br i1 %cmp, label %loop, label %loop2

loop2:
  %indvar2 = phi i64 [0, %loop], [%indvar2.next, %loop2]
  %indvar2f = phi float [%invariant, %loop], [%indvar2f, %loop2]
  %indvar2.next = add i64 %indvar2, 1
  store float %indvar2f, float* %A
  %cmp2 = icmp sle i64 %indvar2, 1024
  br i1 %cmp2, label %loop2, label %end

end:
  ret void

}
