; RUN: opt %loadPolly -polly-codegen-ppcg -polly-invariant-load-hoisting \
; RUN: -S < %s | \
; RUN: FileCheck -check-prefix=HOST-IR %s

; RUN: opt %loadPolly -disable-output -polly-acc-dump-kernel-ir \
; RUN: -polly-codegen-ppcg -polly-scops \
; RUN: -polly-invariant-load-hoisting < %s | FileCheck -check-prefix=KERNEL-IR %s

; REQUIRES: pollyacc

; Verify that invariant loads used in a kernel statement are correctly forwarded
; as subtree value to the GPU kernel.

; HOST-IR: store float %polly.access.p.load, float* %invariant.preload.s2a, align 4

; KERNEL-IR:  define ptx_kernel void @FUNC_foo_SCOP_0_KERNEL_2({{.*}}i8 addrspace(1)* %MemRef_indvar2f__phi{{.*}})
; KERNEL-IR:   %indvar2f.phiops.reload = load float, float* %indvar2f.phiops, align 4
; KERNEL-IR:   store float %indvar2f.phiops.reload, float addrspace(1)* %polly.access.MemRef_A, align 4

; FIXME: store float %indvar2f.phiops.reload, float* %indvar2f.phiops, align 4
; For some reason the above instruction is emitted that stores back to the addess it was just loaded from.

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
  br i1 %cmp, label %loop, label %anotherloop

anotherloop:
  %indvar2 = phi i64 [0, %loop], [%indvar2.next, %anotherloop]
  %indvar2f = phi float [%invariant, %loop], [%indvar2f, %anotherloop]
  %indvar2.next = add i64 %indvar2, 1
  store float %indvar2f, float* %A
  %cmp2 = icmp sle i64 %indvar2, 1024
  br i1 %cmp2, label %anotherloop, label %end

end:
  ret void

}
