; RUN: opt %s -disable-output -S -passes="print<stack-safety-local>" 2>&1 | FileCheck %s

; Datalayout from AMDGPU, p5 is 32 bits and p is 64.
; We used to call SCEV getTruncateOrZeroExtend on %x.ascast/%x which caused an assertion failure.

; CHECK:      @a dso_preemptable
; CHECK-NEXT:   args uses:
; CHECK-NEXT:     x[]: full-set
; CHECK-NEXT:   allocas uses:

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"

define void @a(ptr addrspace(5) %x) {
entry:
  %x.ascast = addrspacecast ptr addrspace(5) %x to ptr
  %tmp = load i64, ptr %x.ascast
  store i64 %tmp, ptr %x.ascast, align 8
  ret void
}
