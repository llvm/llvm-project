; RUN: llc -mtriple=i686-pc-windows-gnu -mattr=+avx512f,+avx512vl,+avx512bw,+avx512fp16 < %s | FileCheck %s
; RUN: llc -mtriple=x86_64-pc-windows-msvc -mattr=+avx512f,+avx512vl,+avx512bw,+avx512fp16 < %s | FileCheck %s --check-prefix=X64

; Verify that half (f16), bfloat (bf16), and fp128 arguments and return values
; are assigned to XMM registers when using the vectorcall calling convention.
; By returning the second argument, we force the compiler to emit a register move,
; explicitly demonstrating the use of XMM registers.

define dso_local x86_vectorcallcc half @vectorcall_f16(half %A, half %B) {
; CHECK-LABEL: vectorcall_f16@@8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmovaps %xmm1, %xmm0
; CHECK-NEXT:    retl
;
; X64-LABEL: vectorcall_f16@@16:
; X64:       # %bb.0:
; X64-NEXT:    vmovaps %xmm1, %xmm0
; X64-NEXT:    retq
entry:
  ret half %B
}

define dso_local x86_vectorcallcc bfloat @vectorcall_bf16(bfloat %A, bfloat %B) {
; CHECK-LABEL: vectorcall_bf16@@8:
; CHECK:       # %bb.0:
; CHECK-NEXT:    vmovaps %xmm1, %xmm0
; CHECK-NEXT:    retl
;
; X64-LABEL: vectorcall_bf16@@16:
; X64:       # %bb.0:
; X64-NEXT:    vmovaps %xmm1, %xmm0
; X64-NEXT:    retq
entry:
  ret bfloat %B
}

define dso_local x86_vectorcallcc fp128 @vectorcall_f128(fp128 %A, fp128 %B) {
; CHECK-LABEL: vectorcall_f128@@32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    movl 4(%esp), %eax
; CHECK-NEXT:    vmovups 24(%esp), %xmm0
; CHECK-NEXT:    vmovaps %xmm0, (%eax)
; CHECK-NEXT:    retl $36
;
; X64-LABEL: vectorcall_f128@@32:
; X64:       # %bb.0:
; X64-NEXT:    vmovaps %xmm1, %xmm0
; X64-NEXT:    retq
entry:
  ret fp128 %B
}
