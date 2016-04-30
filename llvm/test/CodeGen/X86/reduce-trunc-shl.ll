; RUN: llc -mtriple=x86_64-unknown-linux-gnu -mattr=+avx2 < %s | FileCheck %s

; CHECK-LABEL: {{^}}trunc_shl_7_v4i32_v4i64:
; CHECK: vpshufd $136, (%rsi), %ymm0
; CHECK: vpermq	$236, %ymm0, %ymm0
; CHECK: vpslld $7, %xmm0, %xmm0
; CHECK: vmovdqa %xmm0, (%rdi)
define void @trunc_shl_7_v4i32_v4i64(<4 x i32> addrspace(1)* %out, <4 x i64> addrspace(1)* %in) {
  %val = load <4 x i64>, <4 x i64> addrspace(1)* %in
  %shl = shl <4 x i64> %val, <i64 7, i64 7, i64 7, i64 7>
  %trunc = trunc <4 x i64> %shl to <4 x i32>
  store <4 x i32> %trunc, <4 x i32> addrspace(1)* %out
  ret void
}
