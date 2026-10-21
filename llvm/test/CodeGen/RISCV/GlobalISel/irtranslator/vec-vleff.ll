; RUN: not --crash llc -global-isel -mtriple=riscv64 -mattr=+v -filetype=null %s 2>&1 | FileCheck %s

; Intrinsics returning structs and extractvalue of scalable vector are not
; supported yet.
define <vscale x 1 x i64> @intrinsic_vleff_v_nxv1i64_nxv1i64(ptr %0, i64 %1, ptr %2) nounwind {
entry:
  %a = call { <vscale x 1 x i64>, i64 } @llvm.riscv.vleff.nxv1i64(<vscale x 1 x i64> poison, ptr %0, i64 %1)
  %b = extractvalue { <vscale x 1 x i64>, i64 } %a, 0
  %c = extractvalue { <vscale x 1 x i64>, i64 } %a, 1
  store i64 %c, ptr %2
  ret <vscale x 1 x i64> %b
}

; CHECK: LLVM ERROR: unable to translate instruction: call llvm.riscv.vleff.nxv1i64.i64.p0 (in function: intrinsic_vleff_v_nxv1i64_nxv1i64)
