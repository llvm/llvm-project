; RUN: llc -march=hexagon -mcpu=hexagonv73 -mattr=+hmx < %s | FileCheck %s

; Test that the compiler generates code for the HMX intrinsics.

; CHECK-LABEL: M8_cvt_rs_hf:
; CHECK: cvt.hf = acc(r0)
define void @M8_cvt_rs_hf(i32 %a) #0 {
entry:
  call void @llvm.hexagon.M8.cvt.rs.hf(i32 %a)
  ret void
}
declare void @llvm.hexagon.M8.cvt.rs.hf(i32)

; CHECK-LABEL: M8_mxmem:
; CHECK: mxmem(r0,r1) = cvt
define void @M8_mxmem(ptr %a, i32 %b) #0 {
entry:
  call void @llvm.hexagon.M8.mxmem(ptr %a, i32 %b)
  ret void
}
declare void @llvm.hexagon.M8.mxmem(ptr, i32)

attributes #0 = { nounwind }
