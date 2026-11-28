; RUN: llc -march=hexagon -mcpu=hexagonv68 -mattr=+hmx < %s | FileCheck %s

; Test that the compiler generates code for the HMX intrinsics.

; CHECK-LABEL: M8_mxclracc_hf:
; CHECK: mxclracc.hf
define void @M8_mxclracc_hf() #0 {
entry:
  call void @llvm.hexagon.M8.mxclracc.hf()
  ret void
}
declare void @llvm.hexagon.M8.mxclracc.hf()

; CHECK-LABEL: M8_mxmem2_bias:
; CHECK: bias = mxmem2(r0)
define void @M8_mxmem2_bias(ptr %a) #0 {
entry:
  call void @llvm.hexagon.M8.mxmem2.bias(ptr %a)
  ret void
}
declare void @llvm.hexagon.M8.mxmem2.bias(ptr)

; CHECK-LABEL: M8_mxmem2_st_bias:
; CHECK: mxmem2(r0) = bias
define void @M8_mxmem2_st_bias(ptr %a) #0 {
entry:
  call void @llvm.hexagon.M8.mxmem2.st.bias(ptr %a)
  ret void
}
declare void @llvm.hexagon.M8.mxmem2.st.bias(ptr)

; CHECK-LABEL: M8_mxmem_blk_sm_act_hf_M8_mxmem_wei_hf:
; CHECK: activation.hf = mxmem(r0,r1)
; CHECK-NEXT: weight.hf = mxmem(r2,r3)
define void @M8_mxmem_blk_sm_act_hf_M8_mxmem_wei_hf(i32 %a, i32 %b, i32 %c, i32 %d) #0 {
entry:
  call void @llvm.hexagon.M8.mxmem.blk.sm.act.hf(i32 %a, i32 %b)
  call void @llvm.hexagon.M8.mxmem.wei.hf(i32 %c, i32 %d)
  ret void
}

; CHECK-LABEL: M8_mxmem_sm_act_hf_M8_mxmem_wei_hf:
; CHECK: activation.hf = mxmem(r0,r1):deep
; CHECK-NEXT: weight.hf = mxmem(r2,r3)
define void @M8_mxmem_sm_act_hf_M8_mxmem_wei_hf(i32 %a, i32 %b, i32 %c, i32 %d) #0 {
entry:
  call void @llvm.hexagon.M8.mxmem.sm.act.hf(i32 %a, i32 %b)
  call void @llvm.hexagon.M8.mxmem.wei.hf(i32 %c, i32 %d)
  ret void
}

; CHECK-LABEL: M8_mxmems_blk_sm_act_hf_M8_mxmemdp_wei_hf:
; CHECK: activation.hf = mxmem(r0,r1):single
; CHECK-NEXT: weight.hf = mxmem(r2,r3):deep
define void @M8_mxmems_blk_sm_act_hf_M8_mxmemdp_wei_hf(i32 %a, i32 %b, i32 %c, i32 %d) #0 {
entry:
  call void @llvm.hexagon.M8.mxmems.blk.sm.act.hf(i32 %a, i32 %b)
  call void @llvm.hexagon.M8.mxmemdp.wei.hf(i32 %c, i32 %d)
  ret void
}

declare void @llvm.hexagon.M8.mxmem.blk.sm.act.hf(i32, i32)
declare void @llvm.hexagon.M8.mxmem.sm.act.hf(i32, i32)
declare void @llvm.hexagon.M8.mxmems.blk.sm.act.hf(i32, i32)
declare void @llvm.hexagon.M8.mxmem.wei.hf(i32, i32)
declare void @llvm.hexagon.M8.mxmemdp.wei.hf(i32, i32)

attributes #0 = { nounwind }
