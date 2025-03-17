; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_fpga_cluster_attributes %s -o - | FileCheck %s 

; CHECK-DAG: OpCapability FPGAClusterAttributesINTEL 
; CHECK-DAG: OpCapability FPGAClusterAttributesV2INTEL 
; CHECK-DAG: OpExtension "SPV_INTEL_fpga_cluster_attributes" 
; CHECK-DAG: OpDecorate %[[#STALLENABLE_DEC:]] StallEnableINTEL 
; CHECK-DAG: OpDecorate %[[#STALLFREE_DEC:]] StallFreeINTEL 
; CHECK: %[[#STALLENABLE_DEC]] = OpFunction %[[#]] None %[[#]] 
; CHECK: %[[#STALLFREE_DEC]] = OpFunction %[[#]] None %[[#]] 

define spir_func void @test_fpga_stallenable_attr() !stall_enable !0 {
   entry:
    ret void
} 

define spir_func void @test_fpga_stallfree_attr() !stall_free !1 {
   entry:
    ret void
} 

!0 = !{ i32 1 } 
!1 = !{ i32 1 } 