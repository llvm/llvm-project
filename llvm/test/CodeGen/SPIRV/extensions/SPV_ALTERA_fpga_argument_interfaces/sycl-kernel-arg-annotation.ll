; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_ALTERA_fpga_argument_interfaces %s -o - | FileCheck %s
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpCapability FPGAArgumentInterfacesALTERA
; CHECK: OpExtension "SPV_ALTERA_fpga_argument_interfaces"
; CHECK: OpName %[[ID:[0-9]+]] "_arg_p"
; CHECK: OpDecorate %[[ID]] Alignment 4
; CHECK: OpDecorate %[[ID]] MMHostInterfaceAddressWidthALTERA 32
; CHECK: OpDecorate %[[ID]] ConduitKernelArgumentALTERA
; CHECK: OpDecorate %[[ID]] MMHostInterfaceDataWidthALTERA 64
; CHECK: OpDecorate %[[ID]] MMHostInterfaceLatencyALTERA 1
; CHECK: OpDecorate %[[ID]] MMHostInterfaceMaxBurstALTERA 3
; CHECK: OpDecorate %[[ID]] MMHostInterfaceReadWriteModeALTERA 2
; CHECK: OpDecorate %[[ID]] RegisterMapKernelArgumentALTERA
; CHECK: OpDecorate %[[ID]] StableKernelArgumentALTERA
; CHECK: OpDecorate %[[ID]] Restrict
; CHECK: OpDecorate %[[ID]] MMHostInterfaceWaitRequestALTERA 5

$_ZTS4MyIP = comdat any

define weak_odr dso_local spir_kernel void @_ZTS4MyIP(ptr addrspace(4) noundef %_arg_p) #0 comdat !spirv.ParameterDecorations !1588
; CHECK-LLVM-DAG:  !spirv.ParameterDecorations ![[PARMDECOR:[0-9]+]]
{
entry:
		ret void
}

!1587 = !{i32 -1}
!1588 = !{!1589}
!1589 = !{!1590, !1591, !1593, !1594, !1595, !1596, !1597, !1598, !1599, !1600, !1601}
!1590 = !{i32 44, i32 4}
!1591 = !{i32 6177, i32 32}
!1593 = !{i32 6175, i32 1}
!1594 = !{i32 6178, i32 64}
!1595 = !{i32 6179, i32 1}
!1596 = !{i32 6181, i32 3}
!1597 = !{i32 6180, i32 2}
!1598 = !{i32 6176, i32 1}
!1599 = !{i32 6183, i32 1}
!1600 = !{i32 19, i32 1}
!1601 = !{i32 6182, i32 5}
