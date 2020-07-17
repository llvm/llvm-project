; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx803 -mattr=+trap-handler < %s | FileCheck %s --check-prefixes=GCN,TRAP-HANDLER-ENABLE
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx803 -mattr=-trap-handler < %s | FileCheck %s --check-prefixes=GCN,TRAP-HANDLER-DISABLE

; GCN-LABEL: {{^}}amdhsa_trap_num_sgprs
; TRAP-HANDLER-ENABLE:  NumSgprs: 61
; TRAP-HANDLER-DISABLE: NumSgprs: 79
define amdgpu_kernel void @amdhsa_trap_num_sgprs(
    i32 addrspace(1)* %out0, i32 %in0,
    i32 addrspace(1)* %out1, i32 %in1,
    i32 addrspace(1)* %out2, i32 %in2,
    i32 addrspace(1)* %out3, i32 %in3,
    i32 addrspace(1)* %out4, i32 %in4,
    i32 addrspace(1)* %out5, i32 %in5,
    i32 addrspace(1)* %out6, i32 %in6,
    i32 addrspace(1)* %out7, i32 %in7,
    i32 addrspace(1)* %out8, i32 %in8,
    i32 addrspace(1)* %out9, i32 %in9,
    i32 addrspace(1)* %out10, i32 %in10,
    i32 addrspace(1)* %out11, i32 %in11,
    i32 addrspace(1)* %out12, i32 %in12,
    i32 addrspace(1)* %out13, i32 %in13,
    i32 addrspace(1)* %out14, i32 %in14,
    i32 addrspace(1)* %out15, i32 %in15,
    i32 addrspace(1)* %out16, i32 %in16,
    i32 addrspace(1)* %out17, i32 %in17,
    i32 addrspace(1)* %out18, i32 %in18,
    i32 addrspace(1)* %out19, i32 %in19,
    i32 addrspace(1)* %out20, i32 %in20,
    i32 addrspace(1)* %out21, i32 %in21,
    i32 addrspace(1)* %out22, i32 %in22,
    i32 addrspace(1)* %out23, i32 %in23,
    i32 addrspace(1)* %out24, i32 %in24,
    i32 addrspace(1)* %out25, i32 %in25,
    i32 addrspace(1)* %out26, i32 %in26,
    i32 addrspace(1)* %out27, i32 %in27,
    i32 addrspace(1)* %out28, i32 %in28,
    i32 addrspace(1)* %out29, i32 %in29) {
entry:
  store i32 %in0, i32 addrspace(1)* %out0
  store i32 %in1, i32 addrspace(1)* %out1
  store i32 %in2, i32 addrspace(1)* %out2
  store i32 %in3, i32 addrspace(1)* %out3
  store i32 %in4, i32 addrspace(1)* %out4
  store i32 %in5, i32 addrspace(1)* %out5
  store i32 %in6, i32 addrspace(1)* %out6
  store i32 %in7, i32 addrspace(1)* %out7
  store i32 %in8, i32 addrspace(1)* %out8
  store i32 %in9, i32 addrspace(1)* %out9
  store i32 %in10, i32 addrspace(1)* %out10
  store i32 %in11, i32 addrspace(1)* %out11
  store i32 %in12, i32 addrspace(1)* %out12
  store i32 %in13, i32 addrspace(1)* %out13
  store i32 %in14, i32 addrspace(1)* %out14
  store i32 %in15, i32 addrspace(1)* %out15
  store i32 %in16, i32 addrspace(1)* %out16
  store i32 %in17, i32 addrspace(1)* %out17
  store i32 %in18, i32 addrspace(1)* %out18
  store i32 %in19, i32 addrspace(1)* %out19
  store i32 %in20, i32 addrspace(1)* %out20
  store i32 %in21, i32 addrspace(1)* %out21
  store i32 %in22, i32 addrspace(1)* %out22
  store i32 %in23, i32 addrspace(1)* %out23
  store i32 %in24, i32 addrspace(1)* %out24
  store i32 %in25, i32 addrspace(1)* %out25
  store i32 %in26, i32 addrspace(1)* %out26
  store i32 %in27, i32 addrspace(1)* %out27
  store i32 %in28, i32 addrspace(1)* %out28
  store i32 %in29, i32 addrspace(1)* %out29
  ret void
}
