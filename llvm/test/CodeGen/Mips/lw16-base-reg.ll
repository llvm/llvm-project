; RUN: llc %s -march=mips -mcpu=mips32r3 -mattr=micromips -filetype=asm \
; RUN: -relocation-model=pic -O3 -o - | FileCheck %s

; The purpose of this test is to check whether the CodeGen selects
; LW16 instruction with the base register in a range of $2-$7, $16, $17.

%struct.T = type { i32 }

$_ZN1TaSERKS_ = comdat any

define linkonce_odr void @_ZN1TaSERKS_(%struct.T* %this, %struct.T* dereferenceable(4) %t) #0 comdat align 2 {
entry:
  %this.addr = alloca %struct.T*, align 4
  %t.addr = alloca %struct.T*, align 4
  %this1 = load %struct.T*, %struct.T** %this.addr, align 4
  %0 = load %struct.T*, %struct.T** %t.addr, align 4
  %V3 = getelementptr inbounds %struct.T, %struct.T* %0, i32 0, i32 0
  %1 = load i32, i32* %V3, align 4
  %V4 = getelementptr inbounds %struct.T, %struct.T* %this1, i32 0, i32 0
  store i32 %1, i32* %V4, align 4
  ret void
}

; CHECK: lw16 ${{[0-9]+}}, 0(${{[2-7]|16|17}})
