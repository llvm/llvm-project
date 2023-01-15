; RUN: llc %s -march=mips -mcpu=mips32r3 -mattr=micromips -filetype=asm \
; RUN: -relocation-model=pic -O3 -o - | FileCheck %s

; The purpose of this test is to check whether the CodeGen selects
; LW16 instruction with the base register in a range of $2-$7, $16, $17.

%struct.T = type { i32 }

$_ZN1TaSERKS_ = comdat any

define linkonce_odr void @_ZN1TaSERKS_(ptr %this, ptr dereferenceable(4) %t) #0 comdat align 2 {
entry:
  %this.addr = alloca ptr, align 4
  %t.addr = alloca ptr, align 4
  %this1 = load ptr, ptr %this.addr, align 4
  %0 = load ptr, ptr %t.addr, align 4
  %1 = load i32, ptr %0, align 4
  store i32 %1, ptr %this1, align 4
  ret void
}

; CHECK: lw16 ${{[0-9]+}}, 0(${{[2-7]|16|17}})
