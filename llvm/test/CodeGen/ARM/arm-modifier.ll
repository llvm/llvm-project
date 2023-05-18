; RUN: llc -mtriple=arm-eabi -mattr=+vfp2 -no-integrated-as %s -o - | FileCheck %s

define i32 @foo(float %scale, float %scale2) nounwind {
entry:
  %scale.addr = alloca float, align 4
  %scale2.addr = alloca float, align 4
  store float %scale, ptr %scale.addr, align 4
  store float %scale2, ptr %scale2.addr, align 4
  %tmp = load float, ptr %scale.addr, align 4
  %tmp1 = load float, ptr %scale2.addr, align 4
  call void asm sideeffect "vmul.f32    q0, q0, ${0:y} \0A\09vmul.f32    q1, q1, ${0:y} \0A\09vmul.f32    q1, q0, ${1:y} \0A\09", "w,w,~{q0},~{q1}"(float %tmp, float %tmp1) nounwind
  ret i32 0
}

define void @f0() nounwind {
entry:
; CHECK: f0
; CHECK: .word -1
call void asm sideeffect ".word ${0:B} \0A\09", "i"(i32 0) nounwind
ret void
}

define void @f1() nounwind {
entry:
; CHECK: f1
; CHECK: .word 65535
call void asm sideeffect ".word ${0:L} \0A\09", "i"(i32 -1) nounwind
ret void
}

@f2_ptr = internal global ptr @f2_var, align 4
@f2_var = external global i32

define void @f2() nounwind {
entry:
; CHECK: f2
; CHECK: ldr r0, [r{{[0-9]+}}]
call void asm sideeffect "ldr r0, [${0:m}]\0A\09", "*m,~{r0}"(ptr elementtype(ptr) @f2_ptr) nounwind
ret void
}

@f3_ptr = internal global ptr @f3_var, align 4
@f3_var = external global i64
@f3_var2 = external global i64

define void @f3() nounwind {
entry:
; CHECK: f3
; CHECK: stm {{lr|r[0-9]+}}, {[[REG1:(r[0-9]+)]], r{{[0-9]+}}}
; CHECK: adds {{lr|r[0-9]+}}, [[REG1]]
; CHECK: ldm {{lr|r[0-9]+}}, {r{{[0-9]+}}, r{{[0-9]+}}}
%tmp = load i64, ptr @f3_var, align 4
%tmp1 = load i64, ptr @f3_var2, align 4
%0 = call i64 asm sideeffect "stm ${0:m}, ${1:M}\0A\09adds $3, $1\0A\09", "=*m,=r,1,r"(ptr elementtype(ptr) @f3_ptr, i64 %tmp, i64 %tmp1) nounwind
store i64 %0, ptr @f3_var, align 4
%1 = call i64 asm sideeffect "ldm ${1:m}, ${0:M}\0A\09", "=r,*m"(ptr elementtype(ptr) @f3_ptr) nounwind
store i64 %1, ptr @f3_var, align 4
ret void
}

define i64 @f4(ptr %val) nounwind {
entry:
  ;CHECK-LABEL: f4:
  ;CHECK: ldrexd [[REG1:(r[0-9]?[02468])]], {{r[0-9]?[13579]}}, [r{{[0-9]+}}]
  %0 = tail call i64 asm sideeffect "ldrexd $0, ${0:H}, [$1]", "=&r,r,*Qo"(ptr %val, ptr elementtype(i64) %val) nounwind
  ret i64 %0
}

; PR16490
define void @f5(i64 %__pu_val) {
  call void asm sideeffect "$1", "r,i"(i64 %__pu_val, i32 -14)
  ret void
}
