; RUN: not llc -mtriple=amdgcn -mcpu=fiji -O0 -global-isel -global-isel-abort=2 -pass-remarks-missed='gisel*' %s -o - 2>&1 | FileCheck %s

; CHECK: warning: invalid constraint '': aggregate input operands not supported yet
define amdgpu_kernel void @aggregates([4 x i8] %val) {
  tail call void asm sideeffect "s_nop", "r"([4 x i8] %val)
  ret void
}

; CHECK: warning: invalid constraint '{s999}': could not allocate output register for constraint
define amdgpu_kernel void @bad_output() {
  tail call i32 asm sideeffect "s_nop", "={s999}"()
  ret void
}

; CHECK: warning: invalid constraint '{s998}': could not allocate input register for register constraint
define amdgpu_kernel void @bad_input() {
  tail call void asm sideeffect "s_nop", "{s998}"(i32 poison)
  ret void
}
; CHECK: warning: invalid constraint '{s997}': indirect register inputs are not supported yet
define amdgpu_kernel void @indirect_input() {
  tail call void asm sideeffect "s_nop", "*{s997}"(ptr elementtype(i32) poison)
  ret void
}

; CHECK: warning: invalid constraint 'i': unsupported constraint
define amdgpu_kernel void @badimm() {
  tail call void asm sideeffect "s_nop", "i"(i32 poison)
  ret void
}
