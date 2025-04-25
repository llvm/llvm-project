; RUN: llc -S -mtriple=amdgcn -mcpu=fiji -O0 -global-isel -global-isel-abort=2 -pass-remarks-missed='gisel*' %s -o - 2> %t.err

; CHECK: error: invalid constraint '' in 'aggregates': aggregate input operands not supported yet
define amdgpu_kernel void @aggregates([4 x i8] %val) {
  tail call void asm sideeffect "s_nop", "r"([4 x i8] %val)
  ret void
}

; CHECK: error: error: invalid constraint '{s999}' in 'bad_output': could not allocate output register for constraint
define amdgpu_kernel void @bad_output() {
  tail call i32 asm sideeffect "s_nop", "={s999}"()
  ret void
}
