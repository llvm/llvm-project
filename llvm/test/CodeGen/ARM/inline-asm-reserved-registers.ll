; RUN: not llc -mtriple thumbv6m-arm-none-eabi -frame-pointer=all %s -o - 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

; CHECK-ERROR: error: write to reserved register 'R7'
define void @test_framepointer_output(i32 %input) {
entry:
  %0 = call i32 asm sideeffect "mov $0, $1", "={r7},r"(i32 %input)
  ret void
}

; CHECK-ERROR: error: write to reserved register 'R7'
define void @test_framepointer_input(i32 %input) {
entry:
  %0 = call i32 asm sideeffect "mov $0, $1", "=r,{r7}"(i32 %input)
  ret void
}

; CHECK-ERROR: error: write to reserved register 'PC'
define void @test_pc_output(i32 %input) {
entry:
  %0 = call i32 asm sideeffect "mov $0, $1", "={pc},r"(i32 %input)
  ret void
}

; CHECK-ERROR: error: write to reserved register 'PC'
define void @test_pc_input(i32 %input) {
entry:
  %0 = call i32 asm sideeffect "mov $0, $1", "=r,{pc}"(i32 %input)
  ret void
}

; CHECK-ERROR: error: write to reserved register 'R6'
define void @test_basepointer_output(i32 %size, i32 %input) alignstack(8) {
entry:
  %vla = alloca i32, i32 %size, align 4
  %0 = call i32 asm sideeffect "mov $0, $1", "={r6},r"(i32 %input)
  ret void
}

; CHECK-ERROR: error: write to reserved register 'R6'
define void @test_basepointer_input(i32 %size, i32 %input) alignstack(8) {
entry:
  %vla = alloca i32, i32 %size, align 4
  %0 = call i32 asm sideeffect "mov $0, $1", "=r,{r6}"(i32 %input)
  ret void
}
