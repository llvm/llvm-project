.globl _start
.type _start, %function
_start:
  // Set FPU S registers to their register number plus 1.
  mov r0, #1
  vmov s0, r0
  mov r0, #2
  vmov s1, r0
  mov r0, #3
  vmov s2, r0
  mov r0, #4
  vmov s3, r0
  mov r0, #5
  vmov s4, r0
  mov r0, #6
  vmov s5, r0
  mov r0, #7
  vmov s6, r0
  mov r0, #8
  vmov s7, r0
  mov r0, #9
  vmov s8, r0
  mov r0, #10
  vmov s9, r0
  mov r0, #11
  vmov s10, r0
  mov r0, #12
  vmov s11, r0
  mov r0, #13
  vmov s12, r0
  mov r0, #14
  vmov s13, r0
  mov r0, #15
  vmov s14, r0
  mov r0, #16
  vmov s15, r0
  mov r0, #17
  vmov s16, r0
  mov r0, #18
  vmov s17, r0
  mov r0, #19
  vmov s18, r0
  mov r0, #20
  vmov s19, r0
  mov r0, #21
  vmov s20, r0
  mov r0, #22
  vmov s21, r0
  mov r0, #23
  vmov s22, r0
  mov r0, #24
  vmov s23, r0
  mov r0, #25
  vmov s24, r0
  mov r0, #26
  vmov s25, r0
  mov r0, #27
  vmov s26, r0
  mov r0, #28
  vmov s27, r0
  mov r0, #29
  vmov s28, r0
  mov r0, #30
  vmov s29, r0
  mov r0, #31
  vmov s30, r0
  mov r0, #32
  vmov s31, r0

  // Set GPRs to their register number.
  mov r0, #0
  mov r1, #1
  mov r2, #2
  mov r3, #3
  mov r4, #4
  mov r5, #5
  mov r6, #6
  mov r7, #7
  mov r8, #8
  mov r9, #9
  mov r10, #10
  mov r11, #11
  mov r12, #12
  mov r13, #13
  mov r14, #14
  // r15 is the PC, leave this alone.
end: // Loop forever.
  b end

