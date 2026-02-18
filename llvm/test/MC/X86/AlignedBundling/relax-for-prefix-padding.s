# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o - --x86-prefix-pad-for-bundle=1 --x86-pad-max-prefix-size=1 \
# RUN:   | llvm-objdump -d --no-show-raw-insn - | FileCheck %s

  .text
  .bundle_align_mode 5
  .p2align 5

  .rept 3
  callq x                     # 5 bytes each
  .endr
# CHECK:       f: jmp
# CHECK-NEXT: 14: int3
  jmp near_target             # 2 bytes (rel8), has to be relaxed to 5 bytes (rel32)
  .rept 8
  int3                        # 1 byte each, 7 of them are prefix-padded
  .endr
  ## trailing NOPs are only consumed by prefix-padding the int3 instructions
# CHECK:      1e: int3
# CHECK-NEXT: 20: int3

  ## Three full bundles of spacer to push near_target close to rel8 max range
  .rept 3
  .bundle_lock
  .rept 32
  int3
  .endr
  .bundle_unlock
  .endr

  .rept 15
  int3
  .endr
  ## With prefix padding (max 1): instructions absorb trailing NOPs,
  ##   near_target shifts from 0x8f to 0x9e.
  ##   jmp distance = 0x9d - (0xf + len(jmp)) = 0x8c (140), exceeds rel8 range,
  ##   forcing relaxation to rel32.
# CHECK:           <near_target>:
# CHECK-NEXT:      9d: inc
near_target:
  inc %eax

  .bundle_lock
  .rept 32
  int3
  .endr
  .bundle_unlock
