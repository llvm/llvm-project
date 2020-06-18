# RUN: llvm-mc -filetype=obj -triple i386-pc-linux-gnu %s -o - \
# RUN:   | llvm-objdump --triple=i386 -d --no-show-raw-insn - | FileCheck %s

# !!! This test is auto-generated from utils/testgen/mc-bundling-x86-gen.py !!!
#     It tests that bundle-aligned grouping works correctly in MC. Read the
#     source of the script for more details.

  .text
  .bundle_align_mode 4

  .align 32, 0x90
INSTRLEN_1_OFFSET_0:
  .bundle_lock
  .rept 1
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 0: incl

  .align 32, 0x90
INSTRLEN_1_OFFSET_1:
  .fill 1, 1, 0x90
  .bundle_lock
  .rept 1
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 21: incl

  .align 32, 0x90
INSTRLEN_1_OFFSET_2:
  .fill 2, 1, 0x90
  .bundle_lock
  .rept 1
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 42: incl

  .align 32, 0x90
INSTRLEN_1_OFFSET_3:
  .fill 3, 1, 0x90
  .bundle_lock
  .rept 1
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 63: incl

  .align 32, 0x90
INSTRLEN_1_OFFSET_4:
  .fill 4, 1, 0x90
  .bundle_lock
  .rept 1
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 84: incl

  .align 32, 0x90
INSTRLEN_1_OFFSET_5:
  .fill 5, 1, 0x90
  .bundle_lock
  .rept 1
  inc %eax
  .endr
  .bundle_unlock
# CHECK: a5: incl

  .align 32, 0x90
INSTRLEN_1_OFFSET_6:
  .fill 6, 1, 0x90
  .bundle_lock
  .rept 1
  inc %eax
  .endr
  .bundle_unlock
# CHECK: c6: incl

  .align 32, 0x90
INSTRLEN_1_OFFSET_7:
  .fill 7, 1, 0x90
  .bundle_lock
  .rept 1
  inc %eax
  .endr
  .bundle_unlock
# CHECK: e7: incl

  .align 32, 0x90
INSTRLEN_1_OFFSET_8:
  .fill 8, 1, 0x90
  .bundle_lock
  .rept 1
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 108: incl

  .align 32, 0x90
INSTRLEN_1_OFFSET_9:
  .fill 9, 1, 0x90
  .bundle_lock
  .rept 1
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 129: incl

  .align 32, 0x90
INSTRLEN_1_OFFSET_10:
  .fill 10, 1, 0x90
  .bundle_lock
  .rept 1
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 14a: incl

  .align 32, 0x90
INSTRLEN_1_OFFSET_11:
  .fill 11, 1, 0x90
  .bundle_lock
  .rept 1
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 16b: incl

  .align 32, 0x90
INSTRLEN_1_OFFSET_12:
  .fill 12, 1, 0x90
  .bundle_lock
  .rept 1
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 18c: incl

  .align 32, 0x90
INSTRLEN_1_OFFSET_13:
  .fill 13, 1, 0x90
  .bundle_lock
  .rept 1
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1ad: incl

  .align 32, 0x90
INSTRLEN_1_OFFSET_14:
  .fill 14, 1, 0x90
  .bundle_lock
  .rept 1
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1ce: incl

  .align 32, 0x90
INSTRLEN_1_OFFSET_15:
  .fill 15, 1, 0x90
  .bundle_lock
  .rept 1
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1ef: incl

  .align 32, 0x90
INSTRLEN_2_OFFSET_0:
  .bundle_lock
  .rept 2
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 200: incl

  .align 32, 0x90
INSTRLEN_2_OFFSET_1:
  .fill 1, 1, 0x90
  .bundle_lock
  .rept 2
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 221: incl

  .align 32, 0x90
INSTRLEN_2_OFFSET_2:
  .fill 2, 1, 0x90
  .bundle_lock
  .rept 2
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 242: incl

  .align 32, 0x90
INSTRLEN_2_OFFSET_3:
  .fill 3, 1, 0x90
  .bundle_lock
  .rept 2
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 263: incl

  .align 32, 0x90
INSTRLEN_2_OFFSET_4:
  .fill 4, 1, 0x90
  .bundle_lock
  .rept 2
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 284: incl

  .align 32, 0x90
INSTRLEN_2_OFFSET_5:
  .fill 5, 1, 0x90
  .bundle_lock
  .rept 2
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 2a5: incl

  .align 32, 0x90
INSTRLEN_2_OFFSET_6:
  .fill 6, 1, 0x90
  .bundle_lock
  .rept 2
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 2c6: incl

  .align 32, 0x90
INSTRLEN_2_OFFSET_7:
  .fill 7, 1, 0x90
  .bundle_lock
  .rept 2
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 2e7: incl

  .align 32, 0x90
INSTRLEN_2_OFFSET_8:
  .fill 8, 1, 0x90
  .bundle_lock
  .rept 2
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 308: incl

  .align 32, 0x90
INSTRLEN_2_OFFSET_9:
  .fill 9, 1, 0x90
  .bundle_lock
  .rept 2
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 329: incl

  .align 32, 0x90
INSTRLEN_2_OFFSET_10:
  .fill 10, 1, 0x90
  .bundle_lock
  .rept 2
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 34a: incl

  .align 32, 0x90
INSTRLEN_2_OFFSET_11:
  .fill 11, 1, 0x90
  .bundle_lock
  .rept 2
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 36b: incl

  .align 32, 0x90
INSTRLEN_2_OFFSET_12:
  .fill 12, 1, 0x90
  .bundle_lock
  .rept 2
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 38c: incl

  .align 32, 0x90
INSTRLEN_2_OFFSET_13:
  .fill 13, 1, 0x90
  .bundle_lock
  .rept 2
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 3ad: incl

  .align 32, 0x90
INSTRLEN_2_OFFSET_14:
  .fill 14, 1, 0x90
  .bundle_lock
  .rept 2
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 3ce: incl

  .align 32, 0x90
INSTRLEN_2_OFFSET_15:
  .fill 15, 1, 0x90
  .bundle_lock
  .rept 2
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 3ef: nop
# CHECK: 3f0: incl

  .align 32, 0x90
INSTRLEN_3_OFFSET_0:
  .bundle_lock
  .rept 3
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 400: incl

  .align 32, 0x90
INSTRLEN_3_OFFSET_1:
  .fill 1, 1, 0x90
  .bundle_lock
  .rept 3
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 421: incl

  .align 32, 0x90
INSTRLEN_3_OFFSET_2:
  .fill 2, 1, 0x90
  .bundle_lock
  .rept 3
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 442: incl

  .align 32, 0x90
INSTRLEN_3_OFFSET_3:
  .fill 3, 1, 0x90
  .bundle_lock
  .rept 3
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 463: incl

  .align 32, 0x90
INSTRLEN_3_OFFSET_4:
  .fill 4, 1, 0x90
  .bundle_lock
  .rept 3
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 484: incl

  .align 32, 0x90
INSTRLEN_3_OFFSET_5:
  .fill 5, 1, 0x90
  .bundle_lock
  .rept 3
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 4a5: incl

  .align 32, 0x90
INSTRLEN_3_OFFSET_6:
  .fill 6, 1, 0x90
  .bundle_lock
  .rept 3
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 4c6: incl

  .align 32, 0x90
INSTRLEN_3_OFFSET_7:
  .fill 7, 1, 0x90
  .bundle_lock
  .rept 3
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 4e7: incl

  .align 32, 0x90
INSTRLEN_3_OFFSET_8:
  .fill 8, 1, 0x90
  .bundle_lock
  .rept 3
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 508: incl

  .align 32, 0x90
INSTRLEN_3_OFFSET_9:
  .fill 9, 1, 0x90
  .bundle_lock
  .rept 3
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 529: incl

  .align 32, 0x90
INSTRLEN_3_OFFSET_10:
  .fill 10, 1, 0x90
  .bundle_lock
  .rept 3
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 54a: incl

  .align 32, 0x90
INSTRLEN_3_OFFSET_11:
  .fill 11, 1, 0x90
  .bundle_lock
  .rept 3
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 56b: incl

  .align 32, 0x90
INSTRLEN_3_OFFSET_12:
  .fill 12, 1, 0x90
  .bundle_lock
  .rept 3
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 58c: incl

  .align 32, 0x90
INSTRLEN_3_OFFSET_13:
  .fill 13, 1, 0x90
  .bundle_lock
  .rept 3
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 5ad: incl

  .align 32, 0x90
INSTRLEN_3_OFFSET_14:
  .fill 14, 1, 0x90
  .bundle_lock
  .rept 3
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 5ce: nop
# CHECK: 5d0: incl

  .align 32, 0x90
INSTRLEN_3_OFFSET_15:
  .fill 15, 1, 0x90
  .bundle_lock
  .rept 3
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 5ef: nop
# CHECK: 5f0: incl

  .align 32, 0x90
INSTRLEN_4_OFFSET_0:
  .bundle_lock
  .rept 4
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 600: incl

  .align 32, 0x90
INSTRLEN_4_OFFSET_1:
  .fill 1, 1, 0x90
  .bundle_lock
  .rept 4
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 621: incl

  .align 32, 0x90
INSTRLEN_4_OFFSET_2:
  .fill 2, 1, 0x90
  .bundle_lock
  .rept 4
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 642: incl

  .align 32, 0x90
INSTRLEN_4_OFFSET_3:
  .fill 3, 1, 0x90
  .bundle_lock
  .rept 4
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 663: incl

  .align 32, 0x90
INSTRLEN_4_OFFSET_4:
  .fill 4, 1, 0x90
  .bundle_lock
  .rept 4
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 684: incl

  .align 32, 0x90
INSTRLEN_4_OFFSET_5:
  .fill 5, 1, 0x90
  .bundle_lock
  .rept 4
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 6a5: incl

  .align 32, 0x90
INSTRLEN_4_OFFSET_6:
  .fill 6, 1, 0x90
  .bundle_lock
  .rept 4
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 6c6: incl

  .align 32, 0x90
INSTRLEN_4_OFFSET_7:
  .fill 7, 1, 0x90
  .bundle_lock
  .rept 4
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 6e7: incl

  .align 32, 0x90
INSTRLEN_4_OFFSET_8:
  .fill 8, 1, 0x90
  .bundle_lock
  .rept 4
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 708: incl

  .align 32, 0x90
INSTRLEN_4_OFFSET_9:
  .fill 9, 1, 0x90
  .bundle_lock
  .rept 4
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 729: incl

  .align 32, 0x90
INSTRLEN_4_OFFSET_10:
  .fill 10, 1, 0x90
  .bundle_lock
  .rept 4
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 74a: incl

  .align 32, 0x90
INSTRLEN_4_OFFSET_11:
  .fill 11, 1, 0x90
  .bundle_lock
  .rept 4
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 76b: incl

  .align 32, 0x90
INSTRLEN_4_OFFSET_12:
  .fill 12, 1, 0x90
  .bundle_lock
  .rept 4
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 78c: incl

  .align 32, 0x90
INSTRLEN_4_OFFSET_13:
  .fill 13, 1, 0x90
  .bundle_lock
  .rept 4
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 7ad: nop
# CHECK: 7b0: incl

  .align 32, 0x90
INSTRLEN_4_OFFSET_14:
  .fill 14, 1, 0x90
  .bundle_lock
  .rept 4
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 7ce: nop
# CHECK: 7d0: incl

  .align 32, 0x90
INSTRLEN_4_OFFSET_15:
  .fill 15, 1, 0x90
  .bundle_lock
  .rept 4
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 7ef: nop
# CHECK: 7f0: incl

  .align 32, 0x90
INSTRLEN_5_OFFSET_0:
  .bundle_lock
  .rept 5
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 800: incl

  .align 32, 0x90
INSTRLEN_5_OFFSET_1:
  .fill 1, 1, 0x90
  .bundle_lock
  .rept 5
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 821: incl

  .align 32, 0x90
INSTRLEN_5_OFFSET_2:
  .fill 2, 1, 0x90
  .bundle_lock
  .rept 5
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 842: incl

  .align 32, 0x90
INSTRLEN_5_OFFSET_3:
  .fill 3, 1, 0x90
  .bundle_lock
  .rept 5
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 863: incl

  .align 32, 0x90
INSTRLEN_5_OFFSET_4:
  .fill 4, 1, 0x90
  .bundle_lock
  .rept 5
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 884: incl

  .align 32, 0x90
INSTRLEN_5_OFFSET_5:
  .fill 5, 1, 0x90
  .bundle_lock
  .rept 5
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 8a5: incl

  .align 32, 0x90
INSTRLEN_5_OFFSET_6:
  .fill 6, 1, 0x90
  .bundle_lock
  .rept 5
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 8c6: incl

  .align 32, 0x90
INSTRLEN_5_OFFSET_7:
  .fill 7, 1, 0x90
  .bundle_lock
  .rept 5
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 8e7: incl

  .align 32, 0x90
INSTRLEN_5_OFFSET_8:
  .fill 8, 1, 0x90
  .bundle_lock
  .rept 5
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 908: incl

  .align 32, 0x90
INSTRLEN_5_OFFSET_9:
  .fill 9, 1, 0x90
  .bundle_lock
  .rept 5
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 929: incl

  .align 32, 0x90
INSTRLEN_5_OFFSET_10:
  .fill 10, 1, 0x90
  .bundle_lock
  .rept 5
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 94a: incl

  .align 32, 0x90
INSTRLEN_5_OFFSET_11:
  .fill 11, 1, 0x90
  .bundle_lock
  .rept 5
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 96b: incl

  .align 32, 0x90
INSTRLEN_5_OFFSET_12:
  .fill 12, 1, 0x90
  .bundle_lock
  .rept 5
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 98c: nop
# CHECK: 990: incl

  .align 32, 0x90
INSTRLEN_5_OFFSET_13:
  .fill 13, 1, 0x90
  .bundle_lock
  .rept 5
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 9ad: nop
# CHECK: 9b0: incl

  .align 32, 0x90
INSTRLEN_5_OFFSET_14:
  .fill 14, 1, 0x90
  .bundle_lock
  .rept 5
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 9ce: nop
# CHECK: 9d0: incl

  .align 32, 0x90
INSTRLEN_5_OFFSET_15:
  .fill 15, 1, 0x90
  .bundle_lock
  .rept 5
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 9ef: nop
# CHECK: 9f0: incl

  .align 32, 0x90
INSTRLEN_6_OFFSET_0:
  .bundle_lock
  .rept 6
  inc %eax
  .endr
  .bundle_unlock
# CHECK: a00: incl

  .align 32, 0x90
INSTRLEN_6_OFFSET_1:
  .fill 1, 1, 0x90
  .bundle_lock
  .rept 6
  inc %eax
  .endr
  .bundle_unlock
# CHECK: a21: incl

  .align 32, 0x90
INSTRLEN_6_OFFSET_2:
  .fill 2, 1, 0x90
  .bundle_lock
  .rept 6
  inc %eax
  .endr
  .bundle_unlock
# CHECK: a42: incl

  .align 32, 0x90
INSTRLEN_6_OFFSET_3:
  .fill 3, 1, 0x90
  .bundle_lock
  .rept 6
  inc %eax
  .endr
  .bundle_unlock
# CHECK: a63: incl

  .align 32, 0x90
INSTRLEN_6_OFFSET_4:
  .fill 4, 1, 0x90
  .bundle_lock
  .rept 6
  inc %eax
  .endr
  .bundle_unlock
# CHECK: a84: incl

  .align 32, 0x90
INSTRLEN_6_OFFSET_5:
  .fill 5, 1, 0x90
  .bundle_lock
  .rept 6
  inc %eax
  .endr
  .bundle_unlock
# CHECK: aa5: incl

  .align 32, 0x90
INSTRLEN_6_OFFSET_6:
  .fill 6, 1, 0x90
  .bundle_lock
  .rept 6
  inc %eax
  .endr
  .bundle_unlock
# CHECK: ac6: incl

  .align 32, 0x90
INSTRLEN_6_OFFSET_7:
  .fill 7, 1, 0x90
  .bundle_lock
  .rept 6
  inc %eax
  .endr
  .bundle_unlock
# CHECK: ae7: incl

  .align 32, 0x90
INSTRLEN_6_OFFSET_8:
  .fill 8, 1, 0x90
  .bundle_lock
  .rept 6
  inc %eax
  .endr
  .bundle_unlock
# CHECK: b08: incl

  .align 32, 0x90
INSTRLEN_6_OFFSET_9:
  .fill 9, 1, 0x90
  .bundle_lock
  .rept 6
  inc %eax
  .endr
  .bundle_unlock
# CHECK: b29: incl

  .align 32, 0x90
INSTRLEN_6_OFFSET_10:
  .fill 10, 1, 0x90
  .bundle_lock
  .rept 6
  inc %eax
  .endr
  .bundle_unlock
# CHECK: b4a: incl

  .align 32, 0x90
INSTRLEN_6_OFFSET_11:
  .fill 11, 1, 0x90
  .bundle_lock
  .rept 6
  inc %eax
  .endr
  .bundle_unlock
# CHECK: b6b: nop
# CHECK: b70: incl

  .align 32, 0x90
INSTRLEN_6_OFFSET_12:
  .fill 12, 1, 0x90
  .bundle_lock
  .rept 6
  inc %eax
  .endr
  .bundle_unlock
# CHECK: b8c: nop
# CHECK: b90: incl

  .align 32, 0x90
INSTRLEN_6_OFFSET_13:
  .fill 13, 1, 0x90
  .bundle_lock
  .rept 6
  inc %eax
  .endr
  .bundle_unlock
# CHECK: bad: nop
# CHECK: bb0: incl

  .align 32, 0x90
INSTRLEN_6_OFFSET_14:
  .fill 14, 1, 0x90
  .bundle_lock
  .rept 6
  inc %eax
  .endr
  .bundle_unlock
# CHECK: bce: nop
# CHECK: bd0: incl

  .align 32, 0x90
INSTRLEN_6_OFFSET_15:
  .fill 15, 1, 0x90
  .bundle_lock
  .rept 6
  inc %eax
  .endr
  .bundle_unlock
# CHECK: bef: nop
# CHECK: bf0: incl

  .align 32, 0x90
INSTRLEN_7_OFFSET_0:
  .bundle_lock
  .rept 7
  inc %eax
  .endr
  .bundle_unlock
# CHECK: c00: incl

  .align 32, 0x90
INSTRLEN_7_OFFSET_1:
  .fill 1, 1, 0x90
  .bundle_lock
  .rept 7
  inc %eax
  .endr
  .bundle_unlock
# CHECK: c21: incl

  .align 32, 0x90
INSTRLEN_7_OFFSET_2:
  .fill 2, 1, 0x90
  .bundle_lock
  .rept 7
  inc %eax
  .endr
  .bundle_unlock
# CHECK: c42: incl

  .align 32, 0x90
INSTRLEN_7_OFFSET_3:
  .fill 3, 1, 0x90
  .bundle_lock
  .rept 7
  inc %eax
  .endr
  .bundle_unlock
# CHECK: c63: incl

  .align 32, 0x90
INSTRLEN_7_OFFSET_4:
  .fill 4, 1, 0x90
  .bundle_lock
  .rept 7
  inc %eax
  .endr
  .bundle_unlock
# CHECK: c84: incl

  .align 32, 0x90
INSTRLEN_7_OFFSET_5:
  .fill 5, 1, 0x90
  .bundle_lock
  .rept 7
  inc %eax
  .endr
  .bundle_unlock
# CHECK: ca5: incl

  .align 32, 0x90
INSTRLEN_7_OFFSET_6:
  .fill 6, 1, 0x90
  .bundle_lock
  .rept 7
  inc %eax
  .endr
  .bundle_unlock
# CHECK: cc6: incl

  .align 32, 0x90
INSTRLEN_7_OFFSET_7:
  .fill 7, 1, 0x90
  .bundle_lock
  .rept 7
  inc %eax
  .endr
  .bundle_unlock
# CHECK: ce7: incl

  .align 32, 0x90
INSTRLEN_7_OFFSET_8:
  .fill 8, 1, 0x90
  .bundle_lock
  .rept 7
  inc %eax
  .endr
  .bundle_unlock
# CHECK: d08: incl

  .align 32, 0x90
INSTRLEN_7_OFFSET_9:
  .fill 9, 1, 0x90
  .bundle_lock
  .rept 7
  inc %eax
  .endr
  .bundle_unlock
# CHECK: d29: incl

  .align 32, 0x90
INSTRLEN_7_OFFSET_10:
  .fill 10, 1, 0x90
  .bundle_lock
  .rept 7
  inc %eax
  .endr
  .bundle_unlock
# CHECK: d4a: nop
# CHECK: d50: incl

  .align 32, 0x90
INSTRLEN_7_OFFSET_11:
  .fill 11, 1, 0x90
  .bundle_lock
  .rept 7
  inc %eax
  .endr
  .bundle_unlock
# CHECK: d6b: nop
# CHECK: d70: incl

  .align 32, 0x90
INSTRLEN_7_OFFSET_12:
  .fill 12, 1, 0x90
  .bundle_lock
  .rept 7
  inc %eax
  .endr
  .bundle_unlock
# CHECK: d8c: nop
# CHECK: d90: incl

  .align 32, 0x90
INSTRLEN_7_OFFSET_13:
  .fill 13, 1, 0x90
  .bundle_lock
  .rept 7
  inc %eax
  .endr
  .bundle_unlock
# CHECK: dad: nop
# CHECK: db0: incl

  .align 32, 0x90
INSTRLEN_7_OFFSET_14:
  .fill 14, 1, 0x90
  .bundle_lock
  .rept 7
  inc %eax
  .endr
  .bundle_unlock
# CHECK: dce: nop
# CHECK: dd0: incl

  .align 32, 0x90
INSTRLEN_7_OFFSET_15:
  .fill 15, 1, 0x90
  .bundle_lock
  .rept 7
  inc %eax
  .endr
  .bundle_unlock
# CHECK: def: nop
# CHECK: df0: incl

  .align 32, 0x90
INSTRLEN_8_OFFSET_0:
  .bundle_lock
  .rept 8
  inc %eax
  .endr
  .bundle_unlock
# CHECK: e00: incl

  .align 32, 0x90
INSTRLEN_8_OFFSET_1:
  .fill 1, 1, 0x90
  .bundle_lock
  .rept 8
  inc %eax
  .endr
  .bundle_unlock
# CHECK: e21: incl

  .align 32, 0x90
INSTRLEN_8_OFFSET_2:
  .fill 2, 1, 0x90
  .bundle_lock
  .rept 8
  inc %eax
  .endr
  .bundle_unlock
# CHECK: e42: incl

  .align 32, 0x90
INSTRLEN_8_OFFSET_3:
  .fill 3, 1, 0x90
  .bundle_lock
  .rept 8
  inc %eax
  .endr
  .bundle_unlock
# CHECK: e63: incl

  .align 32, 0x90
INSTRLEN_8_OFFSET_4:
  .fill 4, 1, 0x90
  .bundle_lock
  .rept 8
  inc %eax
  .endr
  .bundle_unlock
# CHECK: e84: incl

  .align 32, 0x90
INSTRLEN_8_OFFSET_5:
  .fill 5, 1, 0x90
  .bundle_lock
  .rept 8
  inc %eax
  .endr
  .bundle_unlock
# CHECK: ea5: incl

  .align 32, 0x90
INSTRLEN_8_OFFSET_6:
  .fill 6, 1, 0x90
  .bundle_lock
  .rept 8
  inc %eax
  .endr
  .bundle_unlock
# CHECK: ec6: incl

  .align 32, 0x90
INSTRLEN_8_OFFSET_7:
  .fill 7, 1, 0x90
  .bundle_lock
  .rept 8
  inc %eax
  .endr
  .bundle_unlock
# CHECK: ee7: incl

  .align 32, 0x90
INSTRLEN_8_OFFSET_8:
  .fill 8, 1, 0x90
  .bundle_lock
  .rept 8
  inc %eax
  .endr
  .bundle_unlock
# CHECK: f08: incl

  .align 32, 0x90
INSTRLEN_8_OFFSET_9:
  .fill 9, 1, 0x90
  .bundle_lock
  .rept 8
  inc %eax
  .endr
  .bundle_unlock
# CHECK: f29: nop
# CHECK: f30: incl

  .align 32, 0x90
INSTRLEN_8_OFFSET_10:
  .fill 10, 1, 0x90
  .bundle_lock
  .rept 8
  inc %eax
  .endr
  .bundle_unlock
# CHECK: f4a: nop
# CHECK: f50: incl

  .align 32, 0x90
INSTRLEN_8_OFFSET_11:
  .fill 11, 1, 0x90
  .bundle_lock
  .rept 8
  inc %eax
  .endr
  .bundle_unlock
# CHECK: f6b: nop
# CHECK: f70: incl

  .align 32, 0x90
INSTRLEN_8_OFFSET_12:
  .fill 12, 1, 0x90
  .bundle_lock
  .rept 8
  inc %eax
  .endr
  .bundle_unlock
# CHECK: f8c: nop
# CHECK: f90: incl

  .align 32, 0x90
INSTRLEN_8_OFFSET_13:
  .fill 13, 1, 0x90
  .bundle_lock
  .rept 8
  inc %eax
  .endr
  .bundle_unlock
# CHECK: fad: nop
# CHECK: fb0: incl

  .align 32, 0x90
INSTRLEN_8_OFFSET_14:
  .fill 14, 1, 0x90
  .bundle_lock
  .rept 8
  inc %eax
  .endr
  .bundle_unlock
# CHECK: fce: nop
# CHECK: fd0: incl

  .align 32, 0x90
INSTRLEN_8_OFFSET_15:
  .fill 15, 1, 0x90
  .bundle_lock
  .rept 8
  inc %eax
  .endr
  .bundle_unlock
# CHECK: fef: nop
# CHECK: ff0: incl

  .align 32, 0x90
INSTRLEN_9_OFFSET_0:
  .bundle_lock
  .rept 9
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1000: incl

  .align 32, 0x90
INSTRLEN_9_OFFSET_1:
  .fill 1, 1, 0x90
  .bundle_lock
  .rept 9
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1021: incl

  .align 32, 0x90
INSTRLEN_9_OFFSET_2:
  .fill 2, 1, 0x90
  .bundle_lock
  .rept 9
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1042: incl

  .align 32, 0x90
INSTRLEN_9_OFFSET_3:
  .fill 3, 1, 0x90
  .bundle_lock
  .rept 9
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1063: incl

  .align 32, 0x90
INSTRLEN_9_OFFSET_4:
  .fill 4, 1, 0x90
  .bundle_lock
  .rept 9
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1084: incl

  .align 32, 0x90
INSTRLEN_9_OFFSET_5:
  .fill 5, 1, 0x90
  .bundle_lock
  .rept 9
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 10a5: incl

  .align 32, 0x90
INSTRLEN_9_OFFSET_6:
  .fill 6, 1, 0x90
  .bundle_lock
  .rept 9
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 10c6: incl

  .align 32, 0x90
INSTRLEN_9_OFFSET_7:
  .fill 7, 1, 0x90
  .bundle_lock
  .rept 9
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 10e7: incl

  .align 32, 0x90
INSTRLEN_9_OFFSET_8:
  .fill 8, 1, 0x90
  .bundle_lock
  .rept 9
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1108: nop
# CHECK: 1110: incl

  .align 32, 0x90
INSTRLEN_9_OFFSET_9:
  .fill 9, 1, 0x90
  .bundle_lock
  .rept 9
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1129: nop
# CHECK: 1130: incl

  .align 32, 0x90
INSTRLEN_9_OFFSET_10:
  .fill 10, 1, 0x90
  .bundle_lock
  .rept 9
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 114a: nop
# CHECK: 1150: incl

  .align 32, 0x90
INSTRLEN_9_OFFSET_11:
  .fill 11, 1, 0x90
  .bundle_lock
  .rept 9
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 116b: nop
# CHECK: 1170: incl

  .align 32, 0x90
INSTRLEN_9_OFFSET_12:
  .fill 12, 1, 0x90
  .bundle_lock
  .rept 9
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 118c: nop
# CHECK: 1190: incl

  .align 32, 0x90
INSTRLEN_9_OFFSET_13:
  .fill 13, 1, 0x90
  .bundle_lock
  .rept 9
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 11ad: nop
# CHECK: 11b0: incl

  .align 32, 0x90
INSTRLEN_9_OFFSET_14:
  .fill 14, 1, 0x90
  .bundle_lock
  .rept 9
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 11ce: nop
# CHECK: 11d0: incl

  .align 32, 0x90
INSTRLEN_9_OFFSET_15:
  .fill 15, 1, 0x90
  .bundle_lock
  .rept 9
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 11ef: nop
# CHECK: 11f0: incl

  .align 32, 0x90
INSTRLEN_10_OFFSET_0:
  .bundle_lock
  .rept 10
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1200: incl

  .align 32, 0x90
INSTRLEN_10_OFFSET_1:
  .fill 1, 1, 0x90
  .bundle_lock
  .rept 10
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1221: incl

  .align 32, 0x90
INSTRLEN_10_OFFSET_2:
  .fill 2, 1, 0x90
  .bundle_lock
  .rept 10
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1242: incl

  .align 32, 0x90
INSTRLEN_10_OFFSET_3:
  .fill 3, 1, 0x90
  .bundle_lock
  .rept 10
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1263: incl

  .align 32, 0x90
INSTRLEN_10_OFFSET_4:
  .fill 4, 1, 0x90
  .bundle_lock
  .rept 10
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1284: incl

  .align 32, 0x90
INSTRLEN_10_OFFSET_5:
  .fill 5, 1, 0x90
  .bundle_lock
  .rept 10
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 12a5: incl

  .align 32, 0x90
INSTRLEN_10_OFFSET_6:
  .fill 6, 1, 0x90
  .bundle_lock
  .rept 10
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 12c6: incl

  .align 32, 0x90
INSTRLEN_10_OFFSET_7:
  .fill 7, 1, 0x90
  .bundle_lock
  .rept 10
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 12e7: nop
# CHECK: 12f0: incl

  .align 32, 0x90
INSTRLEN_10_OFFSET_8:
  .fill 8, 1, 0x90
  .bundle_lock
  .rept 10
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1308: nop
# CHECK: 1310: incl

  .align 32, 0x90
INSTRLEN_10_OFFSET_9:
  .fill 9, 1, 0x90
  .bundle_lock
  .rept 10
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1329: nop
# CHECK: 1330: incl

  .align 32, 0x90
INSTRLEN_10_OFFSET_10:
  .fill 10, 1, 0x90
  .bundle_lock
  .rept 10
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 134a: nop
# CHECK: 1350: incl

  .align 32, 0x90
INSTRLEN_10_OFFSET_11:
  .fill 11, 1, 0x90
  .bundle_lock
  .rept 10
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 136b: nop
# CHECK: 1370: incl

  .align 32, 0x90
INSTRLEN_10_OFFSET_12:
  .fill 12, 1, 0x90
  .bundle_lock
  .rept 10
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 138c: nop
# CHECK: 1390: incl

  .align 32, 0x90
INSTRLEN_10_OFFSET_13:
  .fill 13, 1, 0x90
  .bundle_lock
  .rept 10
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 13ad: nop
# CHECK: 13b0: incl

  .align 32, 0x90
INSTRLEN_10_OFFSET_14:
  .fill 14, 1, 0x90
  .bundle_lock
  .rept 10
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 13ce: nop
# CHECK: 13d0: incl

  .align 32, 0x90
INSTRLEN_10_OFFSET_15:
  .fill 15, 1, 0x90
  .bundle_lock
  .rept 10
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 13ef: nop
# CHECK: 13f0: incl

  .align 32, 0x90
INSTRLEN_11_OFFSET_0:
  .bundle_lock
  .rept 11
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1400: incl

  .align 32, 0x90
INSTRLEN_11_OFFSET_1:
  .fill 1, 1, 0x90
  .bundle_lock
  .rept 11
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1421: incl

  .align 32, 0x90
INSTRLEN_11_OFFSET_2:
  .fill 2, 1, 0x90
  .bundle_lock
  .rept 11
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1442: incl

  .align 32, 0x90
INSTRLEN_11_OFFSET_3:
  .fill 3, 1, 0x90
  .bundle_lock
  .rept 11
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1463: incl

  .align 32, 0x90
INSTRLEN_11_OFFSET_4:
  .fill 4, 1, 0x90
  .bundle_lock
  .rept 11
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1484: incl

  .align 32, 0x90
INSTRLEN_11_OFFSET_5:
  .fill 5, 1, 0x90
  .bundle_lock
  .rept 11
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 14a5: incl

  .align 32, 0x90
INSTRLEN_11_OFFSET_6:
  .fill 6, 1, 0x90
  .bundle_lock
  .rept 11
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 14c6: nop
# CHECK: 14d0: incl

  .align 32, 0x90
INSTRLEN_11_OFFSET_7:
  .fill 7, 1, 0x90
  .bundle_lock
  .rept 11
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 14e7: nop
# CHECK: 14f0: incl

  .align 32, 0x90
INSTRLEN_11_OFFSET_8:
  .fill 8, 1, 0x90
  .bundle_lock
  .rept 11
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1508: nop
# CHECK: 1510: incl

  .align 32, 0x90
INSTRLEN_11_OFFSET_9:
  .fill 9, 1, 0x90
  .bundle_lock
  .rept 11
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1529: nop
# CHECK: 1530: incl

  .align 32, 0x90
INSTRLEN_11_OFFSET_10:
  .fill 10, 1, 0x90
  .bundle_lock
  .rept 11
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 154a: nop
# CHECK: 1550: incl

  .align 32, 0x90
INSTRLEN_11_OFFSET_11:
  .fill 11, 1, 0x90
  .bundle_lock
  .rept 11
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 156b: nop
# CHECK: 1570: incl

  .align 32, 0x90
INSTRLEN_11_OFFSET_12:
  .fill 12, 1, 0x90
  .bundle_lock
  .rept 11
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 158c: nop
# CHECK: 1590: incl

  .align 32, 0x90
INSTRLEN_11_OFFSET_13:
  .fill 13, 1, 0x90
  .bundle_lock
  .rept 11
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 15ad: nop
# CHECK: 15b0: incl

  .align 32, 0x90
INSTRLEN_11_OFFSET_14:
  .fill 14, 1, 0x90
  .bundle_lock
  .rept 11
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 15ce: nop
# CHECK: 15d0: incl

  .align 32, 0x90
INSTRLEN_11_OFFSET_15:
  .fill 15, 1, 0x90
  .bundle_lock
  .rept 11
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 15ef: nop
# CHECK: 15f0: incl

  .align 32, 0x90
INSTRLEN_12_OFFSET_0:
  .bundle_lock
  .rept 12
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1600: incl

  .align 32, 0x90
INSTRLEN_12_OFFSET_1:
  .fill 1, 1, 0x90
  .bundle_lock
  .rept 12
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1621: incl

  .align 32, 0x90
INSTRLEN_12_OFFSET_2:
  .fill 2, 1, 0x90
  .bundle_lock
  .rept 12
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1642: incl

  .align 32, 0x90
INSTRLEN_12_OFFSET_3:
  .fill 3, 1, 0x90
  .bundle_lock
  .rept 12
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1663: incl

  .align 32, 0x90
INSTRLEN_12_OFFSET_4:
  .fill 4, 1, 0x90
  .bundle_lock
  .rept 12
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1684: incl

  .align 32, 0x90
INSTRLEN_12_OFFSET_5:
  .fill 5, 1, 0x90
  .bundle_lock
  .rept 12
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 16a5: nop
# CHECK: 16b0: incl

  .align 32, 0x90
INSTRLEN_12_OFFSET_6:
  .fill 6, 1, 0x90
  .bundle_lock
  .rept 12
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 16c6: nop
# CHECK: 16d0: incl

  .align 32, 0x90
INSTRLEN_12_OFFSET_7:
  .fill 7, 1, 0x90
  .bundle_lock
  .rept 12
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 16e7: nop
# CHECK: 16f0: incl

  .align 32, 0x90
INSTRLEN_12_OFFSET_8:
  .fill 8, 1, 0x90
  .bundle_lock
  .rept 12
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1708: nop
# CHECK: 1710: incl

  .align 32, 0x90
INSTRLEN_12_OFFSET_9:
  .fill 9, 1, 0x90
  .bundle_lock
  .rept 12
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1729: nop
# CHECK: 1730: incl

  .align 32, 0x90
INSTRLEN_12_OFFSET_10:
  .fill 10, 1, 0x90
  .bundle_lock
  .rept 12
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 174a: nop
# CHECK: 1750: incl

  .align 32, 0x90
INSTRLEN_12_OFFSET_11:
  .fill 11, 1, 0x90
  .bundle_lock
  .rept 12
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 176b: nop
# CHECK: 1770: incl

  .align 32, 0x90
INSTRLEN_12_OFFSET_12:
  .fill 12, 1, 0x90
  .bundle_lock
  .rept 12
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 178c: nop
# CHECK: 1790: incl

  .align 32, 0x90
INSTRLEN_12_OFFSET_13:
  .fill 13, 1, 0x90
  .bundle_lock
  .rept 12
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 17ad: nop
# CHECK: 17b0: incl

  .align 32, 0x90
INSTRLEN_12_OFFSET_14:
  .fill 14, 1, 0x90
  .bundle_lock
  .rept 12
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 17ce: nop
# CHECK: 17d0: incl

  .align 32, 0x90
INSTRLEN_12_OFFSET_15:
  .fill 15, 1, 0x90
  .bundle_lock
  .rept 12
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 17ef: nop
# CHECK: 17f0: incl

  .align 32, 0x90
INSTRLEN_13_OFFSET_0:
  .bundle_lock
  .rept 13
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1800: incl

  .align 32, 0x90
INSTRLEN_13_OFFSET_1:
  .fill 1, 1, 0x90
  .bundle_lock
  .rept 13
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1821: incl

  .align 32, 0x90
INSTRLEN_13_OFFSET_2:
  .fill 2, 1, 0x90
  .bundle_lock
  .rept 13
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1842: incl

  .align 32, 0x90
INSTRLEN_13_OFFSET_3:
  .fill 3, 1, 0x90
  .bundle_lock
  .rept 13
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1863: incl

  .align 32, 0x90
INSTRLEN_13_OFFSET_4:
  .fill 4, 1, 0x90
  .bundle_lock
  .rept 13
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1884: nop
# CHECK: 1890: incl

  .align 32, 0x90
INSTRLEN_13_OFFSET_5:
  .fill 5, 1, 0x90
  .bundle_lock
  .rept 13
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 18a5: nop
# CHECK: 18b0: incl

  .align 32, 0x90
INSTRLEN_13_OFFSET_6:
  .fill 6, 1, 0x90
  .bundle_lock
  .rept 13
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 18c6: nop
# CHECK: 18d0: incl

  .align 32, 0x90
INSTRLEN_13_OFFSET_7:
  .fill 7, 1, 0x90
  .bundle_lock
  .rept 13
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 18e7: nop
# CHECK: 18f0: incl

  .align 32, 0x90
INSTRLEN_13_OFFSET_8:
  .fill 8, 1, 0x90
  .bundle_lock
  .rept 13
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1908: nop
# CHECK: 1910: incl

  .align 32, 0x90
INSTRLEN_13_OFFSET_9:
  .fill 9, 1, 0x90
  .bundle_lock
  .rept 13
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1929: nop
# CHECK: 1930: incl

  .align 32, 0x90
INSTRLEN_13_OFFSET_10:
  .fill 10, 1, 0x90
  .bundle_lock
  .rept 13
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 194a: nop
# CHECK: 1950: incl

  .align 32, 0x90
INSTRLEN_13_OFFSET_11:
  .fill 11, 1, 0x90
  .bundle_lock
  .rept 13
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 196b: nop
# CHECK: 1970: incl

  .align 32, 0x90
INSTRLEN_13_OFFSET_12:
  .fill 12, 1, 0x90
  .bundle_lock
  .rept 13
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 198c: nop
# CHECK: 1990: incl

  .align 32, 0x90
INSTRLEN_13_OFFSET_13:
  .fill 13, 1, 0x90
  .bundle_lock
  .rept 13
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 19ad: nop
# CHECK: 19b0: incl

  .align 32, 0x90
INSTRLEN_13_OFFSET_14:
  .fill 14, 1, 0x90
  .bundle_lock
  .rept 13
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 19ce: nop
# CHECK: 19d0: incl

  .align 32, 0x90
INSTRLEN_13_OFFSET_15:
  .fill 15, 1, 0x90
  .bundle_lock
  .rept 13
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 19ef: nop
# CHECK: 19f0: incl

  .align 32, 0x90
INSTRLEN_14_OFFSET_0:
  .bundle_lock
  .rept 14
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1a00: incl

  .align 32, 0x90
INSTRLEN_14_OFFSET_1:
  .fill 1, 1, 0x90
  .bundle_lock
  .rept 14
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1a21: incl

  .align 32, 0x90
INSTRLEN_14_OFFSET_2:
  .fill 2, 1, 0x90
  .bundle_lock
  .rept 14
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1a42: incl

  .align 32, 0x90
INSTRLEN_14_OFFSET_3:
  .fill 3, 1, 0x90
  .bundle_lock
  .rept 14
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1a63: nop
# CHECK: 1a70: incl

  .align 32, 0x90
INSTRLEN_14_OFFSET_4:
  .fill 4, 1, 0x90
  .bundle_lock
  .rept 14
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1a84: nop
# CHECK: 1a90: incl

  .align 32, 0x90
INSTRLEN_14_OFFSET_5:
  .fill 5, 1, 0x90
  .bundle_lock
  .rept 14
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1aa5: nop
# CHECK: 1ab0: incl

  .align 32, 0x90
INSTRLEN_14_OFFSET_6:
  .fill 6, 1, 0x90
  .bundle_lock
  .rept 14
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1ac6: nop
# CHECK: 1ad0: incl

  .align 32, 0x90
INSTRLEN_14_OFFSET_7:
  .fill 7, 1, 0x90
  .bundle_lock
  .rept 14
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1ae7: nop
# CHECK: 1af0: incl

  .align 32, 0x90
INSTRLEN_14_OFFSET_8:
  .fill 8, 1, 0x90
  .bundle_lock
  .rept 14
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1b08: nop
# CHECK: 1b10: incl

  .align 32, 0x90
INSTRLEN_14_OFFSET_9:
  .fill 9, 1, 0x90
  .bundle_lock
  .rept 14
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1b29: nop
# CHECK: 1b30: incl

  .align 32, 0x90
INSTRLEN_14_OFFSET_10:
  .fill 10, 1, 0x90
  .bundle_lock
  .rept 14
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1b4a: nop
# CHECK: 1b50: incl

  .align 32, 0x90
INSTRLEN_14_OFFSET_11:
  .fill 11, 1, 0x90
  .bundle_lock
  .rept 14
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1b6b: nop
# CHECK: 1b70: incl

  .align 32, 0x90
INSTRLEN_14_OFFSET_12:
  .fill 12, 1, 0x90
  .bundle_lock
  .rept 14
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1b8c: nop
# CHECK: 1b90: incl

  .align 32, 0x90
INSTRLEN_14_OFFSET_13:
  .fill 13, 1, 0x90
  .bundle_lock
  .rept 14
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1bad: nop
# CHECK: 1bb0: incl

  .align 32, 0x90
INSTRLEN_14_OFFSET_14:
  .fill 14, 1, 0x90
  .bundle_lock
  .rept 14
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1bce: nop
# CHECK: 1bd0: incl

  .align 32, 0x90
INSTRLEN_14_OFFSET_15:
  .fill 15, 1, 0x90
  .bundle_lock
  .rept 14
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1bef: nop
# CHECK: 1bf0: incl

  .align 32, 0x90
INSTRLEN_15_OFFSET_0:
  .bundle_lock
  .rept 15
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1c00: incl

  .align 32, 0x90
INSTRLEN_15_OFFSET_1:
  .fill 1, 1, 0x90
  .bundle_lock
  .rept 15
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1c21: incl

  .align 32, 0x90
INSTRLEN_15_OFFSET_2:
  .fill 2, 1, 0x90
  .bundle_lock
  .rept 15
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1c42: nop
# CHECK: 1c50: incl

  .align 32, 0x90
INSTRLEN_15_OFFSET_3:
  .fill 3, 1, 0x90
  .bundle_lock
  .rept 15
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1c63: nop
# CHECK: 1c70: incl

  .align 32, 0x90
INSTRLEN_15_OFFSET_4:
  .fill 4, 1, 0x90
  .bundle_lock
  .rept 15
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1c84: nop
# CHECK: 1c90: incl

  .align 32, 0x90
INSTRLEN_15_OFFSET_5:
  .fill 5, 1, 0x90
  .bundle_lock
  .rept 15
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1ca5: nop
# CHECK: 1cb0: incl

  .align 32, 0x90
INSTRLEN_15_OFFSET_6:
  .fill 6, 1, 0x90
  .bundle_lock
  .rept 15
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1cc6: nop
# CHECK: 1cd0: incl

  .align 32, 0x90
INSTRLEN_15_OFFSET_7:
  .fill 7, 1, 0x90
  .bundle_lock
  .rept 15
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1ce7: nop
# CHECK: 1cf0: incl

  .align 32, 0x90
INSTRLEN_15_OFFSET_8:
  .fill 8, 1, 0x90
  .bundle_lock
  .rept 15
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1d08: nop
# CHECK: 1d10: incl

  .align 32, 0x90
INSTRLEN_15_OFFSET_9:
  .fill 9, 1, 0x90
  .bundle_lock
  .rept 15
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1d29: nop
# CHECK: 1d30: incl

  .align 32, 0x90
INSTRLEN_15_OFFSET_10:
  .fill 10, 1, 0x90
  .bundle_lock
  .rept 15
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1d4a: nop
# CHECK: 1d50: incl

  .align 32, 0x90
INSTRLEN_15_OFFSET_11:
  .fill 11, 1, 0x90
  .bundle_lock
  .rept 15
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1d6b: nop
# CHECK: 1d70: incl

  .align 32, 0x90
INSTRLEN_15_OFFSET_12:
  .fill 12, 1, 0x90
  .bundle_lock
  .rept 15
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1d8c: nop
# CHECK: 1d90: incl

  .align 32, 0x90
INSTRLEN_15_OFFSET_13:
  .fill 13, 1, 0x90
  .bundle_lock
  .rept 15
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1dad: nop
# CHECK: 1db0: incl

  .align 32, 0x90
INSTRLEN_15_OFFSET_14:
  .fill 14, 1, 0x90
  .bundle_lock
  .rept 15
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1dce: nop
# CHECK: 1dd0: incl

  .align 32, 0x90
INSTRLEN_15_OFFSET_15:
  .fill 15, 1, 0x90
  .bundle_lock
  .rept 15
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1def: nop
# CHECK: 1df0: incl

  .align 32, 0x90
INSTRLEN_16_OFFSET_0:
  .bundle_lock
  .rept 16
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1e00: incl

  .align 32, 0x90
INSTRLEN_16_OFFSET_1:
  .fill 1, 1, 0x90
  .bundle_lock
  .rept 16
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1e21: nop
# CHECK: 1e30: incl

  .align 32, 0x90
INSTRLEN_16_OFFSET_2:
  .fill 2, 1, 0x90
  .bundle_lock
  .rept 16
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1e42: nop
# CHECK: 1e50: incl

  .align 32, 0x90
INSTRLEN_16_OFFSET_3:
  .fill 3, 1, 0x90
  .bundle_lock
  .rept 16
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1e63: nop
# CHECK: 1e70: incl

  .align 32, 0x90
INSTRLEN_16_OFFSET_4:
  .fill 4, 1, 0x90
  .bundle_lock
  .rept 16
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1e84: nop
# CHECK: 1e90: incl

  .align 32, 0x90
INSTRLEN_16_OFFSET_5:
  .fill 5, 1, 0x90
  .bundle_lock
  .rept 16
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1ea5: nop
# CHECK: 1eb0: incl

  .align 32, 0x90
INSTRLEN_16_OFFSET_6:
  .fill 6, 1, 0x90
  .bundle_lock
  .rept 16
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1ec6: nop
# CHECK: 1ed0: incl

  .align 32, 0x90
INSTRLEN_16_OFFSET_7:
  .fill 7, 1, 0x90
  .bundle_lock
  .rept 16
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1ee7: nop
# CHECK: 1ef0: incl

  .align 32, 0x90
INSTRLEN_16_OFFSET_8:
  .fill 8, 1, 0x90
  .bundle_lock
  .rept 16
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1f08: nop
# CHECK: 1f10: incl

  .align 32, 0x90
INSTRLEN_16_OFFSET_9:
  .fill 9, 1, 0x90
  .bundle_lock
  .rept 16
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1f29: nop
# CHECK: 1f30: incl

  .align 32, 0x90
INSTRLEN_16_OFFSET_10:
  .fill 10, 1, 0x90
  .bundle_lock
  .rept 16
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1f4a: nop
# CHECK: 1f50: incl

  .align 32, 0x90
INSTRLEN_16_OFFSET_11:
  .fill 11, 1, 0x90
  .bundle_lock
  .rept 16
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1f6b: nop
# CHECK: 1f70: incl

  .align 32, 0x90
INSTRLEN_16_OFFSET_12:
  .fill 12, 1, 0x90
  .bundle_lock
  .rept 16
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1f8c: nop
# CHECK: 1f90: incl

  .align 32, 0x90
INSTRLEN_16_OFFSET_13:
  .fill 13, 1, 0x90
  .bundle_lock
  .rept 16
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1fad: nop
# CHECK: 1fb0: incl

  .align 32, 0x90
INSTRLEN_16_OFFSET_14:
  .fill 14, 1, 0x90
  .bundle_lock
  .rept 16
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1fce: nop
# CHECK: 1fd0: incl

  .align 32, 0x90
INSTRLEN_16_OFFSET_15:
  .fill 15, 1, 0x90
  .bundle_lock
  .rept 16
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1fef: nop
# CHECK: 1ff0: incl

