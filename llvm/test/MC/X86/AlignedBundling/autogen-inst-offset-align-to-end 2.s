# RUN: llvm-mc -filetype=obj -triple i386-pc-linux-gnu %s -o - \
# RUN:   | llvm-objdump --triple=i386 -d --no-show-raw-insn - | FileCheck %s

# !!! This test is auto-generated from utils/testgen/mc-bundling-x86-gen.py !!!
#     It tests that bundle-aligned grouping works correctly in MC. Read the
#     source of the script for more details.

  .text
  .bundle_align_mode 4

  .align 32, 0x90
INSTRLEN_1_OFFSET_0:
  .bundle_lock align_to_end
  .rept 1
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 0: nop
# CHECK: f: incl

  .align 32, 0x90
INSTRLEN_1_OFFSET_1:
  .fill 1, 1, 0x90
  .bundle_lock align_to_end
  .rept 1
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 21: nop
# CHECK: 2f: incl

  .align 32, 0x90
INSTRLEN_1_OFFSET_2:
  .fill 2, 1, 0x90
  .bundle_lock align_to_end
  .rept 1
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 42: nop
# CHECK: 4f: incl

  .align 32, 0x90
INSTRLEN_1_OFFSET_3:
  .fill 3, 1, 0x90
  .bundle_lock align_to_end
  .rept 1
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 63: nop
# CHECK: 6f: incl

  .align 32, 0x90
INSTRLEN_1_OFFSET_4:
  .fill 4, 1, 0x90
  .bundle_lock align_to_end
  .rept 1
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 84: nop
# CHECK: 8f: incl

  .align 32, 0x90
INSTRLEN_1_OFFSET_5:
  .fill 5, 1, 0x90
  .bundle_lock align_to_end
  .rept 1
  inc %eax
  .endr
  .bundle_unlock
# CHECK: a5: nop
# CHECK: af: incl

  .align 32, 0x90
INSTRLEN_1_OFFSET_6:
  .fill 6, 1, 0x90
  .bundle_lock align_to_end
  .rept 1
  inc %eax
  .endr
  .bundle_unlock
# CHECK: c6: nop
# CHECK: cf: incl

  .align 32, 0x90
INSTRLEN_1_OFFSET_7:
  .fill 7, 1, 0x90
  .bundle_lock align_to_end
  .rept 1
  inc %eax
  .endr
  .bundle_unlock
# CHECK: e7: nop
# CHECK: ef: incl

  .align 32, 0x90
INSTRLEN_1_OFFSET_8:
  .fill 8, 1, 0x90
  .bundle_lock align_to_end
  .rept 1
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 108: nop
# CHECK: 10f: incl

  .align 32, 0x90
INSTRLEN_1_OFFSET_9:
  .fill 9, 1, 0x90
  .bundle_lock align_to_end
  .rept 1
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 129: nop
# CHECK: 12f: incl

  .align 32, 0x90
INSTRLEN_1_OFFSET_10:
  .fill 10, 1, 0x90
  .bundle_lock align_to_end
  .rept 1
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 14a: nop
# CHECK: 14f: incl

  .align 32, 0x90
INSTRLEN_1_OFFSET_11:
  .fill 11, 1, 0x90
  .bundle_lock align_to_end
  .rept 1
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 16b: nop
# CHECK: 16f: incl

  .align 32, 0x90
INSTRLEN_1_OFFSET_12:
  .fill 12, 1, 0x90
  .bundle_lock align_to_end
  .rept 1
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 18c: nop
# CHECK: 18f: incl

  .align 32, 0x90
INSTRLEN_1_OFFSET_13:
  .fill 13, 1, 0x90
  .bundle_lock align_to_end
  .rept 1
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1ad: nop
# CHECK: 1af: incl

  .align 32, 0x90
INSTRLEN_1_OFFSET_14:
  .fill 14, 1, 0x90
  .bundle_lock align_to_end
  .rept 1
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1ce: nop
# CHECK: 1cf: incl

  .align 32, 0x90
INSTRLEN_1_OFFSET_15:
  .fill 15, 1, 0x90
  .bundle_lock align_to_end
  .rept 1
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1ef: incl

  .align 32, 0x90
INSTRLEN_2_OFFSET_0:
  .bundle_lock align_to_end
  .rept 2
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 200: nop
# CHECK: 20e: incl

  .align 32, 0x90
INSTRLEN_2_OFFSET_1:
  .fill 1, 1, 0x90
  .bundle_lock align_to_end
  .rept 2
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 221: nop
# CHECK: 22e: incl

  .align 32, 0x90
INSTRLEN_2_OFFSET_2:
  .fill 2, 1, 0x90
  .bundle_lock align_to_end
  .rept 2
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 242: nop
# CHECK: 24e: incl

  .align 32, 0x90
INSTRLEN_2_OFFSET_3:
  .fill 3, 1, 0x90
  .bundle_lock align_to_end
  .rept 2
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 263: nop
# CHECK: 26e: incl

  .align 32, 0x90
INSTRLEN_2_OFFSET_4:
  .fill 4, 1, 0x90
  .bundle_lock align_to_end
  .rept 2
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 284: nop
# CHECK: 28e: incl

  .align 32, 0x90
INSTRLEN_2_OFFSET_5:
  .fill 5, 1, 0x90
  .bundle_lock align_to_end
  .rept 2
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 2a5: nop
# CHECK: 2ae: incl

  .align 32, 0x90
INSTRLEN_2_OFFSET_6:
  .fill 6, 1, 0x90
  .bundle_lock align_to_end
  .rept 2
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 2c6: nop
# CHECK: 2ce: incl

  .align 32, 0x90
INSTRLEN_2_OFFSET_7:
  .fill 7, 1, 0x90
  .bundle_lock align_to_end
  .rept 2
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 2e7: nop
# CHECK: 2ee: incl

  .align 32, 0x90
INSTRLEN_2_OFFSET_8:
  .fill 8, 1, 0x90
  .bundle_lock align_to_end
  .rept 2
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 308: nop
# CHECK: 30e: incl

  .align 32, 0x90
INSTRLEN_2_OFFSET_9:
  .fill 9, 1, 0x90
  .bundle_lock align_to_end
  .rept 2
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 329: nop
# CHECK: 32e: incl

  .align 32, 0x90
INSTRLEN_2_OFFSET_10:
  .fill 10, 1, 0x90
  .bundle_lock align_to_end
  .rept 2
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 34a: nop
# CHECK: 34e: incl

  .align 32, 0x90
INSTRLEN_2_OFFSET_11:
  .fill 11, 1, 0x90
  .bundle_lock align_to_end
  .rept 2
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 36b: nop
# CHECK: 36e: incl

  .align 32, 0x90
INSTRLEN_2_OFFSET_12:
  .fill 12, 1, 0x90
  .bundle_lock align_to_end
  .rept 2
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 38c: nop
# CHECK: 38e: incl

  .align 32, 0x90
INSTRLEN_2_OFFSET_13:
  .fill 13, 1, 0x90
  .bundle_lock align_to_end
  .rept 2
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 3ad: nop
# CHECK: 3ae: incl

  .align 32, 0x90
INSTRLEN_2_OFFSET_14:
  .fill 14, 1, 0x90
  .bundle_lock align_to_end
  .rept 2
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 3ce: incl

  .align 32, 0x90
INSTRLEN_2_OFFSET_15:
  .fill 15, 1, 0x90
  .bundle_lock align_to_end
  .rept 2
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 3ef: nop
# CHECK: 3f0: nop
# CHECK: 3fe: incl

  .align 32, 0x90
INSTRLEN_3_OFFSET_0:
  .bundle_lock align_to_end
  .rept 3
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 400: nop
# CHECK: 40d: incl

  .align 32, 0x90
INSTRLEN_3_OFFSET_1:
  .fill 1, 1, 0x90
  .bundle_lock align_to_end
  .rept 3
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 421: nop
# CHECK: 42d: incl

  .align 32, 0x90
INSTRLEN_3_OFFSET_2:
  .fill 2, 1, 0x90
  .bundle_lock align_to_end
  .rept 3
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 442: nop
# CHECK: 44d: incl

  .align 32, 0x90
INSTRLEN_3_OFFSET_3:
  .fill 3, 1, 0x90
  .bundle_lock align_to_end
  .rept 3
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 463: nop
# CHECK: 46d: incl

  .align 32, 0x90
INSTRLEN_3_OFFSET_4:
  .fill 4, 1, 0x90
  .bundle_lock align_to_end
  .rept 3
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 484: nop
# CHECK: 48d: incl

  .align 32, 0x90
INSTRLEN_3_OFFSET_5:
  .fill 5, 1, 0x90
  .bundle_lock align_to_end
  .rept 3
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 4a5: nop
# CHECK: 4ad: incl

  .align 32, 0x90
INSTRLEN_3_OFFSET_6:
  .fill 6, 1, 0x90
  .bundle_lock align_to_end
  .rept 3
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 4c6: nop
# CHECK: 4cd: incl

  .align 32, 0x90
INSTRLEN_3_OFFSET_7:
  .fill 7, 1, 0x90
  .bundle_lock align_to_end
  .rept 3
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 4e7: nop
# CHECK: 4ed: incl

  .align 32, 0x90
INSTRLEN_3_OFFSET_8:
  .fill 8, 1, 0x90
  .bundle_lock align_to_end
  .rept 3
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 508: nop
# CHECK: 50d: incl

  .align 32, 0x90
INSTRLEN_3_OFFSET_9:
  .fill 9, 1, 0x90
  .bundle_lock align_to_end
  .rept 3
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 529: nop
# CHECK: 52d: incl

  .align 32, 0x90
INSTRLEN_3_OFFSET_10:
  .fill 10, 1, 0x90
  .bundle_lock align_to_end
  .rept 3
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 54a: nop
# CHECK: 54d: incl

  .align 32, 0x90
INSTRLEN_3_OFFSET_11:
  .fill 11, 1, 0x90
  .bundle_lock align_to_end
  .rept 3
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 56b: nop
# CHECK: 56d: incl

  .align 32, 0x90
INSTRLEN_3_OFFSET_12:
  .fill 12, 1, 0x90
  .bundle_lock align_to_end
  .rept 3
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 58c: nop
# CHECK: 58d: incl

  .align 32, 0x90
INSTRLEN_3_OFFSET_13:
  .fill 13, 1, 0x90
  .bundle_lock align_to_end
  .rept 3
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 5ad: incl

  .align 32, 0x90
INSTRLEN_3_OFFSET_14:
  .fill 14, 1, 0x90
  .bundle_lock align_to_end
  .rept 3
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 5ce: nop
# CHECK: 5d0: nop
# CHECK: 5dd: incl

  .align 32, 0x90
INSTRLEN_3_OFFSET_15:
  .fill 15, 1, 0x90
  .bundle_lock align_to_end
  .rept 3
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 5ef: nop
# CHECK: 5f0: nop
# CHECK: 5fd: incl

  .align 32, 0x90
INSTRLEN_4_OFFSET_0:
  .bundle_lock align_to_end
  .rept 4
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 600: nop
# CHECK: 60c: incl

  .align 32, 0x90
INSTRLEN_4_OFFSET_1:
  .fill 1, 1, 0x90
  .bundle_lock align_to_end
  .rept 4
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 621: nop
# CHECK: 62c: incl

  .align 32, 0x90
INSTRLEN_4_OFFSET_2:
  .fill 2, 1, 0x90
  .bundle_lock align_to_end
  .rept 4
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 642: nop
# CHECK: 64c: incl

  .align 32, 0x90
INSTRLEN_4_OFFSET_3:
  .fill 3, 1, 0x90
  .bundle_lock align_to_end
  .rept 4
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 663: nop
# CHECK: 66c: incl

  .align 32, 0x90
INSTRLEN_4_OFFSET_4:
  .fill 4, 1, 0x90
  .bundle_lock align_to_end
  .rept 4
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 684: nop
# CHECK: 68c: incl

  .align 32, 0x90
INSTRLEN_4_OFFSET_5:
  .fill 5, 1, 0x90
  .bundle_lock align_to_end
  .rept 4
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 6a5: nop
# CHECK: 6ac: incl

  .align 32, 0x90
INSTRLEN_4_OFFSET_6:
  .fill 6, 1, 0x90
  .bundle_lock align_to_end
  .rept 4
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 6c6: nop
# CHECK: 6cc: incl

  .align 32, 0x90
INSTRLEN_4_OFFSET_7:
  .fill 7, 1, 0x90
  .bundle_lock align_to_end
  .rept 4
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 6e7: nop
# CHECK: 6ec: incl

  .align 32, 0x90
INSTRLEN_4_OFFSET_8:
  .fill 8, 1, 0x90
  .bundle_lock align_to_end
  .rept 4
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 708: nop
# CHECK: 70c: incl

  .align 32, 0x90
INSTRLEN_4_OFFSET_9:
  .fill 9, 1, 0x90
  .bundle_lock align_to_end
  .rept 4
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 729: nop
# CHECK: 72c: incl

  .align 32, 0x90
INSTRLEN_4_OFFSET_10:
  .fill 10, 1, 0x90
  .bundle_lock align_to_end
  .rept 4
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 74a: nop
# CHECK: 74c: incl

  .align 32, 0x90
INSTRLEN_4_OFFSET_11:
  .fill 11, 1, 0x90
  .bundle_lock align_to_end
  .rept 4
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 76b: nop
# CHECK: 76c: incl

  .align 32, 0x90
INSTRLEN_4_OFFSET_12:
  .fill 12, 1, 0x90
  .bundle_lock align_to_end
  .rept 4
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 78c: incl

  .align 32, 0x90
INSTRLEN_4_OFFSET_13:
  .fill 13, 1, 0x90
  .bundle_lock align_to_end
  .rept 4
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 7ad: nop
# CHECK: 7b0: nop
# CHECK: 7bc: incl

  .align 32, 0x90
INSTRLEN_4_OFFSET_14:
  .fill 14, 1, 0x90
  .bundle_lock align_to_end
  .rept 4
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 7ce: nop
# CHECK: 7d0: nop
# CHECK: 7dc: incl

  .align 32, 0x90
INSTRLEN_4_OFFSET_15:
  .fill 15, 1, 0x90
  .bundle_lock align_to_end
  .rept 4
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 7ef: nop
# CHECK: 7f0: nop
# CHECK: 7fc: incl

  .align 32, 0x90
INSTRLEN_5_OFFSET_0:
  .bundle_lock align_to_end
  .rept 5
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 800: nop
# CHECK: 80b: incl

  .align 32, 0x90
INSTRLEN_5_OFFSET_1:
  .fill 1, 1, 0x90
  .bundle_lock align_to_end
  .rept 5
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 821: nop
# CHECK: 82b: incl

  .align 32, 0x90
INSTRLEN_5_OFFSET_2:
  .fill 2, 1, 0x90
  .bundle_lock align_to_end
  .rept 5
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 842: nop
# CHECK: 84b: incl

  .align 32, 0x90
INSTRLEN_5_OFFSET_3:
  .fill 3, 1, 0x90
  .bundle_lock align_to_end
  .rept 5
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 863: nop
# CHECK: 86b: incl

  .align 32, 0x90
INSTRLEN_5_OFFSET_4:
  .fill 4, 1, 0x90
  .bundle_lock align_to_end
  .rept 5
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 884: nop
# CHECK: 88b: incl

  .align 32, 0x90
INSTRLEN_5_OFFSET_5:
  .fill 5, 1, 0x90
  .bundle_lock align_to_end
  .rept 5
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 8a5: nop
# CHECK: 8ab: incl

  .align 32, 0x90
INSTRLEN_5_OFFSET_6:
  .fill 6, 1, 0x90
  .bundle_lock align_to_end
  .rept 5
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 8c6: nop
# CHECK: 8cb: incl

  .align 32, 0x90
INSTRLEN_5_OFFSET_7:
  .fill 7, 1, 0x90
  .bundle_lock align_to_end
  .rept 5
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 8e7: nop
# CHECK: 8eb: incl

  .align 32, 0x90
INSTRLEN_5_OFFSET_8:
  .fill 8, 1, 0x90
  .bundle_lock align_to_end
  .rept 5
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 908: nop
# CHECK: 90b: incl

  .align 32, 0x90
INSTRLEN_5_OFFSET_9:
  .fill 9, 1, 0x90
  .bundle_lock align_to_end
  .rept 5
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 929: nop
# CHECK: 92b: incl

  .align 32, 0x90
INSTRLEN_5_OFFSET_10:
  .fill 10, 1, 0x90
  .bundle_lock align_to_end
  .rept 5
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 94a: nop
# CHECK: 94b: incl

  .align 32, 0x90
INSTRLEN_5_OFFSET_11:
  .fill 11, 1, 0x90
  .bundle_lock align_to_end
  .rept 5
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 96b: incl

  .align 32, 0x90
INSTRLEN_5_OFFSET_12:
  .fill 12, 1, 0x90
  .bundle_lock align_to_end
  .rept 5
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 98c: nop
# CHECK: 990: nop
# CHECK: 99b: incl

  .align 32, 0x90
INSTRLEN_5_OFFSET_13:
  .fill 13, 1, 0x90
  .bundle_lock align_to_end
  .rept 5
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 9ad: nop
# CHECK: 9b0: nop
# CHECK: 9bb: incl

  .align 32, 0x90
INSTRLEN_5_OFFSET_14:
  .fill 14, 1, 0x90
  .bundle_lock align_to_end
  .rept 5
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 9ce: nop
# CHECK: 9d0: nop
# CHECK: 9db: incl

  .align 32, 0x90
INSTRLEN_5_OFFSET_15:
  .fill 15, 1, 0x90
  .bundle_lock align_to_end
  .rept 5
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 9ef: nop
# CHECK: 9f0: nop
# CHECK: 9fb: incl

  .align 32, 0x90
INSTRLEN_6_OFFSET_0:
  .bundle_lock align_to_end
  .rept 6
  inc %eax
  .endr
  .bundle_unlock
# CHECK: a00: nop
# CHECK: a0a: incl

  .align 32, 0x90
INSTRLEN_6_OFFSET_1:
  .fill 1, 1, 0x90
  .bundle_lock align_to_end
  .rept 6
  inc %eax
  .endr
  .bundle_unlock
# CHECK: a21: nop
# CHECK: a2a: incl

  .align 32, 0x90
INSTRLEN_6_OFFSET_2:
  .fill 2, 1, 0x90
  .bundle_lock align_to_end
  .rept 6
  inc %eax
  .endr
  .bundle_unlock
# CHECK: a42: nop
# CHECK: a4a: incl

  .align 32, 0x90
INSTRLEN_6_OFFSET_3:
  .fill 3, 1, 0x90
  .bundle_lock align_to_end
  .rept 6
  inc %eax
  .endr
  .bundle_unlock
# CHECK: a63: nop
# CHECK: a6a: incl

  .align 32, 0x90
INSTRLEN_6_OFFSET_4:
  .fill 4, 1, 0x90
  .bundle_lock align_to_end
  .rept 6
  inc %eax
  .endr
  .bundle_unlock
# CHECK: a84: nop
# CHECK: a8a: incl

  .align 32, 0x90
INSTRLEN_6_OFFSET_5:
  .fill 5, 1, 0x90
  .bundle_lock align_to_end
  .rept 6
  inc %eax
  .endr
  .bundle_unlock
# CHECK: aa5: nop
# CHECK: aaa: incl

  .align 32, 0x90
INSTRLEN_6_OFFSET_6:
  .fill 6, 1, 0x90
  .bundle_lock align_to_end
  .rept 6
  inc %eax
  .endr
  .bundle_unlock
# CHECK: ac6: nop
# CHECK: aca: incl

  .align 32, 0x90
INSTRLEN_6_OFFSET_7:
  .fill 7, 1, 0x90
  .bundle_lock align_to_end
  .rept 6
  inc %eax
  .endr
  .bundle_unlock
# CHECK: ae7: nop
# CHECK: aea: incl

  .align 32, 0x90
INSTRLEN_6_OFFSET_8:
  .fill 8, 1, 0x90
  .bundle_lock align_to_end
  .rept 6
  inc %eax
  .endr
  .bundle_unlock
# CHECK: b08: nop
# CHECK: b0a: incl

  .align 32, 0x90
INSTRLEN_6_OFFSET_9:
  .fill 9, 1, 0x90
  .bundle_lock align_to_end
  .rept 6
  inc %eax
  .endr
  .bundle_unlock
# CHECK: b29: nop
# CHECK: b2a: incl

  .align 32, 0x90
INSTRLEN_6_OFFSET_10:
  .fill 10, 1, 0x90
  .bundle_lock align_to_end
  .rept 6
  inc %eax
  .endr
  .bundle_unlock
# CHECK: b4a: incl

  .align 32, 0x90
INSTRLEN_6_OFFSET_11:
  .fill 11, 1, 0x90
  .bundle_lock align_to_end
  .rept 6
  inc %eax
  .endr
  .bundle_unlock
# CHECK: b6b: nop
# CHECK: b70: nop
# CHECK: b7a: incl

  .align 32, 0x90
INSTRLEN_6_OFFSET_12:
  .fill 12, 1, 0x90
  .bundle_lock align_to_end
  .rept 6
  inc %eax
  .endr
  .bundle_unlock
# CHECK: b8c: nop
# CHECK: b90: nop
# CHECK: b9a: incl

  .align 32, 0x90
INSTRLEN_6_OFFSET_13:
  .fill 13, 1, 0x90
  .bundle_lock align_to_end
  .rept 6
  inc %eax
  .endr
  .bundle_unlock
# CHECK: bad: nop
# CHECK: bb0: nop
# CHECK: bba: incl

  .align 32, 0x90
INSTRLEN_6_OFFSET_14:
  .fill 14, 1, 0x90
  .bundle_lock align_to_end
  .rept 6
  inc %eax
  .endr
  .bundle_unlock
# CHECK: bce: nop
# CHECK: bd0: nop
# CHECK: bda: incl

  .align 32, 0x90
INSTRLEN_6_OFFSET_15:
  .fill 15, 1, 0x90
  .bundle_lock align_to_end
  .rept 6
  inc %eax
  .endr
  .bundle_unlock
# CHECK: bef: nop
# CHECK: bf0: nop
# CHECK: bfa: incl

  .align 32, 0x90
INSTRLEN_7_OFFSET_0:
  .bundle_lock align_to_end
  .rept 7
  inc %eax
  .endr
  .bundle_unlock
# CHECK: c00: nop
# CHECK: c09: incl

  .align 32, 0x90
INSTRLEN_7_OFFSET_1:
  .fill 1, 1, 0x90
  .bundle_lock align_to_end
  .rept 7
  inc %eax
  .endr
  .bundle_unlock
# CHECK: c21: nop
# CHECK: c29: incl

  .align 32, 0x90
INSTRLEN_7_OFFSET_2:
  .fill 2, 1, 0x90
  .bundle_lock align_to_end
  .rept 7
  inc %eax
  .endr
  .bundle_unlock
# CHECK: c42: nop
# CHECK: c49: incl

  .align 32, 0x90
INSTRLEN_7_OFFSET_3:
  .fill 3, 1, 0x90
  .bundle_lock align_to_end
  .rept 7
  inc %eax
  .endr
  .bundle_unlock
# CHECK: c63: nop
# CHECK: c69: incl

  .align 32, 0x90
INSTRLEN_7_OFFSET_4:
  .fill 4, 1, 0x90
  .bundle_lock align_to_end
  .rept 7
  inc %eax
  .endr
  .bundle_unlock
# CHECK: c84: nop
# CHECK: c89: incl

  .align 32, 0x90
INSTRLEN_7_OFFSET_5:
  .fill 5, 1, 0x90
  .bundle_lock align_to_end
  .rept 7
  inc %eax
  .endr
  .bundle_unlock
# CHECK: ca5: nop
# CHECK: ca9: incl

  .align 32, 0x90
INSTRLEN_7_OFFSET_6:
  .fill 6, 1, 0x90
  .bundle_lock align_to_end
  .rept 7
  inc %eax
  .endr
  .bundle_unlock
# CHECK: cc6: nop
# CHECK: cc9: incl

  .align 32, 0x90
INSTRLEN_7_OFFSET_7:
  .fill 7, 1, 0x90
  .bundle_lock align_to_end
  .rept 7
  inc %eax
  .endr
  .bundle_unlock
# CHECK: ce7: nop
# CHECK: ce9: incl

  .align 32, 0x90
INSTRLEN_7_OFFSET_8:
  .fill 8, 1, 0x90
  .bundle_lock align_to_end
  .rept 7
  inc %eax
  .endr
  .bundle_unlock
# CHECK: d08: nop
# CHECK: d09: incl

  .align 32, 0x90
INSTRLEN_7_OFFSET_9:
  .fill 9, 1, 0x90
  .bundle_lock align_to_end
  .rept 7
  inc %eax
  .endr
  .bundle_unlock
# CHECK: d29: incl

  .align 32, 0x90
INSTRLEN_7_OFFSET_10:
  .fill 10, 1, 0x90
  .bundle_lock align_to_end
  .rept 7
  inc %eax
  .endr
  .bundle_unlock
# CHECK: d4a: nop
# CHECK: d50: nop
# CHECK: d59: incl

  .align 32, 0x90
INSTRLEN_7_OFFSET_11:
  .fill 11, 1, 0x90
  .bundle_lock align_to_end
  .rept 7
  inc %eax
  .endr
  .bundle_unlock
# CHECK: d6b: nop
# CHECK: d70: nop
# CHECK: d79: incl

  .align 32, 0x90
INSTRLEN_7_OFFSET_12:
  .fill 12, 1, 0x90
  .bundle_lock align_to_end
  .rept 7
  inc %eax
  .endr
  .bundle_unlock
# CHECK: d8c: nop
# CHECK: d90: nop
# CHECK: d99: incl

  .align 32, 0x90
INSTRLEN_7_OFFSET_13:
  .fill 13, 1, 0x90
  .bundle_lock align_to_end
  .rept 7
  inc %eax
  .endr
  .bundle_unlock
# CHECK: dad: nop
# CHECK: db0: nop
# CHECK: db9: incl

  .align 32, 0x90
INSTRLEN_7_OFFSET_14:
  .fill 14, 1, 0x90
  .bundle_lock align_to_end
  .rept 7
  inc %eax
  .endr
  .bundle_unlock
# CHECK: dce: nop
# CHECK: dd0: nop
# CHECK: dd9: incl

  .align 32, 0x90
INSTRLEN_7_OFFSET_15:
  .fill 15, 1, 0x90
  .bundle_lock align_to_end
  .rept 7
  inc %eax
  .endr
  .bundle_unlock
# CHECK: def: nop
# CHECK: df0: nop
# CHECK: df9: incl

  .align 32, 0x90
INSTRLEN_8_OFFSET_0:
  .bundle_lock align_to_end
  .rept 8
  inc %eax
  .endr
  .bundle_unlock
# CHECK: e00: nop
# CHECK: e08: incl

  .align 32, 0x90
INSTRLEN_8_OFFSET_1:
  .fill 1, 1, 0x90
  .bundle_lock align_to_end
  .rept 8
  inc %eax
  .endr
  .bundle_unlock
# CHECK: e21: nop
# CHECK: e28: incl

  .align 32, 0x90
INSTRLEN_8_OFFSET_2:
  .fill 2, 1, 0x90
  .bundle_lock align_to_end
  .rept 8
  inc %eax
  .endr
  .bundle_unlock
# CHECK: e42: nop
# CHECK: e48: incl

  .align 32, 0x90
INSTRLEN_8_OFFSET_3:
  .fill 3, 1, 0x90
  .bundle_lock align_to_end
  .rept 8
  inc %eax
  .endr
  .bundle_unlock
# CHECK: e63: nop
# CHECK: e68: incl

  .align 32, 0x90
INSTRLEN_8_OFFSET_4:
  .fill 4, 1, 0x90
  .bundle_lock align_to_end
  .rept 8
  inc %eax
  .endr
  .bundle_unlock
# CHECK: e84: nop
# CHECK: e88: incl

  .align 32, 0x90
INSTRLEN_8_OFFSET_5:
  .fill 5, 1, 0x90
  .bundle_lock align_to_end
  .rept 8
  inc %eax
  .endr
  .bundle_unlock
# CHECK: ea5: nop
# CHECK: ea8: incl

  .align 32, 0x90
INSTRLEN_8_OFFSET_6:
  .fill 6, 1, 0x90
  .bundle_lock align_to_end
  .rept 8
  inc %eax
  .endr
  .bundle_unlock
# CHECK: ec6: nop
# CHECK: ec8: incl

  .align 32, 0x90
INSTRLEN_8_OFFSET_7:
  .fill 7, 1, 0x90
  .bundle_lock align_to_end
  .rept 8
  inc %eax
  .endr
  .bundle_unlock
# CHECK: ee7: nop
# CHECK: ee8: incl

  .align 32, 0x90
INSTRLEN_8_OFFSET_8:
  .fill 8, 1, 0x90
  .bundle_lock align_to_end
  .rept 8
  inc %eax
  .endr
  .bundle_unlock
# CHECK: f08: incl

  .align 32, 0x90
INSTRLEN_8_OFFSET_9:
  .fill 9, 1, 0x90
  .bundle_lock align_to_end
  .rept 8
  inc %eax
  .endr
  .bundle_unlock
# CHECK: f29: nop
# CHECK: f30: nop
# CHECK: f38: incl

  .align 32, 0x90
INSTRLEN_8_OFFSET_10:
  .fill 10, 1, 0x90
  .bundle_lock align_to_end
  .rept 8
  inc %eax
  .endr
  .bundle_unlock
# CHECK: f4a: nop
# CHECK: f50: nop
# CHECK: f58: incl

  .align 32, 0x90
INSTRLEN_8_OFFSET_11:
  .fill 11, 1, 0x90
  .bundle_lock align_to_end
  .rept 8
  inc %eax
  .endr
  .bundle_unlock
# CHECK: f6b: nop
# CHECK: f70: nop
# CHECK: f78: incl

  .align 32, 0x90
INSTRLEN_8_OFFSET_12:
  .fill 12, 1, 0x90
  .bundle_lock align_to_end
  .rept 8
  inc %eax
  .endr
  .bundle_unlock
# CHECK: f8c: nop
# CHECK: f90: nop
# CHECK: f98: incl

  .align 32, 0x90
INSTRLEN_8_OFFSET_13:
  .fill 13, 1, 0x90
  .bundle_lock align_to_end
  .rept 8
  inc %eax
  .endr
  .bundle_unlock
# CHECK: fad: nop
# CHECK: fb0: nop
# CHECK: fb8: incl

  .align 32, 0x90
INSTRLEN_8_OFFSET_14:
  .fill 14, 1, 0x90
  .bundle_lock align_to_end
  .rept 8
  inc %eax
  .endr
  .bundle_unlock
# CHECK: fce: nop
# CHECK: fd0: nop
# CHECK: fd8: incl

  .align 32, 0x90
INSTRLEN_8_OFFSET_15:
  .fill 15, 1, 0x90
  .bundle_lock align_to_end
  .rept 8
  inc %eax
  .endr
  .bundle_unlock
# CHECK: fef: nop
# CHECK: ff0: nop
# CHECK: ff8: incl

  .align 32, 0x90
INSTRLEN_9_OFFSET_0:
  .bundle_lock align_to_end
  .rept 9
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1000: nop
# CHECK: 1007: incl

  .align 32, 0x90
INSTRLEN_9_OFFSET_1:
  .fill 1, 1, 0x90
  .bundle_lock align_to_end
  .rept 9
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1021: nop
# CHECK: 1027: incl

  .align 32, 0x90
INSTRLEN_9_OFFSET_2:
  .fill 2, 1, 0x90
  .bundle_lock align_to_end
  .rept 9
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1042: nop
# CHECK: 1047: incl

  .align 32, 0x90
INSTRLEN_9_OFFSET_3:
  .fill 3, 1, 0x90
  .bundle_lock align_to_end
  .rept 9
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1063: nop
# CHECK: 1067: incl

  .align 32, 0x90
INSTRLEN_9_OFFSET_4:
  .fill 4, 1, 0x90
  .bundle_lock align_to_end
  .rept 9
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1084: nop
# CHECK: 1087: incl

  .align 32, 0x90
INSTRLEN_9_OFFSET_5:
  .fill 5, 1, 0x90
  .bundle_lock align_to_end
  .rept 9
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 10a5: nop
# CHECK: 10a7: incl

  .align 32, 0x90
INSTRLEN_9_OFFSET_6:
  .fill 6, 1, 0x90
  .bundle_lock align_to_end
  .rept 9
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 10c6: nop
# CHECK: 10c7: incl

  .align 32, 0x90
INSTRLEN_9_OFFSET_7:
  .fill 7, 1, 0x90
  .bundle_lock align_to_end
  .rept 9
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 10e7: incl

  .align 32, 0x90
INSTRLEN_9_OFFSET_8:
  .fill 8, 1, 0x90
  .bundle_lock align_to_end
  .rept 9
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1108: nop
# CHECK: 1110: nop
# CHECK: 1117: incl

  .align 32, 0x90
INSTRLEN_9_OFFSET_9:
  .fill 9, 1, 0x90
  .bundle_lock align_to_end
  .rept 9
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1129: nop
# CHECK: 1130: nop
# CHECK: 1137: incl

  .align 32, 0x90
INSTRLEN_9_OFFSET_10:
  .fill 10, 1, 0x90
  .bundle_lock align_to_end
  .rept 9
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 114a: nop
# CHECK: 1150: nop
# CHECK: 1157: incl

  .align 32, 0x90
INSTRLEN_9_OFFSET_11:
  .fill 11, 1, 0x90
  .bundle_lock align_to_end
  .rept 9
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 116b: nop
# CHECK: 1170: nop
# CHECK: 1177: incl

  .align 32, 0x90
INSTRLEN_9_OFFSET_12:
  .fill 12, 1, 0x90
  .bundle_lock align_to_end
  .rept 9
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 118c: nop
# CHECK: 1190: nop
# CHECK: 1197: incl

  .align 32, 0x90
INSTRLEN_9_OFFSET_13:
  .fill 13, 1, 0x90
  .bundle_lock align_to_end
  .rept 9
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 11ad: nop
# CHECK: 11b0: nop
# CHECK: 11b7: incl

  .align 32, 0x90
INSTRLEN_9_OFFSET_14:
  .fill 14, 1, 0x90
  .bundle_lock align_to_end
  .rept 9
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 11ce: nop
# CHECK: 11d0: nop
# CHECK: 11d7: incl

  .align 32, 0x90
INSTRLEN_9_OFFSET_15:
  .fill 15, 1, 0x90
  .bundle_lock align_to_end
  .rept 9
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 11ef: nop
# CHECK: 11f0: nop
# CHECK: 11f7: incl

  .align 32, 0x90
INSTRLEN_10_OFFSET_0:
  .bundle_lock align_to_end
  .rept 10
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1200: nop
# CHECK: 1206: incl

  .align 32, 0x90
INSTRLEN_10_OFFSET_1:
  .fill 1, 1, 0x90
  .bundle_lock align_to_end
  .rept 10
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1221: nop
# CHECK: 1226: incl

  .align 32, 0x90
INSTRLEN_10_OFFSET_2:
  .fill 2, 1, 0x90
  .bundle_lock align_to_end
  .rept 10
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1242: nop
# CHECK: 1246: incl

  .align 32, 0x90
INSTRLEN_10_OFFSET_3:
  .fill 3, 1, 0x90
  .bundle_lock align_to_end
  .rept 10
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1263: nop
# CHECK: 1266: incl

  .align 32, 0x90
INSTRLEN_10_OFFSET_4:
  .fill 4, 1, 0x90
  .bundle_lock align_to_end
  .rept 10
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1284: nop
# CHECK: 1286: incl

  .align 32, 0x90
INSTRLEN_10_OFFSET_5:
  .fill 5, 1, 0x90
  .bundle_lock align_to_end
  .rept 10
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 12a5: nop
# CHECK: 12a6: incl

  .align 32, 0x90
INSTRLEN_10_OFFSET_6:
  .fill 6, 1, 0x90
  .bundle_lock align_to_end
  .rept 10
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 12c6: incl

  .align 32, 0x90
INSTRLEN_10_OFFSET_7:
  .fill 7, 1, 0x90
  .bundle_lock align_to_end
  .rept 10
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 12e7: nop
# CHECK: 12f0: nop
# CHECK: 12f6: incl

  .align 32, 0x90
INSTRLEN_10_OFFSET_8:
  .fill 8, 1, 0x90
  .bundle_lock align_to_end
  .rept 10
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1308: nop
# CHECK: 1310: nop
# CHECK: 1316: incl

  .align 32, 0x90
INSTRLEN_10_OFFSET_9:
  .fill 9, 1, 0x90
  .bundle_lock align_to_end
  .rept 10
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1329: nop
# CHECK: 1330: nop
# CHECK: 1336: incl

  .align 32, 0x90
INSTRLEN_10_OFFSET_10:
  .fill 10, 1, 0x90
  .bundle_lock align_to_end
  .rept 10
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 134a: nop
# CHECK: 1350: nop
# CHECK: 1356: incl

  .align 32, 0x90
INSTRLEN_10_OFFSET_11:
  .fill 11, 1, 0x90
  .bundle_lock align_to_end
  .rept 10
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 136b: nop
# CHECK: 1370: nop
# CHECK: 1376: incl

  .align 32, 0x90
INSTRLEN_10_OFFSET_12:
  .fill 12, 1, 0x90
  .bundle_lock align_to_end
  .rept 10
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 138c: nop
# CHECK: 1390: nop
# CHECK: 1396: incl

  .align 32, 0x90
INSTRLEN_10_OFFSET_13:
  .fill 13, 1, 0x90
  .bundle_lock align_to_end
  .rept 10
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 13ad: nop
# CHECK: 13b0: nop
# CHECK: 13b6: incl

  .align 32, 0x90
INSTRLEN_10_OFFSET_14:
  .fill 14, 1, 0x90
  .bundle_lock align_to_end
  .rept 10
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 13ce: nop
# CHECK: 13d0: nop
# CHECK: 13d6: incl

  .align 32, 0x90
INSTRLEN_10_OFFSET_15:
  .fill 15, 1, 0x90
  .bundle_lock align_to_end
  .rept 10
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 13ef: nop
# CHECK: 13f0: nop
# CHECK: 13f6: incl

  .align 32, 0x90
INSTRLEN_11_OFFSET_0:
  .bundle_lock align_to_end
  .rept 11
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1400: nop
# CHECK: 1405: incl

  .align 32, 0x90
INSTRLEN_11_OFFSET_1:
  .fill 1, 1, 0x90
  .bundle_lock align_to_end
  .rept 11
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1421: nop
# CHECK: 1425: incl

  .align 32, 0x90
INSTRLEN_11_OFFSET_2:
  .fill 2, 1, 0x90
  .bundle_lock align_to_end
  .rept 11
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1442: nop
# CHECK: 1445: incl

  .align 32, 0x90
INSTRLEN_11_OFFSET_3:
  .fill 3, 1, 0x90
  .bundle_lock align_to_end
  .rept 11
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1463: nop
# CHECK: 1465: incl

  .align 32, 0x90
INSTRLEN_11_OFFSET_4:
  .fill 4, 1, 0x90
  .bundle_lock align_to_end
  .rept 11
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1484: nop
# CHECK: 1485: incl

  .align 32, 0x90
INSTRLEN_11_OFFSET_5:
  .fill 5, 1, 0x90
  .bundle_lock align_to_end
  .rept 11
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 14a5: incl

  .align 32, 0x90
INSTRLEN_11_OFFSET_6:
  .fill 6, 1, 0x90
  .bundle_lock align_to_end
  .rept 11
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 14c6: nop
# CHECK: 14d0: nop
# CHECK: 14d5: incl

  .align 32, 0x90
INSTRLEN_11_OFFSET_7:
  .fill 7, 1, 0x90
  .bundle_lock align_to_end
  .rept 11
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 14e7: nop
# CHECK: 14f0: nop
# CHECK: 14f5: incl

  .align 32, 0x90
INSTRLEN_11_OFFSET_8:
  .fill 8, 1, 0x90
  .bundle_lock align_to_end
  .rept 11
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1508: nop
# CHECK: 1510: nop
# CHECK: 1515: incl

  .align 32, 0x90
INSTRLEN_11_OFFSET_9:
  .fill 9, 1, 0x90
  .bundle_lock align_to_end
  .rept 11
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1529: nop
# CHECK: 1530: nop
# CHECK: 1535: incl

  .align 32, 0x90
INSTRLEN_11_OFFSET_10:
  .fill 10, 1, 0x90
  .bundle_lock align_to_end
  .rept 11
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 154a: nop
# CHECK: 1550: nop
# CHECK: 1555: incl

  .align 32, 0x90
INSTRLEN_11_OFFSET_11:
  .fill 11, 1, 0x90
  .bundle_lock align_to_end
  .rept 11
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 156b: nop
# CHECK: 1570: nop
# CHECK: 1575: incl

  .align 32, 0x90
INSTRLEN_11_OFFSET_12:
  .fill 12, 1, 0x90
  .bundle_lock align_to_end
  .rept 11
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 158c: nop
# CHECK: 1590: nop
# CHECK: 1595: incl

  .align 32, 0x90
INSTRLEN_11_OFFSET_13:
  .fill 13, 1, 0x90
  .bundle_lock align_to_end
  .rept 11
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 15ad: nop
# CHECK: 15b0: nop
# CHECK: 15b5: incl

  .align 32, 0x90
INSTRLEN_11_OFFSET_14:
  .fill 14, 1, 0x90
  .bundle_lock align_to_end
  .rept 11
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 15ce: nop
# CHECK: 15d0: nop
# CHECK: 15d5: incl

  .align 32, 0x90
INSTRLEN_11_OFFSET_15:
  .fill 15, 1, 0x90
  .bundle_lock align_to_end
  .rept 11
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 15ef: nop
# CHECK: 15f0: nop
# CHECK: 15f5: incl

  .align 32, 0x90
INSTRLEN_12_OFFSET_0:
  .bundle_lock align_to_end
  .rept 12
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1600: nop
# CHECK: 1604: incl

  .align 32, 0x90
INSTRLEN_12_OFFSET_1:
  .fill 1, 1, 0x90
  .bundle_lock align_to_end
  .rept 12
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1621: nop
# CHECK: 1624: incl

  .align 32, 0x90
INSTRLEN_12_OFFSET_2:
  .fill 2, 1, 0x90
  .bundle_lock align_to_end
  .rept 12
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1642: nop
# CHECK: 1644: incl

  .align 32, 0x90
INSTRLEN_12_OFFSET_3:
  .fill 3, 1, 0x90
  .bundle_lock align_to_end
  .rept 12
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1663: nop
# CHECK: 1664: incl

  .align 32, 0x90
INSTRLEN_12_OFFSET_4:
  .fill 4, 1, 0x90
  .bundle_lock align_to_end
  .rept 12
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1684: incl

  .align 32, 0x90
INSTRLEN_12_OFFSET_5:
  .fill 5, 1, 0x90
  .bundle_lock align_to_end
  .rept 12
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 16a5: nop
# CHECK: 16b0: nop
# CHECK: 16b4: incl

  .align 32, 0x90
INSTRLEN_12_OFFSET_6:
  .fill 6, 1, 0x90
  .bundle_lock align_to_end
  .rept 12
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 16c6: nop
# CHECK: 16d0: nop
# CHECK: 16d4: incl

  .align 32, 0x90
INSTRLEN_12_OFFSET_7:
  .fill 7, 1, 0x90
  .bundle_lock align_to_end
  .rept 12
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 16e7: nop
# CHECK: 16f0: nop
# CHECK: 16f4: incl

  .align 32, 0x90
INSTRLEN_12_OFFSET_8:
  .fill 8, 1, 0x90
  .bundle_lock align_to_end
  .rept 12
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1708: nop
# CHECK: 1710: nop
# CHECK: 1714: incl

  .align 32, 0x90
INSTRLEN_12_OFFSET_9:
  .fill 9, 1, 0x90
  .bundle_lock align_to_end
  .rept 12
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1729: nop
# CHECK: 1730: nop
# CHECK: 1734: incl

  .align 32, 0x90
INSTRLEN_12_OFFSET_10:
  .fill 10, 1, 0x90
  .bundle_lock align_to_end
  .rept 12
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 174a: nop
# CHECK: 1750: nop
# CHECK: 1754: incl

  .align 32, 0x90
INSTRLEN_12_OFFSET_11:
  .fill 11, 1, 0x90
  .bundle_lock align_to_end
  .rept 12
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 176b: nop
# CHECK: 1770: nop
# CHECK: 1774: incl

  .align 32, 0x90
INSTRLEN_12_OFFSET_12:
  .fill 12, 1, 0x90
  .bundle_lock align_to_end
  .rept 12
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 178c: nop
# CHECK: 1790: nop
# CHECK: 1794: incl

  .align 32, 0x90
INSTRLEN_12_OFFSET_13:
  .fill 13, 1, 0x90
  .bundle_lock align_to_end
  .rept 12
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 17ad: nop
# CHECK: 17b0: nop
# CHECK: 17b4: incl

  .align 32, 0x90
INSTRLEN_12_OFFSET_14:
  .fill 14, 1, 0x90
  .bundle_lock align_to_end
  .rept 12
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 17ce: nop
# CHECK: 17d0: nop
# CHECK: 17d4: incl

  .align 32, 0x90
INSTRLEN_12_OFFSET_15:
  .fill 15, 1, 0x90
  .bundle_lock align_to_end
  .rept 12
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 17ef: nop
# CHECK: 17f0: nop
# CHECK: 17f4: incl

  .align 32, 0x90
INSTRLEN_13_OFFSET_0:
  .bundle_lock align_to_end
  .rept 13
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1800: nop
# CHECK: 1803: incl

  .align 32, 0x90
INSTRLEN_13_OFFSET_1:
  .fill 1, 1, 0x90
  .bundle_lock align_to_end
  .rept 13
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1821: nop
# CHECK: 1823: incl

  .align 32, 0x90
INSTRLEN_13_OFFSET_2:
  .fill 2, 1, 0x90
  .bundle_lock align_to_end
  .rept 13
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1842: nop
# CHECK: 1843: incl

  .align 32, 0x90
INSTRLEN_13_OFFSET_3:
  .fill 3, 1, 0x90
  .bundle_lock align_to_end
  .rept 13
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1863: incl

  .align 32, 0x90
INSTRLEN_13_OFFSET_4:
  .fill 4, 1, 0x90
  .bundle_lock align_to_end
  .rept 13
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1884: nop
# CHECK: 1890: nop
# CHECK: 1893: incl

  .align 32, 0x90
INSTRLEN_13_OFFSET_5:
  .fill 5, 1, 0x90
  .bundle_lock align_to_end
  .rept 13
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 18a5: nop
# CHECK: 18b0: nop
# CHECK: 18b3: incl

  .align 32, 0x90
INSTRLEN_13_OFFSET_6:
  .fill 6, 1, 0x90
  .bundle_lock align_to_end
  .rept 13
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 18c6: nop
# CHECK: 18d0: nop
# CHECK: 18d3: incl

  .align 32, 0x90
INSTRLEN_13_OFFSET_7:
  .fill 7, 1, 0x90
  .bundle_lock align_to_end
  .rept 13
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 18e7: nop
# CHECK: 18f0: nop
# CHECK: 18f3: incl

  .align 32, 0x90
INSTRLEN_13_OFFSET_8:
  .fill 8, 1, 0x90
  .bundle_lock align_to_end
  .rept 13
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1908: nop
# CHECK: 1910: nop
# CHECK: 1913: incl

  .align 32, 0x90
INSTRLEN_13_OFFSET_9:
  .fill 9, 1, 0x90
  .bundle_lock align_to_end
  .rept 13
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1929: nop
# CHECK: 1930: nop
# CHECK: 1933: incl

  .align 32, 0x90
INSTRLEN_13_OFFSET_10:
  .fill 10, 1, 0x90
  .bundle_lock align_to_end
  .rept 13
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 194a: nop
# CHECK: 1950: nop
# CHECK: 1953: incl

  .align 32, 0x90
INSTRLEN_13_OFFSET_11:
  .fill 11, 1, 0x90
  .bundle_lock align_to_end
  .rept 13
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 196b: nop
# CHECK: 1970: nop
# CHECK: 1973: incl

  .align 32, 0x90
INSTRLEN_13_OFFSET_12:
  .fill 12, 1, 0x90
  .bundle_lock align_to_end
  .rept 13
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 198c: nop
# CHECK: 1990: nop
# CHECK: 1993: incl

  .align 32, 0x90
INSTRLEN_13_OFFSET_13:
  .fill 13, 1, 0x90
  .bundle_lock align_to_end
  .rept 13
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 19ad: nop
# CHECK: 19b0: nop
# CHECK: 19b3: incl

  .align 32, 0x90
INSTRLEN_13_OFFSET_14:
  .fill 14, 1, 0x90
  .bundle_lock align_to_end
  .rept 13
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 19ce: nop
# CHECK: 19d0: nop
# CHECK: 19d3: incl

  .align 32, 0x90
INSTRLEN_13_OFFSET_15:
  .fill 15, 1, 0x90
  .bundle_lock align_to_end
  .rept 13
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 19ef: nop
# CHECK: 19f0: nop
# CHECK: 19f3: incl

  .align 32, 0x90
INSTRLEN_14_OFFSET_0:
  .bundle_lock align_to_end
  .rept 14
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1a00: nop
# CHECK: 1a02: incl

  .align 32, 0x90
INSTRLEN_14_OFFSET_1:
  .fill 1, 1, 0x90
  .bundle_lock align_to_end
  .rept 14
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1a21: nop
# CHECK: 1a22: incl

  .align 32, 0x90
INSTRLEN_14_OFFSET_2:
  .fill 2, 1, 0x90
  .bundle_lock align_to_end
  .rept 14
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1a42: incl

  .align 32, 0x90
INSTRLEN_14_OFFSET_3:
  .fill 3, 1, 0x90
  .bundle_lock align_to_end
  .rept 14
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1a63: nop
# CHECK: 1a70: nop
# CHECK: 1a72: incl

  .align 32, 0x90
INSTRLEN_14_OFFSET_4:
  .fill 4, 1, 0x90
  .bundle_lock align_to_end
  .rept 14
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1a84: nop
# CHECK: 1a90: nop
# CHECK: 1a92: incl

  .align 32, 0x90
INSTRLEN_14_OFFSET_5:
  .fill 5, 1, 0x90
  .bundle_lock align_to_end
  .rept 14
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1aa5: nop
# CHECK: 1ab0: nop
# CHECK: 1ab2: incl

  .align 32, 0x90
INSTRLEN_14_OFFSET_6:
  .fill 6, 1, 0x90
  .bundle_lock align_to_end
  .rept 14
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1ac6: nop
# CHECK: 1ad0: nop
# CHECK: 1ad2: incl

  .align 32, 0x90
INSTRLEN_14_OFFSET_7:
  .fill 7, 1, 0x90
  .bundle_lock align_to_end
  .rept 14
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1ae7: nop
# CHECK: 1af0: nop
# CHECK: 1af2: incl

  .align 32, 0x90
INSTRLEN_14_OFFSET_8:
  .fill 8, 1, 0x90
  .bundle_lock align_to_end
  .rept 14
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1b08: nop
# CHECK: 1b10: nop
# CHECK: 1b12: incl

  .align 32, 0x90
INSTRLEN_14_OFFSET_9:
  .fill 9, 1, 0x90
  .bundle_lock align_to_end
  .rept 14
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1b29: nop
# CHECK: 1b30: nop
# CHECK: 1b32: incl

  .align 32, 0x90
INSTRLEN_14_OFFSET_10:
  .fill 10, 1, 0x90
  .bundle_lock align_to_end
  .rept 14
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1b4a: nop
# CHECK: 1b50: nop
# CHECK: 1b52: incl

  .align 32, 0x90
INSTRLEN_14_OFFSET_11:
  .fill 11, 1, 0x90
  .bundle_lock align_to_end
  .rept 14
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1b6b: nop
# CHECK: 1b70: nop
# CHECK: 1b72: incl

  .align 32, 0x90
INSTRLEN_14_OFFSET_12:
  .fill 12, 1, 0x90
  .bundle_lock align_to_end
  .rept 14
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1b8c: nop
# CHECK: 1b90: nop
# CHECK: 1b92: incl

  .align 32, 0x90
INSTRLEN_14_OFFSET_13:
  .fill 13, 1, 0x90
  .bundle_lock align_to_end
  .rept 14
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1bad: nop
# CHECK: 1bb0: nop
# CHECK: 1bb2: incl

  .align 32, 0x90
INSTRLEN_14_OFFSET_14:
  .fill 14, 1, 0x90
  .bundle_lock align_to_end
  .rept 14
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1bce: nop
# CHECK: 1bd0: nop
# CHECK: 1bd2: incl

  .align 32, 0x90
INSTRLEN_14_OFFSET_15:
  .fill 15, 1, 0x90
  .bundle_lock align_to_end
  .rept 14
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1bef: nop
# CHECK: 1bf0: nop
# CHECK: 1bf2: incl

  .align 32, 0x90
INSTRLEN_15_OFFSET_0:
  .bundle_lock align_to_end
  .rept 15
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1c00: nop
# CHECK: 1c01: incl

  .align 32, 0x90
INSTRLEN_15_OFFSET_1:
  .fill 1, 1, 0x90
  .bundle_lock align_to_end
  .rept 15
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1c21: incl

  .align 32, 0x90
INSTRLEN_15_OFFSET_2:
  .fill 2, 1, 0x90
  .bundle_lock align_to_end
  .rept 15
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1c42: nop
# CHECK: 1c50: nop
# CHECK: 1c51: incl

  .align 32, 0x90
INSTRLEN_15_OFFSET_3:
  .fill 3, 1, 0x90
  .bundle_lock align_to_end
  .rept 15
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1c63: nop
# CHECK: 1c70: nop
# CHECK: 1c71: incl

  .align 32, 0x90
INSTRLEN_15_OFFSET_4:
  .fill 4, 1, 0x90
  .bundle_lock align_to_end
  .rept 15
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1c84: nop
# CHECK: 1c90: nop
# CHECK: 1c91: incl

  .align 32, 0x90
INSTRLEN_15_OFFSET_5:
  .fill 5, 1, 0x90
  .bundle_lock align_to_end
  .rept 15
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1ca5: nop
# CHECK: 1cb0: nop
# CHECK: 1cb1: incl

  .align 32, 0x90
INSTRLEN_15_OFFSET_6:
  .fill 6, 1, 0x90
  .bundle_lock align_to_end
  .rept 15
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1cc6: nop
# CHECK: 1cd0: nop
# CHECK: 1cd1: incl

  .align 32, 0x90
INSTRLEN_15_OFFSET_7:
  .fill 7, 1, 0x90
  .bundle_lock align_to_end
  .rept 15
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1ce7: nop
# CHECK: 1cf0: nop
# CHECK: 1cf1: incl

  .align 32, 0x90
INSTRLEN_15_OFFSET_8:
  .fill 8, 1, 0x90
  .bundle_lock align_to_end
  .rept 15
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1d08: nop
# CHECK: 1d10: nop
# CHECK: 1d11: incl

  .align 32, 0x90
INSTRLEN_15_OFFSET_9:
  .fill 9, 1, 0x90
  .bundle_lock align_to_end
  .rept 15
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1d29: nop
# CHECK: 1d30: nop
# CHECK: 1d31: incl

  .align 32, 0x90
INSTRLEN_15_OFFSET_10:
  .fill 10, 1, 0x90
  .bundle_lock align_to_end
  .rept 15
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1d4a: nop
# CHECK: 1d50: nop
# CHECK: 1d51: incl

  .align 32, 0x90
INSTRLEN_15_OFFSET_11:
  .fill 11, 1, 0x90
  .bundle_lock align_to_end
  .rept 15
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1d6b: nop
# CHECK: 1d70: nop
# CHECK: 1d71: incl

  .align 32, 0x90
INSTRLEN_15_OFFSET_12:
  .fill 12, 1, 0x90
  .bundle_lock align_to_end
  .rept 15
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1d8c: nop
# CHECK: 1d90: nop
# CHECK: 1d91: incl

  .align 32, 0x90
INSTRLEN_15_OFFSET_13:
  .fill 13, 1, 0x90
  .bundle_lock align_to_end
  .rept 15
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1dad: nop
# CHECK: 1db0: nop
# CHECK: 1db1: incl

  .align 32, 0x90
INSTRLEN_15_OFFSET_14:
  .fill 14, 1, 0x90
  .bundle_lock align_to_end
  .rept 15
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1dce: nop
# CHECK: 1dd0: nop
# CHECK: 1dd1: incl

  .align 32, 0x90
INSTRLEN_15_OFFSET_15:
  .fill 15, 1, 0x90
  .bundle_lock align_to_end
  .rept 15
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1def: nop
# CHECK: 1df0: nop
# CHECK: 1df1: incl

  .align 32, 0x90
INSTRLEN_16_OFFSET_0:
  .bundle_lock align_to_end
  .rept 16
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1e00: incl

  .align 32, 0x90
INSTRLEN_16_OFFSET_1:
  .fill 1, 1, 0x90
  .bundle_lock align_to_end
  .rept 16
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1e21: nop
# CHECK: 1e30: incl

  .align 32, 0x90
INSTRLEN_16_OFFSET_2:
  .fill 2, 1, 0x90
  .bundle_lock align_to_end
  .rept 16
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1e42: nop
# CHECK: 1e50: incl

  .align 32, 0x90
INSTRLEN_16_OFFSET_3:
  .fill 3, 1, 0x90
  .bundle_lock align_to_end
  .rept 16
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1e63: nop
# CHECK: 1e70: incl

  .align 32, 0x90
INSTRLEN_16_OFFSET_4:
  .fill 4, 1, 0x90
  .bundle_lock align_to_end
  .rept 16
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1e84: nop
# CHECK: 1e90: incl

  .align 32, 0x90
INSTRLEN_16_OFFSET_5:
  .fill 5, 1, 0x90
  .bundle_lock align_to_end
  .rept 16
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1ea5: nop
# CHECK: 1eb0: incl

  .align 32, 0x90
INSTRLEN_16_OFFSET_6:
  .fill 6, 1, 0x90
  .bundle_lock align_to_end
  .rept 16
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1ec6: nop
# CHECK: 1ed0: incl

  .align 32, 0x90
INSTRLEN_16_OFFSET_7:
  .fill 7, 1, 0x90
  .bundle_lock align_to_end
  .rept 16
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1ee7: nop
# CHECK: 1ef0: incl

  .align 32, 0x90
INSTRLEN_16_OFFSET_8:
  .fill 8, 1, 0x90
  .bundle_lock align_to_end
  .rept 16
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1f08: nop
# CHECK: 1f10: incl

  .align 32, 0x90
INSTRLEN_16_OFFSET_9:
  .fill 9, 1, 0x90
  .bundle_lock align_to_end
  .rept 16
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1f29: nop
# CHECK: 1f30: incl

  .align 32, 0x90
INSTRLEN_16_OFFSET_10:
  .fill 10, 1, 0x90
  .bundle_lock align_to_end
  .rept 16
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1f4a: nop
# CHECK: 1f50: incl

  .align 32, 0x90
INSTRLEN_16_OFFSET_11:
  .fill 11, 1, 0x90
  .bundle_lock align_to_end
  .rept 16
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1f6b: nop
# CHECK: 1f70: incl

  .align 32, 0x90
INSTRLEN_16_OFFSET_12:
  .fill 12, 1, 0x90
  .bundle_lock align_to_end
  .rept 16
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1f8c: nop
# CHECK: 1f90: incl

  .align 32, 0x90
INSTRLEN_16_OFFSET_13:
  .fill 13, 1, 0x90
  .bundle_lock align_to_end
  .rept 16
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1fad: nop
# CHECK: 1fb0: incl

  .align 32, 0x90
INSTRLEN_16_OFFSET_14:
  .fill 14, 1, 0x90
  .bundle_lock align_to_end
  .rept 16
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1fce: nop
# CHECK: 1fd0: incl

  .align 32, 0x90
INSTRLEN_16_OFFSET_15:
  .fill 15, 1, 0x90
  .bundle_lock align_to_end
  .rept 16
  inc %eax
  .endr
  .bundle_unlock
# CHECK: 1fef: nop
# CHECK: 1ff0: incl

