# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple x86_64 %t/prefix-pad.s --x86-pad-max-prefix-size=5 \
# RUN:   | llvm-objdump -d --no-show-raw-insn - | FileCheck %t/prefix-pad.s
# RUN: llvm-mc -filetype=obj -triple x86_64 %t/no-prefix-pad.s --x86-pad-max-prefix-size=1 \
# RUN:   | llvm-objdump -d - | FileCheck %t/no-prefix-pad.s

#--- prefix-pad.s
  .section text1, "x"
add_prefix_prev:
  .bundle_align_mode 5
## This callq instruction is 5 bytes long
  callq   bar
  callq   bar
  callq   bar
  callq   bar
  .bundle_lock align_to_end
  callq   bar
  .bundle_unlock
# CHECK:        0:  call
# CHECK-NEXT:   5:  call
# CHECK-NEXT:   a:  call
# CHECK-NEXT:   11:  call
# CHECK-NEXT:   1b:  call

  .p2align 5
add_prefix_prev_next:
  callq   bar
  .bundle_lock align_to_end
  ## instructions inside a bundle lock can also be prefix-padded.
  callq   bar
  .bundle_unlock
# CHECK:        20: call
# CHECK-NEXT:   2a: nop
# CHECK:        36: call

  .p2align 5
ignore_nop_for_p2align:
  int3
  ## no prefix padding with this 15-byte nop.
  .p2align 4
  ## prefix padding with only callq, not the early int3
  .bundle_lock align_to_end
  callq bar
  .bundle_unlock
# CHECK:        40: int3
# CHECK-NEXT:   41: nop
# CHECK-NEXT:   50: nop
# CHECK-NEXT:   56: call

ignore_nop_for_p2align5:
  callq   bar
  .p2align 5
.L1:
  callq   bar
# CHECK:        60: call
# CHECK-NEXT:   65: nop
# CHECK:        80: call

  .section text2, "x"
## ensure the last instructions are not prefix-padded
tail_bundle:
  .bundle_lock
  callq   bar
  .bundle_unlock
  nop
# CHECK:        0: call
# CHECK-NEXT:   5: nop

## Without prefix padding, the align_to_end group creates a 24-byte nop that
## spans a bundle boundary, out of a single BoundaryAlign Fragment. This test
## case ensures the 24-byte nop is consumed without overflowing any bundle.
  .section text3, "x"
ensure_bundle_boundary:
  ## 20-byte group (5-byte * 4)
  .bundle_lock
  .rept 4
  callq foo
  .endr
  .bundle_unlock
  ## consume 12-byte by prefix-padding the first group (back-to-front).
  ## consume another 12-byte by prefix-padding the second group (front-to-back).
  .bundle_lock align_to_end
  ## Although a maximum prefix budget is 20 bytes from this group,
  ## only 12 bytes will be used.
  .rept 4
  callq bar
  .endr
  .bundle_unlock
# CHECK:        0:  call
# CHECK-NEXT:   5:  call
# CHECK-NEXT:   c:  call
# CHECK-NEXT:   16: call
# CHECK-NEXT:   20: call
# CHECK-NEXT:   2a: call
# CHECK-NEXT:   34: call
# CHECK-NEXT:   3b: call

## This test contains instructions that must not be padded.
#--- no-prefix-pad.s
  .text
  .bundle_align_mode 5
prefix_cmpxchg_in_a_bundle:
  movl	48(%rbx), %r12d
  movl	__thread_list_lock(%rip), %eax
  cmpl	%r12d, %eax
  jne	.LBB4_7
  incl	tl_lock_count(%rip)
  jmp	.LBB4_10
  .LBB4_7:
  xorl	%eax, %eax
  lock
  cmpxchgl	%r12d, __thread_list_lock(%rip)
# CHECK:        1c: 2e 31 c0                    xorl
# CHECK-NEXT:   1f: 90                          nop
## lock must start the new bundle, not replacing the nop right above
# CHECK-NEXT:   20: f0                          lock
# CHECK-NEXT:   21: 44 0f b1 25 00 00 00 00     cmpxchgl %r12d, (%rip)
  .LBB4_10:
  xor %rax, %rax

no_pad_before_after_prefix:
  .p2align 5
  lock
  cmpxchgl	%r12d, __thread_list_lock(%rip)
# CHECK:        40: f0                           lock
# CHECK-NEXT:   41: 44 0f b1 25 00 00 00 00      cmpxchgl %r12d, (%rip)
