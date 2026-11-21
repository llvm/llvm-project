; RUN: llc -mtriple=mipsel-unknown-linux-gnu -mcpu=mips32r2 -mattr=+dsp \
; RUN:   -enable-shrink-wrap=true -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=mipsel-unknown-linux-gnu -mcpu=mips32r2 -mattr=+dsp \
; RUN:   -enable-shrink-wrap=false -verify-machineinstrs < %s | FileCheck %s

declare i64 @llvm.mips.mthlip(i64, i32) nounwind
declare i32 @llvm.mips.extr.w(i64, i32) nounwind
declare void @slow_call()

; Verify that shrink-wrapping does not sink stack frame setup past an
; accumulator COPY that expands to GPR virtual registers and scavenges via an
; emergency spill slot.
define i32 @copy_acc_dsp(i64 %acc) nounwind {
; CHECK-LABEL: copy_acc_dsp:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    addiu $sp, $sp, -24
; CHECK:         mflo ${{[0-9]+|ra}}
; CHECK:         mtlo ${{[0-9]+|ra}}, $ac1
; CHECK:         mfhi ${{[0-9]+|ra}}
; CHECK:         mthi ${{[0-9]+|ra}}, $ac1
entry:
  %x0 = call i32 @llvm.mips.extr.w(i64 %acc, i32 0)
  %regs = call { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
      asm sideeffect "", "={$1},={$2},={$3},={$4},={$5},={$6},={$7},={$8},={$9},={$10},={$11},={$12},={$13},={$14},={$15},={$24},={$25},={$28}"()
  %r1  = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %regs, 0
  %r2  = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %regs, 1
  %r3  = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %regs, 2
  %r4  = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %regs, 3
  %r5  = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %regs, 4
  %r6  = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %regs, 5
  %r7  = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %regs, 6
  %r8  = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %regs, 7
  %r9  = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %regs, 8
  %r10 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %regs, 9
  %r11 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %regs, 10
  %r12 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %regs, 11
  %r13 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %regs, 12
  %r14 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %regs, 13
  %r15 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %regs, 14
  %r24 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %regs, 15
  %r25 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %regs, 16
  %r28 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %regs, 17
  %unused = call i64 @llvm.mips.mthlip(i64 %acc, i32 %r1)
  call void asm sideeffect "", "{$1},{$2},{$3},{$4},{$5},{$6},{$7},{$8},{$9},{$10},{$11},{$12},{$13},{$14},{$15},{$24},{$25},{$28}"(
      i32 %r1, i32 %r2, i32 %r3, i32 %r4, i32 %r5, i32 %r6, i32 %r7, i32 %r8, i32 %r9,
      i32 %r10, i32 %r11, i32 %r12, i32 %r13, i32 %r14, i32 %r15, i32 %r24, i32 %r25, i32 %r28)
  %x1 = call i32 @llvm.mips.extr.w(i64 %acc, i32 1)
  %cmp = icmp eq i32 %x1, 0
  br i1 %cmp, label %slow, label %exit
slow:
  call void @slow_call()
  ret i32 0
exit:
  ret i32 %x1
}
