;; KCFI checks next to instructions with their own packetization constraints:
;; slot-restricted cache ops, HVX, multiply-accumulates, loop compare-and-branch
;; pairs, and register pressure that pushes the call target out of the scratch
;; registers the lowering prefers.  Target intrinsics are used because they are
;; the only way to place a *particular* instruction next to the check from IR.

; RUN: llc -mtriple=hexagon -mattr=+hvxv68,+hvx-length128b -filetype=obj < %s \
; RUN:   | llvm-objdump -d --no-show-raw-insn - | FileCheck %s

; RUN: llc -mtriple=hexagon -mattr=+hvxv68,+hvx-length128b --disable-packetizer \
; RUN:   -filetype=obj < %s | llvm-objdump -d --no-show-raw-insn - | FileCheck %s

; RUN: llc -mtriple=hexagon -mattr=+hvxv68,+hvx-length128b -filetype=obj < %s -o %t.o
; RUN: llc -mtriple=hexagon -mattr=+hvxv68,+hvx-length128b -filetype=asm < %s -o %t.s
; RUN: llvm-mc -triple=hexagon -mattr=+hvxv68,+hvx-length128b -filetype=obj %t.s -o %t.a.o
; RUN: llvm-objcopy -O binary --only-section=.text %t.o %t.bin
; RUN: llvm-objcopy -O binary --only-section=.text %t.a.o %t.a.bin
; RUN: cmp %t.bin %t.a.bin

declare void @llvm.hexagon.Y2.dccleana(ptr)
declare i32 @llvm.hexagon.A2.abs(i32)
declare i32 @llvm.hexagon.M2.mpy.acc.sat.ll.s0(i32, i32, i32)
declare <32 x i32> @llvm.hexagon.V6.vaddw.128B(<32 x i32>, <32 x i32>)
declare <32 x i32> @llvm.hexagon.V6.vmpyiwb.128B(<32 x i32>, i32)
declare void @llvm.prefetch.p0(ptr, i32, i32, i32)

;; Cache operations are restricted to a single slot, so the packetizer cannot
;; fold them into the check's packets.  The braces are what actually asserts
;; that: each dccleana stands alone, and the call is not pulled in with either.
define void @cache_ops(ptr noundef %fp, ptr %p) {
; CHECK-LABEL: <cache_ops>:
; CHECK:        { dccleana(r{{[0-9]+}}) }
; CHECK-NEXT:   { immext(#
; CHECK-NEXT:     r{{[0-9]+}} = ##0xbc614e
; CHECK-NEXT:     r{{[0-9]+}} = memw(r0+#-0x4) }
; CHECK-NEXT:   { p0 = cmp.eq(r{{[0-9]+}},r{{[0-9]+}})
; CHECK:          if (p0.new) jump:t {{.*}} }
; CHECK-NEXT:   { immext(#0xbadc0fc0)
; CHECK-NEXT:     r{{[0-9]+}}:{{[0-9]+}} = memd(##0xbadc0fee) }
; CHECK-NEXT:   { callr r0 }
; CHECK-NEXT:   { dccleana(r{{[0-9]+}}) }
  call void @llvm.hexagon.Y2.dccleana(ptr %p)
  call void %fp() [ "kcfi"(i32 12345678) ]
  call void @llvm.hexagon.Y2.dccleana(ptr %p)
  ret void
}

;; HVX vector ops either side of the check.  Vector work uses slots 0 and 1 and
;; forces vector spills around the call, which lands stores next to the check.
define <32 x i32> @hvx_neighbors(ptr noundef %fp, <32 x i32> %a, <32 x i32> %b) {
; CHECK-LABEL: <hvx_neighbors>:
; CHECK:        v{{[0-9]+}}.w = vadd(v{{[0-9]+}}.w,v{{[0-9]+}}.w)
; CHECK:        immext(#
; CHECK-NEXT:   r{{[0-9]+}} = ##0x3e7
; CHECK-NEXT:   r{{[0-9]+}} = memw(r0+#-0x4) }
; CHECK:        immext(#0xbadc0fc0)
; CHECK-NEXT:   r{{[0-9]+}}:{{[0-9]+}} = memd(##0xbadc0fee)
; CHECK:        callr r0
  %v = call <32 x i32> @llvm.hexagon.V6.vaddw.128B(<32 x i32> %a, <32 x i32> %b)
  call void %fp() [ "kcfi"(i32 999) ]
  %w = call <32 x i32> @llvm.hexagon.V6.vaddw.128B(<32 x i32> %v, <32 x i32> %b)
  ret <32 x i32> %w
}

;; A vector multiply that needs a scalar operand keeps a GPR live across the
;; check, competing with the scratch registers it wants.
define <32 x i32> @hvx_scalar_operand(ptr noundef %fp, <32 x i32> %a, i32 %s) {
; CHECK-LABEL: <hvx_scalar_operand>:
; CHECK:        immext(#0xbadc0fc0)
; CHECK-NEXT:   r{{[0-9]+}}:{{[0-9]+}} = memd(##0xbadc0fee)
; CHECK:        callr r0
  %v = call <32 x i32> @llvm.hexagon.V6.vmpyiwb.128B(<32 x i32> %a, i32 %s)
  call void %fp() [ "kcfi"(i32 12345678) ]
  %w = call <32 x i32> @llvm.hexagon.V6.vmpyiwb.128B(<32 x i32> %v, i32 %s)
  ret <32 x i32> %w
}

;; A multiply-accumulate has a read-modify-write operand the packetizer tracks
;; separately; keep one live across the check.
define i32 @mpy_accumulate(ptr noundef %fp, i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: <mpy_accumulate>:
; CHECK:        r{{[0-9]+}} += mpy(r{{[0-9]+}}.l,r{{[0-9]+}}.l):sat
; CHECK:        immext(#
; CHECK-NEXT:   r{{[0-9]+}} = ##0x4d2
; CHECK:        immext(#0xbadc0fc0)
; CHECK-NEXT:   r{{[0-9]+}}:{{[0-9]+}} = memd(##0xbadc0fee)
  %m = call i32 @llvm.hexagon.M2.mpy.acc.sat.ll.s0(i32 %a, i32 %b, i32 %c)
  call void %fp() [ "kcfi"(i32 1234) ]
  %n = call i32 @llvm.hexagon.A2.abs(i32 %m)
  ret i32 %n
}

;; dcfetch sits immediately before the check, and stays out of its packets.
define void @prefetch_before(ptr noundef %fp, ptr %p) {
; CHECK-LABEL: <prefetch_before>:
; CHECK:        { dcfetch(r{{[0-9]+}}{{.*}}) }
; CHECK-NEXT:   { immext(#
; CHECK-NEXT:     r{{[0-9]+}} = ##0x63
; CHECK-NEXT:     r{{[0-9]+}} = memw(r0+#-0x4) }
; CHECK-NEXT:   { p0 = cmp.eq(r{{[0-9]+}},r{{[0-9]+}})
; CHECK:          if (p0.new) jump:t {{.*}} }
; CHECK-NEXT:   { immext(#0xbadc0fc0)
; CHECK-NEXT:     r{{[0-9]+}}:{{[0-9]+}} = memd(##0xbadc0fee) }
; CHECK-NEXT:   { callr r0 }
  call void @llvm.prefetch.p0(ptr %p, i32 0, i32 3, i32 1)
  call void %fp() [ "kcfi"(i32 99) ]
  ret void
}

;; Six arguments fill r0-r5, forcing the lowering off its r6/r7 scratch pair.
;; r8 cannot be encoded in the compare-jump compound, so the compare and jump
;; stay two instructions.  Registers are spelled out and CHECK-NEXT used so the
;; compounded form cannot match instead.
define void @scratch_fallback(ptr noundef %fp, i32 %a, i32 %b, i32 %c,
                              i32 %d, i32 %e, i32 %f) {
; CHECK-LABEL: <scratch_fallback>:
; CHECK:        { immext(#
; CHECK-NEXT:     r7 = ##0xbc614e
; CHECK-NEXT:     r8 = memw(r6+#-0x4) }
; CHECK-NEXT:   { p0 = cmp.eq(r8,r7)
; CHECK-NEXT:     if (p0.new) jump:t {{.*}} }
; CHECK-NEXT:   { immext(#0xbadc0fc0)
; CHECK-NEXT:     r{{[0-9]+}}:{{[0-9]+}} = memd(##0xbadc0fee) }
; CHECK-NEXT:   { callr r6 }
  call void %fp(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f)
      [ "kcfi"(i32 12345678) ]
  ret void
}

;; The check inside a loop body, sharing the block with the loop's own
;; compare-and-branch -- both want the predicate registers and slot 2/3.
define void @loop_body(ptr noundef %fp, i32 %n) {
; CHECK-LABEL: <loop_body>:
; CHECK:        immext(#0xbadc0fc0)
; CHECK-NEXT:   r{{[0-9]+}}:{{[0-9]+}} = memd(##0xbadc0fee)
; CHECK:        callr
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %inc, %loop ]
  call void %fp() [ "kcfi"(i32 12345678) ]
  %inc = add i32 %i, 1
  %cmp = icmp slt i32 %inc, %n
  br i1 %cmp, label %loop, label %exit
exit:
  ret void
}

;; Back-to-back indirect calls: three checks with nothing between them, so any
;; state carried between packets shows up here.
define void @back_to_back(ptr noundef %f, ptr noundef %g, ptr noundef %h) {
; CHECK-LABEL: <back_to_back>:
; CHECK:        immext(#0xbadc0fc0)
; CHECK-NEXT:   r{{[0-9]+}}:{{[0-9]+}} = memd(##0xbadc0fee)
; CHECK:        immext(#0xbadc0fc0)
; CHECK-NEXT:   r{{[0-9]+}}:{{[0-9]+}} = memd(##0xbadc0fee)
; CHECK:        immext(#0xbadc0fc0)
; CHECK-NEXT:   r{{[0-9]+}}:{{[0-9]+}} = memd(##0xbadc0fee)
  call void %f() [ "kcfi"(i32 1) ]
  call void %g() [ "kcfi"(i32 2) ]
  call void %h() [ "kcfi"(i32 3) ]
  ret void
}

;; The target is loaded from memory immediately before the check, so the
;; address register is defined in the packet right before the check's load.
define void @target_from_memory(ptr noundef %slot) {
; CHECK-LABEL: <target_from_memory>:
; CHECK:        r{{[0-9]+}} = memw(r0
; CHECK:        immext(#0xbadc0fc0)
; CHECK-NEXT:   r{{[0-9]+}}:{{[0-9]+}} = memd(##0xbadc0fee)
  %fp = load ptr, ptr %slot
  call void %fp() [ "kcfi"(i32 12345678) ]
  ret void
}

;; Conditional call: only one arm is checked, so the check's own branch has to
;; coexist with the surrounding control flow.
define void @conditional(ptr noundef %fp, i1 %c) {
; CHECK-LABEL: <conditional>:
; CHECK:        immext(#0xbadc0fc0)
; CHECK-NEXT:   r{{[0-9]+}}:{{[0-9]+}} = memd(##0xbadc0fee)
entry:
  br i1 %c, label %do, label %skip
do:
  call void %fp() [ "kcfi"(i32 12345678) ]
  br label %skip
skip:
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 4, !"kcfi", i32 1}
