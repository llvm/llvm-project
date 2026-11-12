; RUN: llc < %s -mtriple=aarch64-none-linux-gnu --verify-machineinstrs -aarch64-b-offset-bits=9 -aarch64-tbz-offset-bits=6 -aarch64-cbz-offset-bits=6 -aarch64-bcc-offset-bits=6 | FileCheck %s

define void @relax_b_nospill(i1 zeroext %0) {
; CHECK-LABEL: relax_b_nospill:
; CHECK:       // %bb.0:                               // %entry
; CHECK-NEXT:    tbnz w0,
; CHECK-SAME:                 LBB0_1
; CHECK-NEXT:  // %bb.3:                               // %entry
; CHECK-NEXT:          b      .LBB0_2
; CHECK-NEXT:  .LBB0_1:                                // %iftrue
; CHECK-NEXT:          //APP
; CHECK-NEXT:          .zero   2048
; CHECK-NEXT:          //NO_APP
; CHECK-NEXT:          ret
; CHECK-NEXT:  .LBB0_2:                                // %iffalse
; CHECK-NEXT:          //APP
; CHECK-NEXT:          .zero   8
; CHECK-NEXT:          //NO_APP
; CHECK-NEXT:          ret
entry:
  br i1 %0, label %iftrue, label %iffalse

iftrue:
  call void asm sideeffect ".space 2048", ""()
  ret void

iffalse:
  call void asm sideeffect ".space 8", ""()
  ret void
}

define void @relax_b_spill() {
; CHECK-LABEL:    relax_b_spill:                          // @relax_b_spill
; CHECK:          // %bb.0:                               // %entry
; CHECK-COUNT-5:          // 16-byte Folded Spill
; CHECK-NOT:              // 16-byte Folded Spill
; CHECK:                  //APP
; CHECK-COUNT-29:         mov     {{x[0-9]+}},
; CHECK-NOT:              mov     {{x[0-9]+}},
; CHECK-NEXT:             //NO_APP
; CHECK-NEXT:             b.eq    .LBB1_1
; CHECK-NEXT:     // %bb.4:                               // %entry
; CHECK-NEXT:             str     [[SPILL_REGISTER:x[0-9]+]], [sp,
; CHECK-SAME:                                                       -16]!
; CHECK-NEXT:             b       .LBB1_5
; CHECK-NEXT:     .LBB1_1:                                // %iftrue
; CHECK-NEXT:             //APP
; CHECK-NEXT:             .zero   2048
; CHECK-NEXT:             //NO_APP
; CHECK-NEXT:             b       .LBB1_3
; CHECK-NEXT:     .LBB1_5:                                // %iffalse
; CHECK-NEXT:             ldr     [[SPILL_REGISTER]], [sp], 
; CHECK-SAME:                                                        16
; CHECK-NEXT:     // %bb.2:                               // %iffalse
; CHECK-NEXT:             //APP
; CHECK-COUNT-29:         // reg use {{x[0-9]+}}
; CHECK-NOT:              // reg use {{x[0-9]+}}
; CHECK-NEXT:             //NO_APP
; CHECK-NEXT:     .LBB1_3:                                // %common.ret
; CHECK-COUNT-5:          // 16-byte Folded Reload
; CHECK-NOT:              // 16-byte Folded Reload
; CHECK-NEXT:             ret
entry:
  %x0 = call i64 asm sideeffect "mov x0, 1", "={x0}"()
  %x1 = call i64 asm sideeffect "mov x1, 1", "={x1}"()
  %x2 = call i64 asm sideeffect "mov x2, 1", "={x2}"()
  %x3 = call i64 asm sideeffect "mov x3, 1", "={x3}"()
  %x4 = call i64 asm sideeffect "mov x4, 1", "={x4}"()
  %x5 = call i64 asm sideeffect "mov x5, 1", "={x5}"()
  %x6 = call i64 asm sideeffect "mov x6, 1", "={x6}"()
  %x7 = call i64 asm sideeffect "mov x7, 1", "={x7}"()
  %x8 = call i64 asm sideeffect "mov x8, 1", "={x8}"()
  %x9 = call i64 asm sideeffect "mov x9, 1", "={x9}"()
  %x10 = call i64 asm sideeffect "mov x10, 1", "={x10}"()
  %x11 = call i64 asm sideeffect "mov x11, 1", "={x11}"()
  %x12 = call i64 asm sideeffect "mov x12, 1", "={x12}"()
  %x13 = call i64 asm sideeffect "mov x13, 1", "={x13}"()
  %x14 = call i64 asm sideeffect "mov x14, 1", "={x14}"()
  %x15 = call i64 asm sideeffect "mov x15, 1", "={x15}"()
  %x16 = call i64 asm sideeffect "mov x16, 1", "={x16}"()
  %x17 = call i64 asm sideeffect "mov x17, 1", "={x17}"()
  %x18 = call i64 asm sideeffect "mov x18, 1", "={x18}"()
  %x19 = call i64 asm sideeffect "mov x19, 1", "={x19}"()
  %x20 = call i64 asm sideeffect "mov x20, 1", "={x20}"()
  %x21 = call i64 asm sideeffect "mov x21, 1", "={x21}"()
  %x22 = call i64 asm sideeffect "mov x22, 1", "={x22}"()
  %x23 = call i64 asm sideeffect "mov x23, 1", "={x23}"()
  %x24 = call i64 asm sideeffect "mov x24, 1", "={x24}"()
  %x25 = call i64 asm sideeffect "mov x25, 1", "={x25}"()
  %x26 = call i64 asm sideeffect "mov x26, 1", "={x26}"()
  %x27 = call i64 asm sideeffect "mov x27, 1", "={x27}"()
  %x28 = call i64 asm sideeffect "mov x28, 1", "={x28}"()

  %cmp = icmp eq i64 %x16, %x15
  br i1 %cmp, label %iftrue, label %iffalse

iftrue:
  call void asm sideeffect ".space 2048", ""()
  ret void

iffalse:
  call void asm sideeffect "# reg use $0", "{x0}"(i64 %x0)
  call void asm sideeffect "# reg use $0", "{x1}"(i64 %x1)
  call void asm sideeffect "# reg use $0", "{x2}"(i64 %x2)
  call void asm sideeffect "# reg use $0", "{x3}"(i64 %x3)
  call void asm sideeffect "# reg use $0", "{x4}"(i64 %x4)
  call void asm sideeffect "# reg use $0", "{x5}"(i64 %x5)
  call void asm sideeffect "# reg use $0", "{x6}"(i64 %x6)
  call void asm sideeffect "# reg use $0", "{x7}"(i64 %x7)
  call void asm sideeffect "# reg use $0", "{x8}"(i64 %x8)
  call void asm sideeffect "# reg use $0", "{x9}"(i64 %x9)
  call void asm sideeffect "# reg use $0", "{x10}"(i64 %x10)
  call void asm sideeffect "# reg use $0", "{x11}"(i64 %x11)
  call void asm sideeffect "# reg use $0", "{x12}"(i64 %x12)
  call void asm sideeffect "# reg use $0", "{x13}"(i64 %x13)
  call void asm sideeffect "# reg use $0", "{x14}"(i64 %x14)
  call void asm sideeffect "# reg use $0", "{x15}"(i64 %x15)
  call void asm sideeffect "# reg use $0", "{x16}"(i64 %x16)
  call void asm sideeffect "# reg use $0", "{x17}"(i64 %x17)
  call void asm sideeffect "# reg use $0", "{x18}"(i64 %x18)
  call void asm sideeffect "# reg use $0", "{x19}"(i64 %x19)
  call void asm sideeffect "# reg use $0", "{x20}"(i64 %x20)
  call void asm sideeffect "# reg use $0", "{x21}"(i64 %x21)
  call void asm sideeffect "# reg use $0", "{x22}"(i64 %x22)
  call void asm sideeffect "# reg use $0", "{x23}"(i64 %x23)
  call void asm sideeffect "# reg use $0", "{x24}"(i64 %x24)
  call void asm sideeffect "# reg use $0", "{x25}"(i64 %x25)
  call void asm sideeffect "# reg use $0", "{x26}"(i64 %x26)
  call void asm sideeffect "# reg use $0", "{x27}"(i64 %x27)
  call void asm sideeffect "# reg use $0", "{x28}"(i64 %x28)
  ret void
}

define void @relax_b_x16_taken() {
; CHECK-LABEL:    relax_b_x16_taken:                      // @relax_b_x16_taken
; COM: Since the source of the out-of-range branch is hot and x16 is
; COM: taken, it makes sense to spill x16 and let the linker insert
; COM: fixup code for this branch rather than inflating the hot code
; COM: size by eagerly relaxing the unconditional branch.
; CHECK:          // %bb.0:                               // %entry
; CHECK-NEXT:             //APP
; CHECK-NEXT:             mov     x16, #1
; CHECK-NEXT:             //NO_APP
; CHECK-NEXT:             cbnz    x16, .LBB2_1
; CHECK-NEXT:     // %bb.3:                               // %entry
; CHECK-NEXT:             str     [[SPILL_REGISTER]], [sp,
; CHECK-SAME:                                                       -16]!
; CHECK-NEXT:             b       .LBB2_4
; CHECK-NEXT:     .LBB2_1:                                // %iftrue
; CHECK-NEXT:             //APP
; CHECK-NEXT:             .zero   2048
; CHECK-NEXT:             //NO_APP
; CHECK-NEXT:             ret
; CHECK-NEXT:     .LBB2_4:                                // %iffalse
; CHECK-NEXT:             ldr     [[SPILL_REGISTER]], [sp], 
; CHECK-SAME:                                                        16
; CHECK-NEXT:     // %bb.2:                               // %iffalse
; CHECK-NEXT:             //APP
; CHECK-NEXT:             // reg use x16
; CHECK-NEXT:             //NO_APP
; CHECK-NEXT:             ret
entry:
  %x16 = call i64 asm sideeffect "mov x16, 1", "={x16}"()

  %cmp = icmp eq i64 %x16, 0
  br i1 %cmp, label %iffalse, label %iftrue

iftrue:
  call void asm sideeffect ".space 2048", ""()
  ret void

iffalse:
  call void asm sideeffect "# reg use $0", "{x16}"(i64 %x16)
  ret void
}

declare i32 @bar()
declare i32 @baz()
