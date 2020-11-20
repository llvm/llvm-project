; RUN: llc -mtriple=aarch64-none-linux-gnu -disable-post-ra -verify-machineinstrs -mattr=+lse < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-none-linux-gnu -disable-post-ra -verify-machineinstrs -mattr=+lse -mattr=+outline-atomics < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-none-linux-gnu -disable-post-ra -verify-machineinstrs -mattr=+outline-atomics < %s | FileCheck %s --check-prefix=OUTLINE-ATOMICS
; RUN: llc -mtriple=aarch64_be-none-linux-gnu -disable-post-ra -verify-machineinstrs -mattr=+lse < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-none-linux-gnu -disable-post-ra -verify-machineinstrs -mattr=+lse < %s | FileCheck %s --check-prefix=CHECK-REG
; RUN: llc -mtriple=aarch64-none-linux-gnu -disable-post-ra -verify-machineinstrs -mcpu=saphira < %s | FileCheck %s

; Point of CHECK-REG is to make sure UNPREDICTABLE instructions aren't created
; (i.e. reusing a register for status & data in store exclusive).
; CHECK-REG-NOT: stlxrb w[[NEW:[0-9]+]], w[[NEW]], [x{{[0-9]+}}]
; CHECK-REG-NOT: stlxrb w[[NEW:[0-9]+]], x[[NEW]], [x{{[0-9]+}}]

@var8 = global i8 0
@var16 = global i16 0
@var32 = global i32 0
@var64 = global i64 0
@var128 = global i128 0

define i8 @test_atomic_load_add_i8(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i8:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_add_i8:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var8
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var8
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd1_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw add i8* @var8, i8 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: ldaddalb w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_load_add_i16(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i16:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_add_i16:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var16
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var16
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd2_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw add i16* @var16, i16 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: ldaddalh w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_load_add_i32(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i32:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_add_i32:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd4_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw add i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldaddal w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_load_add_i64(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i64:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_add_i64:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd8_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw add i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldaddal x[[OLD:[0-9]+]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define void @test_atomic_load_add_i32_noret(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i32_noret:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_add_i32_noret:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd4_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw add i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldaddal w0, w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_load_add_i64_noret(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i64_noret:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_add_i64_noret:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd8_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw add i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldaddal x0, x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define i8 @test_atomic_load_or_i8(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_or_i8:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_or_i8:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var8
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var8
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldset1_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw or i8* @var8, i8 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: ldsetalb w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_load_or_i16(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_or_i16:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_or_i16:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var16
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var16
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldset2_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw or i16* @var16, i16 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: ldsetalh w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_load_or_i32(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_or_i32:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_or_i32:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldset4_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw or i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldsetal w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_load_or_i64(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_or_i64:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_or_i64:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldset8_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw or i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldsetal x[[OLD:[0-9]+]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define void @test_atomic_load_or_i32_noret(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_or_i32_noret:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_or_i32_noret:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldset4_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw or i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldsetal w0, w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_load_or_i64_noret(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_or_i64_noret:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_or_i64_noret:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldset8_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw or i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldsetal x0, x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define i8 @test_atomic_load_xor_i8(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xor_i8:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xor_i8:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var8
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var8
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldeor1_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw xor i8* @var8, i8 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: ldeoralb w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_load_xor_i16(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xor_i16:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xor_i16:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var16
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var16
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldeor2_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw xor i16* @var16, i16 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: ldeoralh w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_load_xor_i32(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xor_i32:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xor_i32:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldeor4_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw xor i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldeoral w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_load_xor_i64(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xor_i64:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xor_i64:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldeor8_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw xor i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldeoral x[[OLD:[0-9]+]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define void @test_atomic_load_xor_i32_noret(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xor_i32_noret:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xor_i32_noret:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldeor4_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw xor i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldeoral w0, w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_load_xor_i64_noret(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xor_i64_noret:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xor_i64_noret:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldeor8_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw xor i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldeoral x0, x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define i8 @test_atomic_load_min_i8(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_min_i8:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_min_i8:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var8
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var8
; OUTLINE-ATOMICS-NEXT:  .LBB18_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxrb w10, [x9]
; OUTLINE-ATOMICS-NEXT:    sxtb w8, w10
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0, sxtb
; OUTLINE-ATOMICS-NEXT:    csel w10, w10, w0, le
; OUTLINE-ATOMICS-NEXT:    stlxrb w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB18_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw min i8* @var8, i8 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: ldsminalb w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_load_min_i16(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_min_i16:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_min_i16:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var16
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var16
; OUTLINE-ATOMICS-NEXT:  .LBB19_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxrh w10, [x9]
; OUTLINE-ATOMICS-NEXT:    sxth w8, w10
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0, sxth
; OUTLINE-ATOMICS-NEXT:    csel w10, w10, w0, le
; OUTLINE-ATOMICS-NEXT:    stlxrh w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB19_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw min i16* @var16, i16 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: ldsminalh w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_load_min_i32(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_min_i32:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_min_i32:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var32
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var32
; OUTLINE-ATOMICS-NEXT:  .LBB20_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr w8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0
; OUTLINE-ATOMICS-NEXT:    csel w10, w8, w0, le
; OUTLINE-ATOMICS-NEXT:    stlxr w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB20_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw min i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldsminal w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_load_min_i64(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_min_i64:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_min_i64:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var64
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var64
; OUTLINE-ATOMICS-NEXT:  .LBB21_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr x8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp x8, x0
; OUTLINE-ATOMICS-NEXT:    csel x10, x8, x0, le
; OUTLINE-ATOMICS-NEXT:    stlxr w11, x10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB21_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov x0, x8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw min i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldsminal x[[OLD:[0-9]+]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define void @test_atomic_load_min_i32_noret(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_min_i32_noret:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_min_i32_noret:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x8, var32
; OUTLINE-ATOMICS-NEXT:    add x8, x8, :lo12:var32
; OUTLINE-ATOMICS-NEXT:  .LBB22_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr w9, [x8]
; OUTLINE-ATOMICS-NEXT:    cmp w9, w0
; OUTLINE-ATOMICS-NEXT:    csel w9, w9, w0, le
; OUTLINE-ATOMICS-NEXT:    stlxr w10, w9, [x8]
; OUTLINE-ATOMICS-NEXT:    cbnz w10, .LBB22_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw min i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldsminal w0, w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_load_min_i64_noret(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_min_i64_noret:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_min_i64_noret:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x8, var64
; OUTLINE-ATOMICS-NEXT:    add x8, x8, :lo12:var64
; OUTLINE-ATOMICS-NEXT:  .LBB23_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr x9, [x8]
; OUTLINE-ATOMICS-NEXT:    cmp x9, x0
; OUTLINE-ATOMICS-NEXT:    csel x9, x9, x0, le
; OUTLINE-ATOMICS-NEXT:    stlxr w10, x9, [x8]
; OUTLINE-ATOMICS-NEXT:    cbnz w10, .LBB23_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw min i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldsminal x0, x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define i8 @test_atomic_load_umin_i8(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umin_i8:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umin_i8:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var8
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var8
; OUTLINE-ATOMICS-NEXT:  .LBB24_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxrb w8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0, uxtb
; OUTLINE-ATOMICS-NEXT:    csel w10, w8, w0, ls
; OUTLINE-ATOMICS-NEXT:    stlxrb w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB24_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw umin i8* @var8, i8 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: lduminalb w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_load_umin_i16(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umin_i16:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umin_i16:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var16
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var16
; OUTLINE-ATOMICS-NEXT:  .LBB25_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxrh w8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0, uxth
; OUTLINE-ATOMICS-NEXT:    csel w10, w8, w0, ls
; OUTLINE-ATOMICS-NEXT:    stlxrh w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB25_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw umin i16* @var16, i16 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: lduminalh w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_load_umin_i32(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umin_i32:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umin_i32:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var32
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var32
; OUTLINE-ATOMICS-NEXT:  .LBB26_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr w8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0
; OUTLINE-ATOMICS-NEXT:    csel w10, w8, w0, ls
; OUTLINE-ATOMICS-NEXT:    stlxr w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB26_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw umin i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: lduminal w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_load_umin_i64(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umin_i64:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umin_i64:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var64
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var64
; OUTLINE-ATOMICS-NEXT:  .LBB27_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr x8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp x8, x0
; OUTLINE-ATOMICS-NEXT:    csel x10, x8, x0, ls
; OUTLINE-ATOMICS-NEXT:    stlxr w11, x10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB27_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov x0, x8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw umin i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: lduminal x[[OLD:[0-9]+]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define void @test_atomic_load_umin_i32_noret(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umin_i32_noret:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umin_i32_noret:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x8, var32
; OUTLINE-ATOMICS-NEXT:    add x8, x8, :lo12:var32
; OUTLINE-ATOMICS-NEXT:  .LBB28_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr w9, [x8]
; OUTLINE-ATOMICS-NEXT:    cmp w9, w0
; OUTLINE-ATOMICS-NEXT:    csel w9, w9, w0, ls
; OUTLINE-ATOMICS-NEXT:    stlxr w10, w9, [x8]
; OUTLINE-ATOMICS-NEXT:    cbnz w10, .LBB28_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw umin i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: lduminal w0, w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_load_umin_i64_noret(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umin_i64_noret:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umin_i64_noret:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x8, var64
; OUTLINE-ATOMICS-NEXT:    add x8, x8, :lo12:var64
; OUTLINE-ATOMICS-NEXT:  .LBB29_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr x9, [x8]
; OUTLINE-ATOMICS-NEXT:    cmp x9, x0
; OUTLINE-ATOMICS-NEXT:    csel x9, x9, x0, ls
; OUTLINE-ATOMICS-NEXT:    stlxr w10, x9, [x8]
; OUTLINE-ATOMICS-NEXT:    cbnz w10, .LBB29_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw umin i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: lduminal x0, x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define i8 @test_atomic_load_max_i8(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_max_i8:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_max_i8:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var8
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var8
; OUTLINE-ATOMICS-NEXT:  .LBB30_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxrb w10, [x9]
; OUTLINE-ATOMICS-NEXT:    sxtb w8, w10
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0, sxtb
; OUTLINE-ATOMICS-NEXT:    csel w10, w10, w0, gt
; OUTLINE-ATOMICS-NEXT:    stlxrb w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB30_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw max i8* @var8, i8 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: ldsmaxalb w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_load_max_i16(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_max_i16:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_max_i16:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var16
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var16
; OUTLINE-ATOMICS-NEXT:  .LBB31_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxrh w10, [x9]
; OUTLINE-ATOMICS-NEXT:    sxth w8, w10
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0, sxth
; OUTLINE-ATOMICS-NEXT:    csel w10, w10, w0, gt
; OUTLINE-ATOMICS-NEXT:    stlxrh w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB31_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw max i16* @var16, i16 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: ldsmaxalh w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_load_max_i32(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_max_i32:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_max_i32:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var32
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var32
; OUTLINE-ATOMICS-NEXT:  .LBB32_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr w8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0
; OUTLINE-ATOMICS-NEXT:    csel w10, w8, w0, gt
; OUTLINE-ATOMICS-NEXT:    stlxr w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB32_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw max i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldsmaxal w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_load_max_i64(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_max_i64:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_max_i64:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var64
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var64
; OUTLINE-ATOMICS-NEXT:  .LBB33_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr x8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp x8, x0
; OUTLINE-ATOMICS-NEXT:    csel x10, x8, x0, gt
; OUTLINE-ATOMICS-NEXT:    stlxr w11, x10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB33_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov x0, x8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw max i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldsmaxal x[[OLD:[0-9]+]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define void @test_atomic_load_max_i32_noret(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_max_i32_noret:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_max_i32_noret:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x8, var32
; OUTLINE-ATOMICS-NEXT:    add x8, x8, :lo12:var32
; OUTLINE-ATOMICS-NEXT:  .LBB34_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr w9, [x8]
; OUTLINE-ATOMICS-NEXT:    cmp w9, w0
; OUTLINE-ATOMICS-NEXT:    csel w9, w9, w0, gt
; OUTLINE-ATOMICS-NEXT:    stlxr w10, w9, [x8]
; OUTLINE-ATOMICS-NEXT:    cbnz w10, .LBB34_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw max i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldsmaxal w0, w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_load_max_i64_noret(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_max_i64_noret:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_max_i64_noret:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x8, var64
; OUTLINE-ATOMICS-NEXT:    add x8, x8, :lo12:var64
; OUTLINE-ATOMICS-NEXT:  .LBB35_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr x9, [x8]
; OUTLINE-ATOMICS-NEXT:    cmp x9, x0
; OUTLINE-ATOMICS-NEXT:    csel x9, x9, x0, gt
; OUTLINE-ATOMICS-NEXT:    stlxr w10, x9, [x8]
; OUTLINE-ATOMICS-NEXT:    cbnz w10, .LBB35_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw max i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldsmaxal x0, x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define i8 @test_atomic_load_umax_i8(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umax_i8:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umax_i8:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var8
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var8
; OUTLINE-ATOMICS-NEXT:  .LBB36_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxrb w8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0, uxtb
; OUTLINE-ATOMICS-NEXT:    csel w10, w8, w0, hi
; OUTLINE-ATOMICS-NEXT:    stlxrb w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB36_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw umax i8* @var8, i8 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: ldumaxalb w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_load_umax_i16(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umax_i16:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umax_i16:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var16
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var16
; OUTLINE-ATOMICS-NEXT:  .LBB37_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxrh w8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0, uxth
; OUTLINE-ATOMICS-NEXT:    csel w10, w8, w0, hi
; OUTLINE-ATOMICS-NEXT:    stlxrh w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB37_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw umax i16* @var16, i16 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: ldumaxalh w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_load_umax_i32(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umax_i32:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umax_i32:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var32
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var32
; OUTLINE-ATOMICS-NEXT:  .LBB38_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr w8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0
; OUTLINE-ATOMICS-NEXT:    csel w10, w8, w0, hi
; OUTLINE-ATOMICS-NEXT:    stlxr w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB38_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw umax i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldumaxal w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_load_umax_i64(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umax_i64:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umax_i64:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var64
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var64
; OUTLINE-ATOMICS-NEXT:  .LBB39_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr x8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp x8, x0
; OUTLINE-ATOMICS-NEXT:    csel x10, x8, x0, hi
; OUTLINE-ATOMICS-NEXT:    stlxr w11, x10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB39_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov x0, x8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw umax i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldumaxal x[[OLD:[0-9]+]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define void @test_atomic_load_umax_i32_noret(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umax_i32_noret:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umax_i32_noret:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x8, var32
; OUTLINE-ATOMICS-NEXT:    add x8, x8, :lo12:var32
; OUTLINE-ATOMICS-NEXT:  .LBB40_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr w9, [x8]
; OUTLINE-ATOMICS-NEXT:    cmp w9, w0
; OUTLINE-ATOMICS-NEXT:    csel w9, w9, w0, hi
; OUTLINE-ATOMICS-NEXT:    stlxr w10, w9, [x8]
; OUTLINE-ATOMICS-NEXT:    cbnz w10, .LBB40_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw umax i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldumaxal w0, w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_load_umax_i64_noret(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umax_i64_noret:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umax_i64_noret:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x8, var64
; OUTLINE-ATOMICS-NEXT:    add x8, x8, :lo12:var64
; OUTLINE-ATOMICS-NEXT:  .LBB41_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr x9, [x8]
; OUTLINE-ATOMICS-NEXT:    cmp x9, x0
; OUTLINE-ATOMICS-NEXT:    csel x9, x9, x0, hi
; OUTLINE-ATOMICS-NEXT:    stlxr w10, x9, [x8]
; OUTLINE-ATOMICS-NEXT:    cbnz w10, .LBB41_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw umax i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldumaxal x0, x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define i8 @test_atomic_load_xchg_i8(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xchg_i8:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xchg_i8:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var8
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var8
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_swp1_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw xchg i8* @var8, i8 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: swpalb w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_load_xchg_i16(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xchg_i16:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xchg_i16:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var16
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var16
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_swp2_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw xchg i16* @var16, i16 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: swpalh w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_load_xchg_i32(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xchg_i32:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xchg_i32:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_swp4_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw xchg i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: swpal w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_load_xchg_i64(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xchg_i64:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xchg_i64:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_swp8_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw xchg i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: swpal x[[OLD:[0-9]+]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define void @test_atomic_load_xchg_i32_noret(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xchg_i32_noret:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xchg_i32_noret:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_swp4_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw xchg i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: swpal w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret void
}

define void @test_atomic_load_xchg_i64_noret(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xchg_i64_noret:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xchg_i64_noret:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_swp8_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw xchg i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: swpal x[[OLD:[0-9]+]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret void
}

define i8 @test_atomic_cmpxchg_i8(i8 %wanted, i8 %new) nounwind {
; CHECK-LABEL: test_atomic_cmpxchg_i8:
; OUTLINE-ATOMICS-LABEL: test_atomic_cmpxchg_i8:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x2, var8
; OUTLINE-ATOMICS-NEXT:    add x2, x2, :lo12:var8
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_cas1_acq
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %pair = cmpxchg i8* @var8, i8 %wanted, i8 %new acquire acquire
   %old = extractvalue { i8, i1 } %pair, 0

; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK-NEXT: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8
; CHECK-NEXT: casab w0, w1, [x[[ADDR]]]
; CHECK-NEXT: ret

   ret i8 %old
}

define i1 @test_atomic_cmpxchg_i8_1(i8 %wanted, i8 %new) nounwind {
; CHECK-LABEL: test_atomic_cmpxchg_i8_1:
; OUTLINE-ATOMICS-LABEL: test_atomic_cmpxchg_i8_1:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    stp x30, x19, [sp, #-16]! // 16-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    mov w19, w0
; OUTLINE-ATOMICS-NEXT:    adrp x2, var8
; OUTLINE-ATOMICS-NEXT:    add x2, x2, :lo12:var8
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_cas1_acq
; OUTLINE-ATOMICS-NEXT:    cmp w0, w19, uxtb
; OUTLINE-ATOMICS-NEXT:    cset w0, eq
; OUTLINE-ATOMICS-NEXT:    ldp x30, x19, [sp], #16 // 16-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %pair = cmpxchg i8* @var8, i8 %wanted, i8 %new acquire acquire
   %success = extractvalue { i8, i1 } %pair, 1

; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: casab w[[NEW:[0-9]+]], w1, [x[[ADDR]]]
; CHECK-NEXT: cmp w[[NEW]], w0, uxtb
; CHECK-NEXT: cset w0, eq
; CHECK-NEXT: ret
   ret i1 %success
}

define i16 @test_atomic_cmpxchg_i16(i16 %wanted, i16 %new) nounwind {
; CHECK-LABEL: test_atomic_cmpxchg_i16:
; OUTLINE-ATOMICS-LABEL: test_atomic_cmpxchg_i16:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x2, var16
; OUTLINE-ATOMICS-NEXT:    add x2, x2, :lo12:var16
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_cas2_acq
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %pair = cmpxchg i16* @var16, i16 %wanted, i16 %new acquire acquire
   %old = extractvalue { i16, i1 } %pair, 0

; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK-NEXT: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16
; CHECK-NEXT: casah w0, w1, [x[[ADDR]]]
; CHECK-NEXT: ret

   ret i16 %old
}

define i1 @test_atomic_cmpxchg_i16_1(i16 %wanted, i16 %new) nounwind {
; CHECK-LABEL: test_atomic_cmpxchg_i16_1:
; OUTLINE-ATOMICS-LABEL: test_atomic_cmpxchg_i16_1:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    stp x30, x19, [sp, #-16]! // 16-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    mov w19, w0
; OUTLINE-ATOMICS-NEXT:    adrp x2, var16
; OUTLINE-ATOMICS-NEXT:    add x2, x2, :lo12:var16
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_cas2_acq
; OUTLINE-ATOMICS-NEXT:    cmp w0, w19, uxth
; OUTLINE-ATOMICS-NEXT:    cset w0, eq
; OUTLINE-ATOMICS-NEXT:    ldp x30, x19, [sp], #16 // 16-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %pair = cmpxchg i16* @var16, i16 %wanted, i16 %new acquire acquire
   %success = extractvalue { i16, i1 } %pair, 1

; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK-NEXT: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: casah w[[NEW:[0-9]+]], w1, [x[[ADDR]]]
; CHECK-NEXT: cmp w[[NEW]], w0, uxth
; CHECK-NEXT: cset w0, eq
; CHECK-NEXT: ret

   ret i1 %success
}

define i32 @test_atomic_cmpxchg_i32(i32 %wanted, i32 %new) nounwind {
; CHECK-LABEL: test_atomic_cmpxchg_i32:
; OUTLINE-ATOMICS-LABEL: test_atomic_cmpxchg_i32:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x2, var32
; OUTLINE-ATOMICS-NEXT:    add x2, x2, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_cas4_acq
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %pair = cmpxchg i32* @var32, i32 %wanted, i32 %new acquire acquire
   %old = extractvalue { i32, i1 } %pair, 0

; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: casa w0, w1, [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_cmpxchg_i64(i64 %wanted, i64 %new) nounwind {
; CHECK-LABEL: test_atomic_cmpxchg_i64:
; OUTLINE-ATOMICS-LABEL: test_atomic_cmpxchg_i64:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x2, var64
; OUTLINE-ATOMICS-NEXT:    add x2, x2, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_cas8_acq
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %pair = cmpxchg i64* @var64, i64 %wanted, i64 %new acquire acquire
   %old = extractvalue { i64, i1 } %pair, 0

; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: casa x0, x1, [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define i128 @test_atomic_cmpxchg_i128(i128 %wanted, i128 %new) nounwind {
; CHECK-LABEL: test_atomic_cmpxchg_i128:
; OUTLINE-ATOMICS-LABEL: test_atomic_cmpxchg_i128:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x4, var128
; OUTLINE-ATOMICS-NEXT:    add x4, x4, :lo12:var128
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_cas16_acq
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %pair = cmpxchg i128* @var128, i128 %wanted, i128 %new acquire acquire
   %old = extractvalue { i128, i1 } %pair, 0

; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var128
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var128

; CHECK: caspa x0, x1, x2, x3, [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i128 %old
}

define i8 @test_atomic_load_sub_i8(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i8:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_sub_i8:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    neg w0, w0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var8
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var8
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd1_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw sub i8* @var8, i8 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: neg w[[NEG:[0-9]+]], w[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: ldaddalb w[[NEG]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

  ret i8 %old
}

define i16 @test_atomic_load_sub_i16(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i16:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_sub_i16:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    neg w0, w0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var16
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var16
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd2_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw sub i16* @var16, i16 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: neg w[[NEG:[0-9]+]], w[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: ldaddalh w[[NEG]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

  ret i16 %old
}

define i32 @test_atomic_load_sub_i32(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i32:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_sub_i32:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    neg w0, w0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd4_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw sub i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: neg w[[NEG:[0-9]+]], w[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldaddal w[[NEG]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

  ret i32 %old
}

define i64 @test_atomic_load_sub_i64(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i64:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_sub_i64:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    neg x0, x0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd8_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw sub i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: neg x[[NEG:[0-9]+]], x[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldaddal x[[NEG]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

  ret i64 %old
}

define void @test_atomic_load_sub_i32_noret(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i32_noret:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_sub_i32_noret:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    neg w0, w0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd4_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  atomicrmw sub i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: neg w[[NEG:[0-9]+]], w[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldaddal w[[NEG]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

  ret void
}

define void @test_atomic_load_sub_i64_noret(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i64_noret:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_sub_i64_noret:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    neg x0, x0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd8_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  atomicrmw sub i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: neg x[[NEG:[0-9]+]], x[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldaddal x[[NEG]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

  ret void
}

define i8 @test_atomic_load_sub_i8_neg_imm() nounwind {
; CHECK-LABEL: test_atomic_load_sub_i8_neg_imm:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_sub_i8_neg_imm:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var8
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var8
; OUTLINE-ATOMICS-NEXT:    mov w0, #1
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd1_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw sub i8* @var8, i8 -1 seq_cst

; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8
; CHECK: mov w[[IMM:[0-9]+]], #1
; CHECK: ldaddalb w[[IMM]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

  ret i8 %old
}

define i16 @test_atomic_load_sub_i16_neg_imm() nounwind {
; CHECK-LABEL: test_atomic_load_sub_i16_neg_imm:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_sub_i16_neg_imm:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var16
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var16
; OUTLINE-ATOMICS-NEXT:    mov w0, #1
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd2_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw sub i16* @var16, i16 -1 seq_cst

; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16
; CHECK: mov w[[IMM:[0-9]+]], #1
; CHECK: ldaddalh w[[IMM]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

  ret i16 %old
}

define i32 @test_atomic_load_sub_i32_neg_imm() nounwind {
; CHECK-LABEL: test_atomic_load_sub_i32_neg_imm:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_sub_i32_neg_imm:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    mov w0, #1
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd4_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw sub i32* @var32, i32 -1 seq_cst

; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32
; CHECK: mov w[[IMM:[0-9]+]], #1
; CHECK: ldaddal w[[IMM]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

  ret i32 %old
}

define i64 @test_atomic_load_sub_i64_neg_imm() nounwind {
; CHECK-LABEL: test_atomic_load_sub_i64_neg_imm:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_sub_i64_neg_imm:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    mov w0, #1
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd8_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw sub i64* @var64, i64 -1 seq_cst

; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64
; CHECK: mov w[[IMM:[0-9]+]], #1
; CHECK: ldaddal x[[IMM]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

  ret i64 %old
}

define i8 @test_atomic_load_sub_i8_neg_arg(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i8_neg_arg:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_sub_i8_neg_arg:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var8
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var8
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd1_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %neg = sub i8 0, %offset
  %old = atomicrmw sub i8* @var8, i8 %neg seq_cst

; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8
; CHECK: ldaddalb w0, w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

  ret i8 %old
}

define i16 @test_atomic_load_sub_i16_neg_arg(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i16_neg_arg:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_sub_i16_neg_arg:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var16
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var16
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd2_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %neg = sub i16 0, %offset
  %old = atomicrmw sub i16* @var16, i16 %neg seq_cst

; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16
; CHECK: ldaddalh w0, w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

  ret i16 %old
}

define i32 @test_atomic_load_sub_i32_neg_arg(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i32_neg_arg:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_sub_i32_neg_arg:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd4_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %neg = sub i32 0, %offset
  %old = atomicrmw sub i32* @var32, i32 %neg seq_cst

; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32
; CHECK: ldaddal w0, w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

  ret i32 %old
}

define i64 @test_atomic_load_sub_i64_neg_arg(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i64_neg_arg:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_sub_i64_neg_arg:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd8_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %neg = sub i64 0, %offset
  %old = atomicrmw sub i64* @var64, i64 %neg seq_cst

; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64
; CHECK: ldaddal x0, x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

  ret i64 %old
}

define i8 @test_atomic_load_and_i8(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i8:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_and_i8:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    mvn w0, w0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var8
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var8
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldclr1_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw and i8* @var8, i8 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: mvn w[[NOT:[0-9]+]], w[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: ldclralb w[[NOT]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret i8 %old
}

define i16 @test_atomic_load_and_i16(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i16:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_and_i16:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    mvn w0, w0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var16
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var16
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldclr2_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw and i16* @var16, i16 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: mvn w[[NOT:[0-9]+]], w[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: ldclralh w[[NOT]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret i16 %old
}

define i32 @test_atomic_load_and_i32(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i32:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_and_i32:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    mvn w0, w0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldclr4_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw and i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: mvn w[[NOT:[0-9]+]], w[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldclral w[[NOT]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret i32 %old
}

define i64 @test_atomic_load_and_i64(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i64:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_and_i64:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    mvn x0, x0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldclr8_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw and i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: mvn x[[NOT:[0-9]+]], x[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldclral x[[NOT]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret i64 %old
}

define i8 @test_atomic_load_and_i8_inv_imm() nounwind {
; CHECK-LABEL: test_atomic_load_and_i8_inv_imm:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_and_i8_inv_imm:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var8
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var8
; OUTLINE-ATOMICS-NEXT:    mov w0, #1
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldclr1_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw and i8* @var8, i8 -2 seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8
; CHECK: mov w[[CONST:[0-9]+]], #1
; CHECK: ldclralb w[[CONST]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret i8 %old
}

define i16 @test_atomic_load_and_i16_inv_imm() nounwind {
; CHECK-LABEL: test_atomic_load_and_i16_inv_imm:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_and_i16_inv_imm:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var16
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var16
; OUTLINE-ATOMICS-NEXT:    mov w0, #1
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldclr2_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw and i16* @var16, i16 -2 seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16
; CHECK: mov w[[CONST:[0-9]+]], #1
; CHECK: ldclralh w[[CONST]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret i16 %old
}

define i32 @test_atomic_load_and_i32_inv_imm() nounwind {
; CHECK-LABEL: test_atomic_load_and_i32_inv_imm:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_and_i32_inv_imm:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    mov w0, #1
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldclr4_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw and i32* @var32, i32 -2 seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32
; CHECK: mov w[[CONST:[0-9]+]], #1
; CHECK: ldclral w[[CONST]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret i32 %old
}

define i64 @test_atomic_load_and_i64_inv_imm() nounwind {
; CHECK-LABEL: test_atomic_load_and_i64_inv_imm:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_and_i64_inv_imm:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    mov w0, #1
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldclr8_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw and i64* @var64, i64 -2 seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64
; CHECK: mov w[[CONST:[0-9]+]], #1
; CHECK: ldclral x[[CONST]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret i64 %old
}

define i8 @test_atomic_load_and_i8_inv_arg(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i8_inv_arg:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_and_i8_inv_arg:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var8
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var8
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldclr1_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %inv = xor i8 %offset, -1
  %old = atomicrmw and i8* @var8, i8 %inv seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8
; CHECK: ldclralb w0, w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret i8 %old
}

define i16 @test_atomic_load_and_i16_inv_arg(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i16_inv_arg:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_and_i16_inv_arg:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var16
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var16
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldclr2_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %inv = xor i16 %offset, -1
  %old = atomicrmw and i16* @var16, i16 %inv seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16
; CHECK: ldclralh w0, w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret i16 %old
}

define i32 @test_atomic_load_and_i32_inv_arg(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i32_inv_arg:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_and_i32_inv_arg:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldclr4_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %inv = xor i32 %offset, -1
  %old = atomicrmw and i32* @var32, i32 %inv seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32
; CHECK: ldclral w0, w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret i32 %old
}

define i64 @test_atomic_load_and_i64_inv_arg(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i64_inv_arg:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_and_i64_inv_arg:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldclr8_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %inv = xor i64 %offset, -1
  %old = atomicrmw and i64* @var64, i64 %inv seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64
; CHECK: ldclral x0, x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret i64 %old
}

define void @test_atomic_load_and_i32_noret(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i32_noret:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_and_i32_noret:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    mvn w0, w0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldclr4_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  atomicrmw and i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: mvn w[[NOT:[0-9]+]], w[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldclral w[[NOT]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_load_and_i64_noret(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i64_noret:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_and_i64_noret:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    mvn x0, x0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldclr8_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  atomicrmw and i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: mvn x[[NOT:[0-9]+]], x[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldclral x[[NOT]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define i8 @test_atomic_load_add_i8_acq_rel(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i8_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_add_i8_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var8
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var8
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd1_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw add i8* @var8, i8 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: ldaddalb w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_load_add_i16_acq_rel(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i16_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_add_i16_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var16
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var16
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd2_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw add i16* @var16, i16 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: ldaddalh w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_load_add_i32_acq_rel(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i32_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_add_i32_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd4_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw add i32* @var32, i32 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldaddal w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_load_add_i64_acq_rel(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i64_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_add_i64_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd8_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw add i64* @var64, i64 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldaddal x[[OLD:[0-9]+]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define void @test_atomic_load_add_i32_noret_acq_rel(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i32_noret_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_add_i32_noret_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd4_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw add i32* @var32, i32 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldaddal w0, w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_load_add_i64_noret_acq_rel(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i64_noret_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_add_i64_noret_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd8_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw add i64* @var64, i64 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldaddal x0, x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define i8 @test_atomic_load_add_i8_acquire(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i8_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_add_i8_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var8
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var8
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd1_acq
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw add i8* @var8, i8 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: ldaddab w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_load_add_i16_acquire(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i16_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_add_i16_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var16
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var16
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd2_acq
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw add i16* @var16, i16 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: ldaddah w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_load_add_i32_acquire(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i32_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_add_i32_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd4_acq
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw add i32* @var32, i32 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldadda w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_load_add_i64_acquire(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i64_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_add_i64_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd8_acq
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw add i64* @var64, i64 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldadda x[[OLD:[0-9]+]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define void @test_atomic_load_add_i32_noret_acquire(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i32_noret_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_add_i32_noret_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd4_acq
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw add i32* @var32, i32 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldadda w0, w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_load_add_i64_noret_acquire(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i64_noret_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_add_i64_noret_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd8_acq
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw add i64* @var64, i64 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldadda x0, x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define i8 @test_atomic_load_add_i8_monotonic(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i8_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_add_i8_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var8
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var8
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd1_relax
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw add i8* @var8, i8 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: ldaddb w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_load_add_i16_monotonic(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i16_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_add_i16_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var16
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var16
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd2_relax
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw add i16* @var16, i16 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: ldaddh w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_load_add_i32_monotonic(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i32_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_add_i32_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd4_relax
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw add i32* @var32, i32 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldadd w[[OLD:[0-9]+]], w[[NEW:[0-9,a-z]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_load_add_i64_monotonic(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i64_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_add_i64_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd8_relax
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw add i64* @var64, i64 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldadd x[[OLD:[0-9]+]], x[[NEW:[0-9,a-z]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define void @test_atomic_load_add_i32_noret_monotonic(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i32_noret_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_add_i32_noret_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd4_relax
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw add i32* @var32, i32 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldadd w{{[0-9]+}}, w{{[1-9][0-9]*}}, [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_load_add_i64_noret_monotonic(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i64_noret_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_add_i64_noret_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd8_relax
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw add i64* @var64, i64 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldadd x{{[0-9]}}, x{{[1-9][0-9]*}}, [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define i8 @test_atomic_load_add_i8_release(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i8_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_add_i8_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var8
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var8
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd1_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw add i8* @var8, i8 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: ldaddlb w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_load_add_i16_release(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i16_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_add_i16_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var16
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var16
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd2_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw add i16* @var16, i16 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: ldaddlh w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_load_add_i32_release(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i32_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_add_i32_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd4_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw add i32* @var32, i32 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldaddl w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_load_add_i64_release(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i64_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_add_i64_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd8_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw add i64* @var64, i64 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldaddl x[[OLD:[0-9]+]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define void @test_atomic_load_add_i32_noret_release(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i32_noret_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_add_i32_noret_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd4_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw add i32* @var32, i32 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldaddl w{{[0-9]+}}, w{{[1-9][0-9]*}}, [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_load_add_i64_noret_release(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i64_noret_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_add_i64_noret_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd8_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw add i64* @var64, i64 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldaddl x{{[0-9]+}}, x{{[1-9][0-9]*}}, [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define i8 @test_atomic_load_add_i8_seq_cst(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i8_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_add_i8_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var8
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var8
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd1_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw add i8* @var8, i8 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: ldaddalb w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_load_add_i16_seq_cst(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i16_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_add_i16_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var16
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var16
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd2_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw add i16* @var16, i16 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: ldaddalh w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_load_add_i32_seq_cst(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i32_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_add_i32_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd4_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw add i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldaddal w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_load_add_i64_seq_cst(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i64_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_add_i64_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd8_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw add i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldaddal x[[OLD:[0-9]+]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define void @test_atomic_load_add_i32_noret_seq_cst(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i32_noret_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_add_i32_noret_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd4_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw add i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldaddal w0, w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_load_add_i64_noret_seq_cst(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i64_noret_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_add_i64_noret_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd8_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw add i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldaddal x0, x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define i8 @test_atomic_load_and_i8_acq_rel(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i8_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_and_i8_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    mvn w0, w0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var8
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var8
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldclr1_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw and i8* @var8, i8 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: mvn w[[NOT:[0-9]+]], w[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: ldclralb w[[NOT]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret i8 %old
}

define i16 @test_atomic_load_and_i16_acq_rel(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i16_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_and_i16_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    mvn w0, w0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var16
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var16
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldclr2_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw and i16* @var16, i16 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: mvn w[[NOT:[0-9]+]], w[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: ldclralh w[[NOT]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret i16 %old
}

define i32 @test_atomic_load_and_i32_acq_rel(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i32_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_and_i32_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    mvn w0, w0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldclr4_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw and i32* @var32, i32 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: mvn w[[NOT:[0-9]+]], w[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldclral w[[NOT]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret i32 %old
}

define i64 @test_atomic_load_and_i64_acq_rel(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i64_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_and_i64_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    mvn x0, x0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldclr8_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw and i64* @var64, i64 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: mvn x[[NOT:[0-9]+]], x[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldclral x[[NOT]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret i64 %old
}

define void @test_atomic_load_and_i32_noret_acq_rel(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i32_noret_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_and_i32_noret_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    mvn w0, w0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldclr4_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  atomicrmw and i32* @var32, i32 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: mvn w[[NOT:[0-9]+]], w[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldclral w[[NOT]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_load_and_i64_noret_acq_rel(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i64_noret_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_and_i64_noret_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    mvn x0, x0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldclr8_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  atomicrmw and i64* @var64, i64 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: mvn x[[NOT:[0-9]+]], x[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldclral x[[NOT]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define i8 @test_atomic_load_and_i8_acquire(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i8_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_and_i8_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    mvn w0, w0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var8
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var8
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldclr1_acq
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw and i8* @var8, i8 %offset acquire
; CHECK-NOT: dmb
; CHECK: mvn w[[NOT:[0-9]+]], w[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: ldclrab w[[NOT]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret i8 %old
}

define i16 @test_atomic_load_and_i16_acquire(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i16_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_and_i16_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    mvn w0, w0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var16
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var16
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldclr2_acq
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw and i16* @var16, i16 %offset acquire
; CHECK-NOT: dmb
; CHECK: mvn w[[NOT:[0-9]+]], w[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: ldclrah w[[NOT]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret i16 %old
}

define i32 @test_atomic_load_and_i32_acquire(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i32_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_and_i32_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    mvn w0, w0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldclr4_acq
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw and i32* @var32, i32 %offset acquire
; CHECK-NOT: dmb
; CHECK: mvn w[[NOT:[0-9]+]], w[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldclra w[[NOT]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret i32 %old
}

define i64 @test_atomic_load_and_i64_acquire(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i64_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_and_i64_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    mvn x0, x0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldclr8_acq
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw and i64* @var64, i64 %offset acquire
; CHECK-NOT: dmb
; CHECK: mvn x[[NOT:[0-9]+]], x[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldclra x[[NOT]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret i64 %old
}

define void @test_atomic_load_and_i32_noret_acquire(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i32_noret_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_and_i32_noret_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    mvn w0, w0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldclr4_acq
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  atomicrmw and i32* @var32, i32 %offset acquire
; CHECK-NOT: dmb
; CHECK: mvn w[[NOT:[0-9]+]], w[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldclra w[[NOT]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_load_and_i64_noret_acquire(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i64_noret_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_and_i64_noret_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    mvn x0, x0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldclr8_acq
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  atomicrmw and i64* @var64, i64 %offset acquire
; CHECK-NOT: dmb
; CHECK: mvn x[[NOT:[0-9]+]], x[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldclra x[[NOT]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define i8 @test_atomic_load_and_i8_monotonic(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i8_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_and_i8_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    mvn w0, w0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var8
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var8
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldclr1_relax
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw and i8* @var8, i8 %offset monotonic
; CHECK-NOT: dmb
; CHECK: mvn w[[NOT:[0-9]+]], w[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: ldclrb w[[NOT]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret i8 %old
}

define i16 @test_atomic_load_and_i16_monotonic(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i16_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_and_i16_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    mvn w0, w0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var16
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var16
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldclr2_relax
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw and i16* @var16, i16 %offset monotonic
; CHECK-NOT: dmb
; CHECK: mvn w[[NOT:[0-9]+]], w[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: ldclrh w[[NOT]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret i16 %old
}

define i32 @test_atomic_load_and_i32_monotonic(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i32_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_and_i32_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    mvn w0, w0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldclr4_relax
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw and i32* @var32, i32 %offset monotonic
; CHECK-NOT: dmb
; CHECK: mvn w[[NOT:[0-9]+]], w[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldclr w[[NOT]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret i32 %old
}

define i64 @test_atomic_load_and_i64_monotonic(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i64_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_and_i64_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    mvn x0, x0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldclr8_relax
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw and i64* @var64, i64 %offset monotonic
; CHECK-NOT: dmb
; CHECK: mvn x[[NOT:[0-9]+]], x[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldclr x[[NOT]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret i64 %old
}

define void @test_atomic_load_and_i32_noret_monotonic(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i32_noret_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_and_i32_noret_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    mvn w0, w0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldclr4_relax
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  atomicrmw and i32* @var32, i32 %offset monotonic
; CHECK-NOT: dmb
; CHECK: mvn w[[NOT:[0-9]+]], w[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldclr w{{[0-9]+}}, w[[NEW:[1-9][0-9]*]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_load_and_i64_noret_monotonic(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i64_noret_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_and_i64_noret_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    mvn x0, x0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldclr8_relax
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  atomicrmw and i64* @var64, i64 %offset monotonic
; CHECK-NOT: dmb
; CHECK: mvn x[[NOT:[0-9]+]], x[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldclr x{{[0-9]+}}, x[[NEW:[1-9][0-9]*]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define i8 @test_atomic_load_and_i8_release(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i8_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_and_i8_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    mvn w0, w0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var8
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var8
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldclr1_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw and i8* @var8, i8 %offset release
; CHECK-NOT: dmb
; CHECK: mvn w[[NOT:[0-9]+]], w[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: ldclrlb w[[NOT]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret i8 %old
}

define i16 @test_atomic_load_and_i16_release(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i16_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_and_i16_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    mvn w0, w0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var16
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var16
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldclr2_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw and i16* @var16, i16 %offset release
; CHECK-NOT: dmb
; CHECK: mvn w[[NOT:[0-9]+]], w[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: ldclrlh w[[NOT]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret i16 %old
}

define i32 @test_atomic_load_and_i32_release(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i32_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_and_i32_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    mvn w0, w0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldclr4_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw and i32* @var32, i32 %offset release
; CHECK-NOT: dmb
; CHECK: mvn w[[NOT:[0-9]+]], w[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldclrl w[[NOT]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret i32 %old
}

define i64 @test_atomic_load_and_i64_release(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i64_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_and_i64_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    mvn x0, x0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldclr8_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw and i64* @var64, i64 %offset release
; CHECK-NOT: dmb
; CHECK: mvn x[[NOT:[0-9]+]], x[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldclrl x[[NOT]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret i64 %old
}

define void @test_atomic_load_and_i32_noret_release(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i32_noret_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_and_i32_noret_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    mvn w0, w0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldclr4_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  atomicrmw and i32* @var32, i32 %offset release
; CHECK-NOT: dmb
; CHECK: mvn w[[NOT:[0-9]+]], w[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldclrl w{{[0-9]*}}, w[[NEW:[1-9][0-9]*]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_load_and_i64_noret_release(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i64_noret_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_and_i64_noret_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    mvn x0, x0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldclr8_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  atomicrmw and i64* @var64, i64 %offset release
; CHECK-NOT: dmb
; CHECK: mvn x[[NOT:[0-9]+]], x[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldclrl x{{[0-9]*}}, x[[NEW:[1-9][0-9]*]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define i8 @test_atomic_load_and_i8_seq_cst(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i8_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_and_i8_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    mvn w0, w0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var8
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var8
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldclr1_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw and i8* @var8, i8 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: mvn w[[NOT:[0-9]+]], w[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: ldclralb w[[NOT]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret i8 %old
}

define i16 @test_atomic_load_and_i16_seq_cst(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i16_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_and_i16_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    mvn w0, w0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var16
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var16
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldclr2_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw and i16* @var16, i16 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: mvn w[[NOT:[0-9]+]], w[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: ldclralh w[[NOT]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret i16 %old
}

define i32 @test_atomic_load_and_i32_seq_cst(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i32_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_and_i32_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    mvn w0, w0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldclr4_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw and i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: mvn w[[NOT:[0-9]+]], w[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldclral w[[NOT]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret i32 %old
}

define i64 @test_atomic_load_and_i64_seq_cst(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i64_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_and_i64_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    mvn x0, x0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldclr8_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw and i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: mvn x[[NOT:[0-9]+]], x[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldclral x[[NOT]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret i64 %old
}

define void @test_atomic_load_and_i32_noret_seq_cst(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i32_noret_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_and_i32_noret_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    mvn w0, w0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldclr4_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  atomicrmw and i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: mvn w[[NOT:[0-9]+]], w[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldclral w[[NOT]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_load_and_i64_noret_seq_cst(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i64_noret_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_and_i64_noret_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    mvn x0, x0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldclr8_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  atomicrmw and i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: mvn x[[NOT:[0-9]+]], x[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldclral x[[NOT]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define i8 @test_atomic_cmpxchg_i8_acquire(i8 %wanted, i8 %new) nounwind {
; CHECK-LABEL: test_atomic_cmpxchg_i8_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_cmpxchg_i8_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x2, var8
; OUTLINE-ATOMICS-NEXT:    add x2, x2, :lo12:var8
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_cas1_acq
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %pair = cmpxchg i8* @var8, i8 %wanted, i8 %new acquire acquire
   %old = extractvalue { i8, i1 } %pair, 0

; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: casab w[[NEW:[0-9]+]], w[[OLD:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_cmpxchg_i16_acquire(i16 %wanted, i16 %new) nounwind {
; CHECK-LABEL: test_atomic_cmpxchg_i16_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_cmpxchg_i16_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x2, var16
; OUTLINE-ATOMICS-NEXT:    add x2, x2, :lo12:var16
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_cas2_acq
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %pair = cmpxchg i16* @var16, i16 %wanted, i16 %new acquire acquire
   %old = extractvalue { i16, i1 } %pair, 0

; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: casah w0, w1, [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_cmpxchg_i32_acquire(i32 %wanted, i32 %new) nounwind {
; CHECK-LABEL: test_atomic_cmpxchg_i32_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_cmpxchg_i32_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x2, var32
; OUTLINE-ATOMICS-NEXT:    add x2, x2, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_cas4_acq
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %pair = cmpxchg i32* @var32, i32 %wanted, i32 %new acquire acquire
   %old = extractvalue { i32, i1 } %pair, 0

; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: casa w0, w1, [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_cmpxchg_i64_acquire(i64 %wanted, i64 %new) nounwind {
; CHECK-LABEL: test_atomic_cmpxchg_i64_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_cmpxchg_i64_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x2, var64
; OUTLINE-ATOMICS-NEXT:    add x2, x2, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_cas8_acq
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %pair = cmpxchg i64* @var64, i64 %wanted, i64 %new acquire acquire
   %old = extractvalue { i64, i1 } %pair, 0

; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: casa x0, x1, [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define i128 @test_atomic_cmpxchg_i128_acquire(i128 %wanted, i128 %new) nounwind {
; CHECK-LABEL: test_atomic_cmpxchg_i128_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_cmpxchg_i128_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x4, var128
; OUTLINE-ATOMICS-NEXT:    add x4, x4, :lo12:var128
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_cas16_acq
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %pair = cmpxchg i128* @var128, i128 %wanted, i128 %new acquire acquire
   %old = extractvalue { i128, i1 } %pair, 0

; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var128
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var128

; CHECK: caspa x0, x1, x2, x3, [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i128 %old
}

define i8 @test_atomic_cmpxchg_i8_monotonic(i8 %wanted, i8 %new) nounwind {
; CHECK-LABEL: test_atomic_cmpxchg_i8_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_cmpxchg_i8_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x2, var8
; OUTLINE-ATOMICS-NEXT:    add x2, x2, :lo12:var8
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_cas1_relax
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %pair = cmpxchg i8* @var8, i8 %wanted, i8 %new monotonic monotonic
   %old = extractvalue { i8, i1 } %pair, 0

; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: casb w[[NEW:[0-9]+]], w[[OLD:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_cmpxchg_i16_monotonic(i16 %wanted, i16 %new) nounwind {
; CHECK-LABEL: test_atomic_cmpxchg_i16_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_cmpxchg_i16_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x2, var16
; OUTLINE-ATOMICS-NEXT:    add x2, x2, :lo12:var16
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_cas2_relax
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %pair = cmpxchg i16* @var16, i16 %wanted, i16 %new monotonic monotonic
   %old = extractvalue { i16, i1 } %pair, 0

; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: cash w0, w1, [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_cmpxchg_i32_monotonic(i32 %wanted, i32 %new) nounwind {
; CHECK-LABEL: test_atomic_cmpxchg_i32_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_cmpxchg_i32_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x2, var32
; OUTLINE-ATOMICS-NEXT:    add x2, x2, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_cas4_relax
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %pair = cmpxchg i32* @var32, i32 %wanted, i32 %new monotonic monotonic
   %old = extractvalue { i32, i1 } %pair, 0

; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: cas w0, w1, [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_cmpxchg_i64_monotonic(i64 %wanted, i64 %new) nounwind {
; CHECK-LABEL: test_atomic_cmpxchg_i64_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_cmpxchg_i64_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x2, var64
; OUTLINE-ATOMICS-NEXT:    add x2, x2, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_cas8_relax
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %pair = cmpxchg i64* @var64, i64 %wanted, i64 %new monotonic monotonic
   %old = extractvalue { i64, i1 } %pair, 0

; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: cas x0, x1, [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define i128 @test_atomic_cmpxchg_i128_monotonic(i128 %wanted, i128 %new) nounwind {
; CHECK-LABEL: test_atomic_cmpxchg_i128_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_cmpxchg_i128_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x4, var128
; OUTLINE-ATOMICS-NEXT:    add x4, x4, :lo12:var128
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_cas16_relax
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %pair = cmpxchg i128* @var128, i128 %wanted, i128 %new monotonic monotonic
   %old = extractvalue { i128, i1 } %pair, 0

; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var128
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var128

; CHECK: casp x0, x1, x2, x3, [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i128 %old
}

define i8 @test_atomic_cmpxchg_i8_seq_cst(i8 %wanted, i8 %new) nounwind {
; CHECK-LABEL: test_atomic_cmpxchg_i8_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_cmpxchg_i8_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x2, var8
; OUTLINE-ATOMICS-NEXT:    add x2, x2, :lo12:var8
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_cas1_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %pair = cmpxchg i8* @var8, i8 %wanted, i8 %new seq_cst seq_cst
   %old = extractvalue { i8, i1 } %pair, 0

; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: casalb w[[NEW:[0-9]+]], w[[OLD:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_cmpxchg_i16_seq_cst(i16 %wanted, i16 %new) nounwind {
; CHECK-LABEL: test_atomic_cmpxchg_i16_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_cmpxchg_i16_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x2, var16
; OUTLINE-ATOMICS-NEXT:    add x2, x2, :lo12:var16
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_cas2_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %pair = cmpxchg i16* @var16, i16 %wanted, i16 %new seq_cst seq_cst
   %old = extractvalue { i16, i1 } %pair, 0

; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: casalh w0, w1, [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_cmpxchg_i32_seq_cst(i32 %wanted, i32 %new) nounwind {
; CHECK-LABEL: test_atomic_cmpxchg_i32_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_cmpxchg_i32_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x2, var32
; OUTLINE-ATOMICS-NEXT:    add x2, x2, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_cas4_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %pair = cmpxchg i32* @var32, i32 %wanted, i32 %new seq_cst seq_cst
   %old = extractvalue { i32, i1 } %pair, 0

; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: casal w0, w1, [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_cmpxchg_i64_seq_cst(i64 %wanted, i64 %new) nounwind {
; CHECK-LABEL: test_atomic_cmpxchg_i64_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_cmpxchg_i64_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x2, var64
; OUTLINE-ATOMICS-NEXT:    add x2, x2, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_cas8_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %pair = cmpxchg i64* @var64, i64 %wanted, i64 %new seq_cst seq_cst
   %old = extractvalue { i64, i1 } %pair, 0

; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: casal x0, x1, [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define i128 @test_atomic_cmpxchg_i128_seq_cst(i128 %wanted, i128 %new) nounwind {
; CHECK-LABEL: test_atomic_cmpxchg_i128_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_cmpxchg_i128_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x4, var128
; OUTLINE-ATOMICS-NEXT:    add x4, x4, :lo12:var128
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_cas16_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %pair = cmpxchg i128* @var128, i128 %wanted, i128 %new seq_cst seq_cst
   %old = extractvalue { i128, i1 } %pair, 0

; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var128
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var128

; CHECK: caspal x0, x1, x2, x3, [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i128 %old
}

define i8 @test_atomic_load_max_i8_acq_rel(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_max_i8_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_max_i8_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var8
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var8
; OUTLINE-ATOMICS-NEXT:  .LBB158_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxrb w10, [x9]
; OUTLINE-ATOMICS-NEXT:    sxtb w8, w10
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0, sxtb
; OUTLINE-ATOMICS-NEXT:    csel w10, w10, w0, gt
; OUTLINE-ATOMICS-NEXT:    stlxrb w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB158_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw max i8* @var8, i8 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: ldsmaxalb w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_load_max_i16_acq_rel(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_max_i16_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_max_i16_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var16
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var16
; OUTLINE-ATOMICS-NEXT:  .LBB159_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxrh w10, [x9]
; OUTLINE-ATOMICS-NEXT:    sxth w8, w10
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0, sxth
; OUTLINE-ATOMICS-NEXT:    csel w10, w10, w0, gt
; OUTLINE-ATOMICS-NEXT:    stlxrh w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB159_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw max i16* @var16, i16 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: ldsmaxalh w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_load_max_i32_acq_rel(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_max_i32_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_max_i32_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var32
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var32
; OUTLINE-ATOMICS-NEXT:  .LBB160_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr w8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0
; OUTLINE-ATOMICS-NEXT:    csel w10, w8, w0, gt
; OUTLINE-ATOMICS-NEXT:    stlxr w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB160_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw max i32* @var32, i32 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldsmaxal w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_load_max_i64_acq_rel(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_max_i64_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_max_i64_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var64
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var64
; OUTLINE-ATOMICS-NEXT:  .LBB161_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr x8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp x8, x0
; OUTLINE-ATOMICS-NEXT:    csel x10, x8, x0, gt
; OUTLINE-ATOMICS-NEXT:    stlxr w11, x10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB161_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov x0, x8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw max i64* @var64, i64 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldsmaxal x[[OLD:[0-9]+]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define void @test_atomic_load_max_i32_noret_acq_rel(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_max_i32_noret_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_max_i32_noret_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x8, var32
; OUTLINE-ATOMICS-NEXT:    add x8, x8, :lo12:var32
; OUTLINE-ATOMICS-NEXT:  .LBB162_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr w9, [x8]
; OUTLINE-ATOMICS-NEXT:    cmp w9, w0
; OUTLINE-ATOMICS-NEXT:    csel w9, w9, w0, gt
; OUTLINE-ATOMICS-NEXT:    stlxr w10, w9, [x8]
; OUTLINE-ATOMICS-NEXT:    cbnz w10, .LBB162_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw max i32* @var32, i32 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldsmaxal w0, w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_load_max_i64_noret_acq_rel(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_max_i64_noret_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_max_i64_noret_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x8, var64
; OUTLINE-ATOMICS-NEXT:    add x8, x8, :lo12:var64
; OUTLINE-ATOMICS-NEXT:  .LBB163_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr x9, [x8]
; OUTLINE-ATOMICS-NEXT:    cmp x9, x0
; OUTLINE-ATOMICS-NEXT:    csel x9, x9, x0, gt
; OUTLINE-ATOMICS-NEXT:    stlxr w10, x9, [x8]
; OUTLINE-ATOMICS-NEXT:    cbnz w10, .LBB163_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw max i64* @var64, i64 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldsmaxal x0, x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define i8 @test_atomic_load_max_i8_acquire(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_max_i8_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_max_i8_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var8
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var8
; OUTLINE-ATOMICS-NEXT:  .LBB164_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxrb w10, [x9]
; OUTLINE-ATOMICS-NEXT:    sxtb w8, w10
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0, sxtb
; OUTLINE-ATOMICS-NEXT:    csel w10, w10, w0, gt
; OUTLINE-ATOMICS-NEXT:    stxrb w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB164_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw max i8* @var8, i8 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: ldsmaxab w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_load_max_i16_acquire(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_max_i16_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_max_i16_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var16
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var16
; OUTLINE-ATOMICS-NEXT:  .LBB165_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxrh w10, [x9]
; OUTLINE-ATOMICS-NEXT:    sxth w8, w10
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0, sxth
; OUTLINE-ATOMICS-NEXT:    csel w10, w10, w0, gt
; OUTLINE-ATOMICS-NEXT:    stxrh w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB165_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw max i16* @var16, i16 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: ldsmaxah w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_load_max_i32_acquire(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_max_i32_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_max_i32_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var32
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var32
; OUTLINE-ATOMICS-NEXT:  .LBB166_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr w8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0
; OUTLINE-ATOMICS-NEXT:    csel w10, w8, w0, gt
; OUTLINE-ATOMICS-NEXT:    stxr w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB166_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw max i32* @var32, i32 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldsmaxa w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_load_max_i64_acquire(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_max_i64_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_max_i64_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var64
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var64
; OUTLINE-ATOMICS-NEXT:  .LBB167_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr x8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp x8, x0
; OUTLINE-ATOMICS-NEXT:    csel x10, x8, x0, gt
; OUTLINE-ATOMICS-NEXT:    stxr w11, x10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB167_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov x0, x8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw max i64* @var64, i64 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldsmaxa x[[OLD:[0-9]+]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define void @test_atomic_load_max_i32_noret_acquire(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_max_i32_noret_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_max_i32_noret_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x8, var32
; OUTLINE-ATOMICS-NEXT:    add x8, x8, :lo12:var32
; OUTLINE-ATOMICS-NEXT:  .LBB168_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr w9, [x8]
; OUTLINE-ATOMICS-NEXT:    cmp w9, w0
; OUTLINE-ATOMICS-NEXT:    csel w9, w9, w0, gt
; OUTLINE-ATOMICS-NEXT:    stxr w10, w9, [x8]
; OUTLINE-ATOMICS-NEXT:    cbnz w10, .LBB168_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw max i32* @var32, i32 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldsmaxa w0, w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_load_max_i64_noret_acquire(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_max_i64_noret_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_max_i64_noret_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x8, var64
; OUTLINE-ATOMICS-NEXT:    add x8, x8, :lo12:var64
; OUTLINE-ATOMICS-NEXT:  .LBB169_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr x9, [x8]
; OUTLINE-ATOMICS-NEXT:    cmp x9, x0
; OUTLINE-ATOMICS-NEXT:    csel x9, x9, x0, gt
; OUTLINE-ATOMICS-NEXT:    stxr w10, x9, [x8]
; OUTLINE-ATOMICS-NEXT:    cbnz w10, .LBB169_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw max i64* @var64, i64 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldsmaxa x0, x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define i8 @test_atomic_load_max_i8_monotonic(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_max_i8_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_max_i8_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var8
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var8
; OUTLINE-ATOMICS-NEXT:  .LBB170_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldxrb w10, [x9]
; OUTLINE-ATOMICS-NEXT:    sxtb w8, w10
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0, sxtb
; OUTLINE-ATOMICS-NEXT:    csel w10, w10, w0, gt
; OUTLINE-ATOMICS-NEXT:    stxrb w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB170_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw max i8* @var8, i8 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: ldsmaxb w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_load_max_i16_monotonic(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_max_i16_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_max_i16_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var16
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var16
; OUTLINE-ATOMICS-NEXT:  .LBB171_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldxrh w10, [x9]
; OUTLINE-ATOMICS-NEXT:    sxth w8, w10
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0, sxth
; OUTLINE-ATOMICS-NEXT:    csel w10, w10, w0, gt
; OUTLINE-ATOMICS-NEXT:    stxrh w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB171_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw max i16* @var16, i16 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: ldsmaxh w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_load_max_i32_monotonic(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_max_i32_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_max_i32_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var32
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var32
; OUTLINE-ATOMICS-NEXT:  .LBB172_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldxr w8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0
; OUTLINE-ATOMICS-NEXT:    csel w10, w8, w0, gt
; OUTLINE-ATOMICS-NEXT:    stxr w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB172_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw max i32* @var32, i32 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldsmax w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_load_max_i64_monotonic(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_max_i64_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_max_i64_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var64
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var64
; OUTLINE-ATOMICS-NEXT:  .LBB173_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldxr x8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp x8, x0
; OUTLINE-ATOMICS-NEXT:    csel x10, x8, x0, gt
; OUTLINE-ATOMICS-NEXT:    stxr w11, x10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB173_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov x0, x8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw max i64* @var64, i64 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldsmax x[[OLD:[0-9]+]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define void @test_atomic_load_max_i32_noret_monotonic(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_max_i32_noret_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_max_i32_noret_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x8, var32
; OUTLINE-ATOMICS-NEXT:    add x8, x8, :lo12:var32
; OUTLINE-ATOMICS-NEXT:  .LBB174_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldxr w9, [x8]
; OUTLINE-ATOMICS-NEXT:    cmp w9, w0
; OUTLINE-ATOMICS-NEXT:    csel w9, w9, w0, gt
; OUTLINE-ATOMICS-NEXT:    stxr w10, w9, [x8]
; OUTLINE-ATOMICS-NEXT:    cbnz w10, .LBB174_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw max i32* @var32, i32 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldsmax w{{[0-9]+}}, w{{[1-9][0-9]*}}, [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_load_max_i64_noret_monotonic(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_max_i64_noret_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_max_i64_noret_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x8, var64
; OUTLINE-ATOMICS-NEXT:    add x8, x8, :lo12:var64
; OUTLINE-ATOMICS-NEXT:  .LBB175_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldxr x9, [x8]
; OUTLINE-ATOMICS-NEXT:    cmp x9, x0
; OUTLINE-ATOMICS-NEXT:    csel x9, x9, x0, gt
; OUTLINE-ATOMICS-NEXT:    stxr w10, x9, [x8]
; OUTLINE-ATOMICS-NEXT:    cbnz w10, .LBB175_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw max i64* @var64, i64 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldsmax x{{[0-9]+}}, x{{[1-9][0-9]*}}, [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define i8 @test_atomic_load_max_i8_release(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_max_i8_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_max_i8_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var8
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var8
; OUTLINE-ATOMICS-NEXT:  .LBB176_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldxrb w10, [x9]
; OUTLINE-ATOMICS-NEXT:    sxtb w8, w10
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0, sxtb
; OUTLINE-ATOMICS-NEXT:    csel w10, w10, w0, gt
; OUTLINE-ATOMICS-NEXT:    stlxrb w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB176_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw max i8* @var8, i8 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: ldsmaxlb w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_load_max_i16_release(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_max_i16_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_max_i16_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var16
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var16
; OUTLINE-ATOMICS-NEXT:  .LBB177_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldxrh w10, [x9]
; OUTLINE-ATOMICS-NEXT:    sxth w8, w10
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0, sxth
; OUTLINE-ATOMICS-NEXT:    csel w10, w10, w0, gt
; OUTLINE-ATOMICS-NEXT:    stlxrh w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB177_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw max i16* @var16, i16 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: ldsmaxlh w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_load_max_i32_release(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_max_i32_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_max_i32_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var32
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var32
; OUTLINE-ATOMICS-NEXT:  .LBB178_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldxr w8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0
; OUTLINE-ATOMICS-NEXT:    csel w10, w8, w0, gt
; OUTLINE-ATOMICS-NEXT:    stlxr w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB178_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw max i32* @var32, i32 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldsmaxl w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_load_max_i64_release(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_max_i64_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_max_i64_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var64
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var64
; OUTLINE-ATOMICS-NEXT:  .LBB179_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldxr x8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp x8, x0
; OUTLINE-ATOMICS-NEXT:    csel x10, x8, x0, gt
; OUTLINE-ATOMICS-NEXT:    stlxr w11, x10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB179_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov x0, x8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw max i64* @var64, i64 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldsmaxl x[[OLD:[0-9]+]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define void @test_atomic_load_max_i32_noret_release(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_max_i32_noret_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_max_i32_noret_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x8, var32
; OUTLINE-ATOMICS-NEXT:    add x8, x8, :lo12:var32
; OUTLINE-ATOMICS-NEXT:  .LBB180_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldxr w9, [x8]
; OUTLINE-ATOMICS-NEXT:    cmp w9, w0
; OUTLINE-ATOMICS-NEXT:    csel w9, w9, w0, gt
; OUTLINE-ATOMICS-NEXT:    stlxr w10, w9, [x8]
; OUTLINE-ATOMICS-NEXT:    cbnz w10, .LBB180_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw max i32* @var32, i32 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldsmaxl w{{[0-9]+}}, w{{[1-9][0-9]*}}, [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_load_max_i64_noret_release(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_max_i64_noret_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_max_i64_noret_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x8, var64
; OUTLINE-ATOMICS-NEXT:    add x8, x8, :lo12:var64
; OUTLINE-ATOMICS-NEXT:  .LBB181_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldxr x9, [x8]
; OUTLINE-ATOMICS-NEXT:    cmp x9, x0
; OUTLINE-ATOMICS-NEXT:    csel x9, x9, x0, gt
; OUTLINE-ATOMICS-NEXT:    stlxr w10, x9, [x8]
; OUTLINE-ATOMICS-NEXT:    cbnz w10, .LBB181_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw max i64* @var64, i64 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldsmaxl x{{[0-9]+}}, x{{[1-9][0-9]*}}, [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define i8 @test_atomic_load_max_i8_seq_cst(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_max_i8_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_max_i8_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var8
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var8
; OUTLINE-ATOMICS-NEXT:  .LBB182_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxrb w10, [x9]
; OUTLINE-ATOMICS-NEXT:    sxtb w8, w10
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0, sxtb
; OUTLINE-ATOMICS-NEXT:    csel w10, w10, w0, gt
; OUTLINE-ATOMICS-NEXT:    stlxrb w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB182_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw max i8* @var8, i8 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: ldsmaxalb w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_load_max_i16_seq_cst(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_max_i16_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_max_i16_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var16
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var16
; OUTLINE-ATOMICS-NEXT:  .LBB183_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxrh w10, [x9]
; OUTLINE-ATOMICS-NEXT:    sxth w8, w10
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0, sxth
; OUTLINE-ATOMICS-NEXT:    csel w10, w10, w0, gt
; OUTLINE-ATOMICS-NEXT:    stlxrh w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB183_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw max i16* @var16, i16 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: ldsmaxalh w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_load_max_i32_seq_cst(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_max_i32_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_max_i32_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var32
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var32
; OUTLINE-ATOMICS-NEXT:  .LBB184_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr w8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0
; OUTLINE-ATOMICS-NEXT:    csel w10, w8, w0, gt
; OUTLINE-ATOMICS-NEXT:    stlxr w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB184_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw max i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldsmaxal w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_load_max_i64_seq_cst(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_max_i64_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_max_i64_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var64
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var64
; OUTLINE-ATOMICS-NEXT:  .LBB185_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr x8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp x8, x0
; OUTLINE-ATOMICS-NEXT:    csel x10, x8, x0, gt
; OUTLINE-ATOMICS-NEXT:    stlxr w11, x10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB185_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov x0, x8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw max i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldsmaxal x[[OLD:[0-9]+]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define void @test_atomic_load_max_i32_noret_seq_cst(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_max_i32_noret_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_max_i32_noret_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x8, var32
; OUTLINE-ATOMICS-NEXT:    add x8, x8, :lo12:var32
; OUTLINE-ATOMICS-NEXT:  .LBB186_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr w9, [x8]
; OUTLINE-ATOMICS-NEXT:    cmp w9, w0
; OUTLINE-ATOMICS-NEXT:    csel w9, w9, w0, gt
; OUTLINE-ATOMICS-NEXT:    stlxr w10, w9, [x8]
; OUTLINE-ATOMICS-NEXT:    cbnz w10, .LBB186_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw max i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldsmaxal w0, w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_load_max_i64_noret_seq_cst(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_max_i64_noret_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_max_i64_noret_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x8, var64
; OUTLINE-ATOMICS-NEXT:    add x8, x8, :lo12:var64
; OUTLINE-ATOMICS-NEXT:  .LBB187_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr x9, [x8]
; OUTLINE-ATOMICS-NEXT:    cmp x9, x0
; OUTLINE-ATOMICS-NEXT:    csel x9, x9, x0, gt
; OUTLINE-ATOMICS-NEXT:    stlxr w10, x9, [x8]
; OUTLINE-ATOMICS-NEXT:    cbnz w10, .LBB187_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw max i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldsmaxal x0, x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define i8 @test_atomic_load_min_i8_acq_rel(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_min_i8_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_min_i8_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var8
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var8
; OUTLINE-ATOMICS-NEXT:  .LBB188_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxrb w10, [x9]
; OUTLINE-ATOMICS-NEXT:    sxtb w8, w10
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0, sxtb
; OUTLINE-ATOMICS-NEXT:    csel w10, w10, w0, le
; OUTLINE-ATOMICS-NEXT:    stlxrb w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB188_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw min i8* @var8, i8 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: ldsminalb w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_load_min_i16_acq_rel(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_min_i16_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_min_i16_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var16
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var16
; OUTLINE-ATOMICS-NEXT:  .LBB189_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxrh w10, [x9]
; OUTLINE-ATOMICS-NEXT:    sxth w8, w10
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0, sxth
; OUTLINE-ATOMICS-NEXT:    csel w10, w10, w0, le
; OUTLINE-ATOMICS-NEXT:    stlxrh w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB189_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw min i16* @var16, i16 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: ldsminalh w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_load_min_i32_acq_rel(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_min_i32_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_min_i32_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var32
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var32
; OUTLINE-ATOMICS-NEXT:  .LBB190_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr w8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0
; OUTLINE-ATOMICS-NEXT:    csel w10, w8, w0, le
; OUTLINE-ATOMICS-NEXT:    stlxr w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB190_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw min i32* @var32, i32 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldsminal w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_load_min_i64_acq_rel(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_min_i64_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_min_i64_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var64
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var64
; OUTLINE-ATOMICS-NEXT:  .LBB191_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr x8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp x8, x0
; OUTLINE-ATOMICS-NEXT:    csel x10, x8, x0, le
; OUTLINE-ATOMICS-NEXT:    stlxr w11, x10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB191_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov x0, x8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw min i64* @var64, i64 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldsminal x[[OLD:[0-9]+]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define void @test_atomic_load_min_i32_noret_acq_rel(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_min_i32_noret_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_min_i32_noret_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x8, var32
; OUTLINE-ATOMICS-NEXT:    add x8, x8, :lo12:var32
; OUTLINE-ATOMICS-NEXT:  .LBB192_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr w9, [x8]
; OUTLINE-ATOMICS-NEXT:    cmp w9, w0
; OUTLINE-ATOMICS-NEXT:    csel w9, w9, w0, le
; OUTLINE-ATOMICS-NEXT:    stlxr w10, w9, [x8]
; OUTLINE-ATOMICS-NEXT:    cbnz w10, .LBB192_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw min i32* @var32, i32 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldsminal w0, w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_load_min_i64_noret_acq_rel(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_min_i64_noret_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_min_i64_noret_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x8, var64
; OUTLINE-ATOMICS-NEXT:    add x8, x8, :lo12:var64
; OUTLINE-ATOMICS-NEXT:  .LBB193_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr x9, [x8]
; OUTLINE-ATOMICS-NEXT:    cmp x9, x0
; OUTLINE-ATOMICS-NEXT:    csel x9, x9, x0, le
; OUTLINE-ATOMICS-NEXT:    stlxr w10, x9, [x8]
; OUTLINE-ATOMICS-NEXT:    cbnz w10, .LBB193_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw min i64* @var64, i64 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldsminal x0, x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define i8 @test_atomic_load_min_i8_acquire(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_min_i8_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_min_i8_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var8
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var8
; OUTLINE-ATOMICS-NEXT:  .LBB194_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxrb w10, [x9]
; OUTLINE-ATOMICS-NEXT:    sxtb w8, w10
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0, sxtb
; OUTLINE-ATOMICS-NEXT:    csel w10, w10, w0, le
; OUTLINE-ATOMICS-NEXT:    stxrb w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB194_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw min i8* @var8, i8 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: ldsminab w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_load_min_i16_acquire(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_min_i16_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_min_i16_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var16
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var16
; OUTLINE-ATOMICS-NEXT:  .LBB195_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxrh w10, [x9]
; OUTLINE-ATOMICS-NEXT:    sxth w8, w10
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0, sxth
; OUTLINE-ATOMICS-NEXT:    csel w10, w10, w0, le
; OUTLINE-ATOMICS-NEXT:    stxrh w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB195_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw min i16* @var16, i16 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: ldsminah w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_load_min_i32_acquire(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_min_i32_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_min_i32_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var32
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var32
; OUTLINE-ATOMICS-NEXT:  .LBB196_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr w8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0
; OUTLINE-ATOMICS-NEXT:    csel w10, w8, w0, le
; OUTLINE-ATOMICS-NEXT:    stxr w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB196_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw min i32* @var32, i32 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldsmina w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_load_min_i64_acquire(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_min_i64_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_min_i64_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var64
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var64
; OUTLINE-ATOMICS-NEXT:  .LBB197_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr x8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp x8, x0
; OUTLINE-ATOMICS-NEXT:    csel x10, x8, x0, le
; OUTLINE-ATOMICS-NEXT:    stxr w11, x10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB197_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov x0, x8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw min i64* @var64, i64 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldsmina x[[OLD:[0-9]+]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define void @test_atomic_load_min_i32_noret_acquire(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_min_i32_noret_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_min_i32_noret_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x8, var32
; OUTLINE-ATOMICS-NEXT:    add x8, x8, :lo12:var32
; OUTLINE-ATOMICS-NEXT:  .LBB198_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr w9, [x8]
; OUTLINE-ATOMICS-NEXT:    cmp w9, w0
; OUTLINE-ATOMICS-NEXT:    csel w9, w9, w0, le
; OUTLINE-ATOMICS-NEXT:    stxr w10, w9, [x8]
; OUTLINE-ATOMICS-NEXT:    cbnz w10, .LBB198_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw min i32* @var32, i32 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldsmina w0, w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_load_min_i64_noret_acquire(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_min_i64_noret_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_min_i64_noret_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x8, var64
; OUTLINE-ATOMICS-NEXT:    add x8, x8, :lo12:var64
; OUTLINE-ATOMICS-NEXT:  .LBB199_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr x9, [x8]
; OUTLINE-ATOMICS-NEXT:    cmp x9, x0
; OUTLINE-ATOMICS-NEXT:    csel x9, x9, x0, le
; OUTLINE-ATOMICS-NEXT:    stxr w10, x9, [x8]
; OUTLINE-ATOMICS-NEXT:    cbnz w10, .LBB199_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw min i64* @var64, i64 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldsmina x0, x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define i8 @test_atomic_load_min_i8_monotonic(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_min_i8_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_min_i8_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var8
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var8
; OUTLINE-ATOMICS-NEXT:  .LBB200_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldxrb w10, [x9]
; OUTLINE-ATOMICS-NEXT:    sxtb w8, w10
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0, sxtb
; OUTLINE-ATOMICS-NEXT:    csel w10, w10, w0, le
; OUTLINE-ATOMICS-NEXT:    stxrb w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB200_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw min i8* @var8, i8 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: ldsminb w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_load_min_i16_monotonic(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_min_i16_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_min_i16_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var16
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var16
; OUTLINE-ATOMICS-NEXT:  .LBB201_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldxrh w10, [x9]
; OUTLINE-ATOMICS-NEXT:    sxth w8, w10
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0, sxth
; OUTLINE-ATOMICS-NEXT:    csel w10, w10, w0, le
; OUTLINE-ATOMICS-NEXT:    stxrh w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB201_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw min i16* @var16, i16 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: ldsminh w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_load_min_i32_monotonic(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_min_i32_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_min_i32_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var32
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var32
; OUTLINE-ATOMICS-NEXT:  .LBB202_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldxr w8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0
; OUTLINE-ATOMICS-NEXT:    csel w10, w8, w0, le
; OUTLINE-ATOMICS-NEXT:    stxr w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB202_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw min i32* @var32, i32 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldsmin w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_load_min_i64_monotonic(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_min_i64_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_min_i64_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var64
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var64
; OUTLINE-ATOMICS-NEXT:  .LBB203_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldxr x8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp x8, x0
; OUTLINE-ATOMICS-NEXT:    csel x10, x8, x0, le
; OUTLINE-ATOMICS-NEXT:    stxr w11, x10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB203_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov x0, x8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw min i64* @var64, i64 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldsmin x[[OLD:[0-9]+]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define void @test_atomic_load_min_i32_noret_monotonic(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_min_i32_noret_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_min_i32_noret_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x8, var32
; OUTLINE-ATOMICS-NEXT:    add x8, x8, :lo12:var32
; OUTLINE-ATOMICS-NEXT:  .LBB204_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldxr w9, [x8]
; OUTLINE-ATOMICS-NEXT:    cmp w9, w0
; OUTLINE-ATOMICS-NEXT:    csel w9, w9, w0, le
; OUTLINE-ATOMICS-NEXT:    stxr w10, w9, [x8]
; OUTLINE-ATOMICS-NEXT:    cbnz w10, .LBB204_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw min i32* @var32, i32 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldsmin w{{[0-9]+}}, w{{[1-9][0-9]*}}, [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_load_min_i64_noret_monotonic(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_min_i64_noret_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_min_i64_noret_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x8, var64
; OUTLINE-ATOMICS-NEXT:    add x8, x8, :lo12:var64
; OUTLINE-ATOMICS-NEXT:  .LBB205_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldxr x9, [x8]
; OUTLINE-ATOMICS-NEXT:    cmp x9, x0
; OUTLINE-ATOMICS-NEXT:    csel x9, x9, x0, le
; OUTLINE-ATOMICS-NEXT:    stxr w10, x9, [x8]
; OUTLINE-ATOMICS-NEXT:    cbnz w10, .LBB205_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw min i64* @var64, i64 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldsmin x{{[0-9]+}}, x{{[1-9][0-9]*}}, [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define i8 @test_atomic_load_min_i8_release(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_min_i8_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_min_i8_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var8
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var8
; OUTLINE-ATOMICS-NEXT:  .LBB206_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldxrb w10, [x9]
; OUTLINE-ATOMICS-NEXT:    sxtb w8, w10
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0, sxtb
; OUTLINE-ATOMICS-NEXT:    csel w10, w10, w0, le
; OUTLINE-ATOMICS-NEXT:    stlxrb w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB206_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw min i8* @var8, i8 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: ldsminlb w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_load_min_i16_release(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_min_i16_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_min_i16_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var16
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var16
; OUTLINE-ATOMICS-NEXT:  .LBB207_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldxrh w10, [x9]
; OUTLINE-ATOMICS-NEXT:    sxth w8, w10
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0, sxth
; OUTLINE-ATOMICS-NEXT:    csel w10, w10, w0, le
; OUTLINE-ATOMICS-NEXT:    stlxrh w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB207_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw min i16* @var16, i16 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: ldsminlh w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_load_min_i32_release(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_min_i32_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_min_i32_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var32
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var32
; OUTLINE-ATOMICS-NEXT:  .LBB208_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldxr w8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0
; OUTLINE-ATOMICS-NEXT:    csel w10, w8, w0, le
; OUTLINE-ATOMICS-NEXT:    stlxr w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB208_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw min i32* @var32, i32 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldsminl w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_load_min_i64_release(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_min_i64_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_min_i64_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var64
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var64
; OUTLINE-ATOMICS-NEXT:  .LBB209_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldxr x8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp x8, x0
; OUTLINE-ATOMICS-NEXT:    csel x10, x8, x0, le
; OUTLINE-ATOMICS-NEXT:    stlxr w11, x10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB209_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov x0, x8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw min i64* @var64, i64 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldsminl x[[OLD:[0-9]+]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define void @test_atomic_load_min_i32_noret_release(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_min_i32_noret_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_min_i32_noret_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x8, var32
; OUTLINE-ATOMICS-NEXT:    add x8, x8, :lo12:var32
; OUTLINE-ATOMICS-NEXT:  .LBB210_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldxr w9, [x8]
; OUTLINE-ATOMICS-NEXT:    cmp w9, w0
; OUTLINE-ATOMICS-NEXT:    csel w9, w9, w0, le
; OUTLINE-ATOMICS-NEXT:    stlxr w10, w9, [x8]
; OUTLINE-ATOMICS-NEXT:    cbnz w10, .LBB210_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw min i32* @var32, i32 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldsminl w{{[0-9]+}}, w{{[1-9][0-9]*}}, [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_load_min_i64_noret_release(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_min_i64_noret_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_min_i64_noret_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x8, var64
; OUTLINE-ATOMICS-NEXT:    add x8, x8, :lo12:var64
; OUTLINE-ATOMICS-NEXT:  .LBB211_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldxr x9, [x8]
; OUTLINE-ATOMICS-NEXT:    cmp x9, x0
; OUTLINE-ATOMICS-NEXT:    csel x9, x9, x0, le
; OUTLINE-ATOMICS-NEXT:    stlxr w10, x9, [x8]
; OUTLINE-ATOMICS-NEXT:    cbnz w10, .LBB211_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw min i64* @var64, i64 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldsminl x{{[0-9]+}}, x{{[1-9][0-9]*}}, [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define i8 @test_atomic_load_min_i8_seq_cst(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_min_i8_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_min_i8_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var8
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var8
; OUTLINE-ATOMICS-NEXT:  .LBB212_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxrb w10, [x9]
; OUTLINE-ATOMICS-NEXT:    sxtb w8, w10
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0, sxtb
; OUTLINE-ATOMICS-NEXT:    csel w10, w10, w0, le
; OUTLINE-ATOMICS-NEXT:    stlxrb w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB212_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw min i8* @var8, i8 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: ldsminalb w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_load_min_i16_seq_cst(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_min_i16_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_min_i16_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var16
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var16
; OUTLINE-ATOMICS-NEXT:  .LBB213_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxrh w10, [x9]
; OUTLINE-ATOMICS-NEXT:    sxth w8, w10
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0, sxth
; OUTLINE-ATOMICS-NEXT:    csel w10, w10, w0, le
; OUTLINE-ATOMICS-NEXT:    stlxrh w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB213_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw min i16* @var16, i16 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: ldsminalh w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_load_min_i32_seq_cst(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_min_i32_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_min_i32_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var32
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var32
; OUTLINE-ATOMICS-NEXT:  .LBB214_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr w8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0
; OUTLINE-ATOMICS-NEXT:    csel w10, w8, w0, le
; OUTLINE-ATOMICS-NEXT:    stlxr w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB214_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw min i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldsminal w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_load_min_i64_seq_cst(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_min_i64_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_min_i64_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var64
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var64
; OUTLINE-ATOMICS-NEXT:  .LBB215_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr x8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp x8, x0
; OUTLINE-ATOMICS-NEXT:    csel x10, x8, x0, le
; OUTLINE-ATOMICS-NEXT:    stlxr w11, x10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB215_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov x0, x8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw min i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldsminal x[[OLD:[0-9]+]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define void @test_atomic_load_min_i32_noret_seq_cst(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_min_i32_noret_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_min_i32_noret_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x8, var32
; OUTLINE-ATOMICS-NEXT:    add x8, x8, :lo12:var32
; OUTLINE-ATOMICS-NEXT:  .LBB216_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr w9, [x8]
; OUTLINE-ATOMICS-NEXT:    cmp w9, w0
; OUTLINE-ATOMICS-NEXT:    csel w9, w9, w0, le
; OUTLINE-ATOMICS-NEXT:    stlxr w10, w9, [x8]
; OUTLINE-ATOMICS-NEXT:    cbnz w10, .LBB216_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw min i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldsminal w0, w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_load_min_i64_noret_seq_cst(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_min_i64_noret_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_min_i64_noret_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x8, var64
; OUTLINE-ATOMICS-NEXT:    add x8, x8, :lo12:var64
; OUTLINE-ATOMICS-NEXT:  .LBB217_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr x9, [x8]
; OUTLINE-ATOMICS-NEXT:    cmp x9, x0
; OUTLINE-ATOMICS-NEXT:    csel x9, x9, x0, le
; OUTLINE-ATOMICS-NEXT:    stlxr w10, x9, [x8]
; OUTLINE-ATOMICS-NEXT:    cbnz w10, .LBB217_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw min i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldsminal x0, x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define i8 @test_atomic_load_or_i8_acq_rel(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_or_i8_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_or_i8_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var8
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var8
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldset1_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw or i8* @var8, i8 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: ldsetalb w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_load_or_i16_acq_rel(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_or_i16_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_or_i16_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var16
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var16
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldset2_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw or i16* @var16, i16 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: ldsetalh w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_load_or_i32_acq_rel(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_or_i32_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_or_i32_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldset4_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw or i32* @var32, i32 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldsetal w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_load_or_i64_acq_rel(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_or_i64_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_or_i64_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldset8_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw or i64* @var64, i64 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldsetal x[[OLD:[0-9]+]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define void @test_atomic_load_or_i32_noret_acq_rel(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_or_i32_noret_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_or_i32_noret_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldset4_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw or i32* @var32, i32 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldsetal w0, w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_load_or_i64_noret_acq_rel(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_or_i64_noret_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_or_i64_noret_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldset8_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw or i64* @var64, i64 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldsetal x0, x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define i8 @test_atomic_load_or_i8_acquire(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_or_i8_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_or_i8_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var8
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var8
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldset1_acq
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw or i8* @var8, i8 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: ldsetab w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_load_or_i16_acquire(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_or_i16_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_or_i16_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var16
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var16
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldset2_acq
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw or i16* @var16, i16 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: ldsetah w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_load_or_i32_acquire(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_or_i32_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_or_i32_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldset4_acq
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw or i32* @var32, i32 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldseta w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_load_or_i64_acquire(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_or_i64_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_or_i64_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldset8_acq
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw or i64* @var64, i64 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldseta x[[OLD:[0-9]+]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define void @test_atomic_load_or_i32_noret_acquire(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_or_i32_noret_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_or_i32_noret_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldset4_acq
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw or i32* @var32, i32 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldseta w0, w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_load_or_i64_noret_acquire(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_or_i64_noret_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_or_i64_noret_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldset8_acq
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw or i64* @var64, i64 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldseta x0, x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define i8 @test_atomic_load_or_i8_monotonic(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_or_i8_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_or_i8_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var8
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var8
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldset1_relax
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw or i8* @var8, i8 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: ldsetb w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_load_or_i16_monotonic(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_or_i16_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_or_i16_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var16
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var16
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldset2_relax
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw or i16* @var16, i16 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: ldseth w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_load_or_i32_monotonic(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_or_i32_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_or_i32_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldset4_relax
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw or i32* @var32, i32 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldset w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_load_or_i64_monotonic(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_or_i64_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_or_i64_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldset8_relax
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw or i64* @var64, i64 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldset x[[OLD:[0-9]+]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define void @test_atomic_load_or_i32_noret_monotonic(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_or_i32_noret_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_or_i32_noret_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldset4_relax
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw or i32* @var32, i32 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldset w{{[0-9]+}}, w{{[1-9][0-9]*}}, [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_load_or_i64_noret_monotonic(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_or_i64_noret_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_or_i64_noret_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldset8_relax
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw or i64* @var64, i64 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldset x{{[0-9]+}}, x{{[1-9][0-9]*}}, [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define i8 @test_atomic_load_or_i8_release(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_or_i8_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_or_i8_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var8
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var8
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldset1_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw or i8* @var8, i8 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: ldsetlb w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_load_or_i16_release(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_or_i16_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_or_i16_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var16
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var16
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldset2_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw or i16* @var16, i16 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: ldsetlh w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_load_or_i32_release(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_or_i32_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_or_i32_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldset4_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw or i32* @var32, i32 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldsetl w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_load_or_i64_release(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_or_i64_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_or_i64_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldset8_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw or i64* @var64, i64 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldsetl x[[OLD:[0-9]+]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define void @test_atomic_load_or_i32_noret_release(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_or_i32_noret_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_or_i32_noret_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldset4_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw or i32* @var32, i32 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldsetl w{{[0-9]+}}, w{{[1-9][0-9]*}}, [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_load_or_i64_noret_release(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_or_i64_noret_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_or_i64_noret_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldset8_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw or i64* @var64, i64 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldsetl x{{[0-9]+}}, x{{[1-9][0-9]*}}, [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define i8 @test_atomic_load_or_i8_seq_cst(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_or_i8_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_or_i8_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var8
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var8
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldset1_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw or i8* @var8, i8 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: ldsetalb w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_load_or_i16_seq_cst(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_or_i16_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_or_i16_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var16
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var16
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldset2_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw or i16* @var16, i16 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: ldsetalh w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_load_or_i32_seq_cst(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_or_i32_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_or_i32_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldset4_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw or i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldsetal w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_load_or_i64_seq_cst(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_or_i64_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_or_i64_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldset8_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw or i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldsetal x[[OLD:[0-9]+]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define void @test_atomic_load_or_i32_noret_seq_cst(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_or_i32_noret_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_or_i32_noret_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldset4_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw or i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldsetal w0, w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_load_or_i64_noret_seq_cst(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_or_i64_noret_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_or_i64_noret_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldset8_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw or i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldsetal x0, x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define i8 @test_atomic_load_sub_i8_acq_rel(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i8_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_sub_i8_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    neg w0, w0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var8
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var8
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd1_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw sub i8* @var8, i8 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: neg w[[NEG:[0-9]+]], w[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: ldaddalb w[[NEG]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

  ret i8 %old
}

define i16 @test_atomic_load_sub_i16_acq_rel(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i16_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_sub_i16_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    neg w0, w0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var16
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var16
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd2_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw sub i16* @var16, i16 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: neg w[[NEG:[0-9]+]], w[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: ldaddalh w[[NEG]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

  ret i16 %old
}

define i32 @test_atomic_load_sub_i32_acq_rel(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i32_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_sub_i32_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    neg w0, w0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd4_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw sub i32* @var32, i32 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: neg w[[NEG:[0-9]+]], w[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldaddal w[[NEG]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

  ret i32 %old
}

define i64 @test_atomic_load_sub_i64_acq_rel(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i64_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_sub_i64_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    neg x0, x0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd8_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw sub i64* @var64, i64 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: neg x[[NEG:[0-9]+]], x[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldaddal x[[NEG]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

  ret i64 %old
}

define void @test_atomic_load_sub_i32_noret_acq_rel(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i32_noret_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_sub_i32_noret_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    neg w0, w0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd4_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  atomicrmw sub i32* @var32, i32 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: neg w[[NEG:[0-9]+]], w[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldaddal w[[NEG]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

  ret void
}

define void @test_atomic_load_sub_i64_noret_acq_rel(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i64_noret_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_sub_i64_noret_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    neg x0, x0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd8_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  atomicrmw sub i64* @var64, i64 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: neg x[[NEG:[0-9]+]], x[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldaddal x[[NEG]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

  ret void
}

define i8 @test_atomic_load_sub_i8_acquire(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i8_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_sub_i8_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    neg w0, w0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var8
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var8
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd1_acq
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw sub i8* @var8, i8 %offset acquire
; CHECK-NOT: dmb
; CHECK: neg w[[NEG:[0-9]+]], w[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: ldaddab w[[NEG]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

  ret i8 %old
}

define i16 @test_atomic_load_sub_i16_acquire(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i16_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_sub_i16_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    neg w0, w0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var16
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var16
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd2_acq
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw sub i16* @var16, i16 %offset acquire
; CHECK-NOT: dmb
; CHECK: neg w[[NEG:[0-9]+]], w[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: ldaddah w[[NEG]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

  ret i16 %old
}

define i32 @test_atomic_load_sub_i32_acquire(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i32_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_sub_i32_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    neg w0, w0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd4_acq
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw sub i32* @var32, i32 %offset acquire
; CHECK-NOT: dmb
; CHECK: neg w[[NEG:[0-9]+]], w[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldadda w[[NEG]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

  ret i32 %old
}

define i64 @test_atomic_load_sub_i64_acquire(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i64_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_sub_i64_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    neg x0, x0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd8_acq
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw sub i64* @var64, i64 %offset acquire
; CHECK-NOT: dmb
; CHECK: neg x[[NEG:[0-9]+]], x[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldadda x[[NEG]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

  ret i64 %old
}

define void @test_atomic_load_sub_i32_noret_acquire(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i32_noret_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_sub_i32_noret_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    neg w0, w0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd4_acq
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  atomicrmw sub i32* @var32, i32 %offset acquire
; CHECK-NOT: dmb
; CHECK: neg w[[NEG:[0-9]+]], w[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldadda w[[NEG]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

  ret void
}

define void @test_atomic_load_sub_i64_noret_acquire(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i64_noret_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_sub_i64_noret_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    neg x0, x0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd8_acq
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  atomicrmw sub i64* @var64, i64 %offset acquire
; CHECK-NOT: dmb
; CHECK: neg x[[NEG:[0-9]+]], x[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldadda x[[NEG]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

  ret void
}

define i8 @test_atomic_load_sub_i8_monotonic(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i8_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_sub_i8_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    neg w0, w0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var8
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var8
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd1_relax
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw sub i8* @var8, i8 %offset monotonic
; CHECK-NOT: dmb
; CHECK: neg w[[NEG:[0-9]+]], w[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: ldaddb w[[NEG]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

  ret i8 %old
}

define i16 @test_atomic_load_sub_i16_monotonic(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i16_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_sub_i16_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    neg w0, w0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var16
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var16
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd2_relax
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw sub i16* @var16, i16 %offset monotonic
; CHECK-NOT: dmb
; CHECK: neg w[[NEG:[0-9]+]], w[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: ldaddh w[[NEG]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

  ret i16 %old
}

define i32 @test_atomic_load_sub_i32_monotonic(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i32_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_sub_i32_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    neg w0, w0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd4_relax
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw sub i32* @var32, i32 %offset monotonic
; CHECK-NOT: dmb
; CHECK: neg w[[NEG:[0-9]+]], w[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldadd w[[NEG]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

  ret i32 %old
}

define i64 @test_atomic_load_sub_i64_monotonic(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i64_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_sub_i64_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    neg x0, x0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd8_relax
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw sub i64* @var64, i64 %offset monotonic
; CHECK-NOT: dmb
; CHECK: neg x[[NEG:[0-9]+]], x[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldadd x[[NEG]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

  ret i64 %old
}

define void @test_atomic_load_sub_i32_noret_monotonic(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i32_noret_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_sub_i32_noret_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    neg w0, w0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd4_relax
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  atomicrmw sub i32* @var32, i32 %offset monotonic
; CHECK-NOT: dmb
; CHECK: neg w[[NEG:[0-9]+]], w[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldadd w{{[0-9]+}}, w[[NEW:[1-9][0-9]*]], [x[[ADDR]]]
; CHECK-NOT: dmb

  ret void
}

define void @test_atomic_load_sub_i64_noret_monotonic(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i64_noret_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_sub_i64_noret_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    neg x0, x0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd8_relax
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  atomicrmw sub i64* @var64, i64 %offset monotonic
; CHECK-NOT: dmb
; CHECK: neg x[[NEG:[0-9]+]], x[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldadd x{{[0-9]+}}, x[[NEW:[1-9][0-9]*]], [x[[ADDR]]]
; CHECK-NOT: dmb

  ret void
}

define i8 @test_atomic_load_sub_i8_release(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i8_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_sub_i8_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    neg w0, w0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var8
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var8
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd1_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw sub i8* @var8, i8 %offset release
; CHECK-NOT: dmb
; CHECK: neg w[[NEG:[0-9]+]], w[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: ldaddlb w[[NEG]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

  ret i8 %old
}

define i16 @test_atomic_load_sub_i16_release(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i16_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_sub_i16_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    neg w0, w0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var16
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var16
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd2_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw sub i16* @var16, i16 %offset release
; CHECK-NOT: dmb
; CHECK: neg w[[NEG:[0-9]+]], w[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: ldaddlh w[[NEG]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

  ret i16 %old
}

define i32 @test_atomic_load_sub_i32_release(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i32_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_sub_i32_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    neg w0, w0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd4_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw sub i32* @var32, i32 %offset release
; CHECK-NOT: dmb
; CHECK: neg w[[NEG:[0-9]+]], w[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldaddl w[[NEG]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

  ret i32 %old
}

define i64 @test_atomic_load_sub_i64_release(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i64_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_sub_i64_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    neg x0, x0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd8_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw sub i64* @var64, i64 %offset release
; CHECK-NOT: dmb
; CHECK: neg x[[NEG:[0-9]+]], x[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldaddl x[[NEG]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

  ret i64 %old
}

define void @test_atomic_load_sub_i32_noret_release(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i32_noret_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_sub_i32_noret_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    neg w0, w0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd4_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  atomicrmw sub i32* @var32, i32 %offset release
; CHECK-NOT: dmb
; CHECK: neg w[[NEG:[0-9]+]], w[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldaddl w{{[0-9]*}}, w[[NEW:[1-9][0-9]*]], [x[[ADDR]]]
; CHECK-NOT: dmb

  ret void
}

define void @test_atomic_load_sub_i64_noret_release(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i64_noret_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_sub_i64_noret_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    neg x0, x0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd8_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  atomicrmw sub i64* @var64, i64 %offset release
; CHECK-NOT: dmb
; CHECK: neg x[[NEG:[0-9]+]], x[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldaddl x{{[0-9]*}}, x[[NEW:[1-9][0-9]*]], [x[[ADDR]]]
; CHECK-NOT: dmb

  ret void
}

define i8 @test_atomic_load_sub_i8_seq_cst(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i8_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_sub_i8_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    neg w0, w0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var8
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var8
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd1_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw sub i8* @var8, i8 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: neg w[[NEG:[0-9]+]], w[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: ldaddalb w[[NEG]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

  ret i8 %old
}

define i16 @test_atomic_load_sub_i16_seq_cst(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i16_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_sub_i16_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    neg w0, w0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var16
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var16
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd2_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw sub i16* @var16, i16 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: neg w[[NEG:[0-9]+]], w[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: ldaddalh w[[NEG]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

  ret i16 %old
}

define i32 @test_atomic_load_sub_i32_seq_cst(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i32_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_sub_i32_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    neg w0, w0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd4_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw sub i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: neg w[[NEG:[0-9]+]], w[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldaddal w[[NEG]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

  ret i32 %old
}

define i64 @test_atomic_load_sub_i64_seq_cst(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i64_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_sub_i64_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    neg x0, x0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd8_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  %old = atomicrmw sub i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: neg x[[NEG:[0-9]+]], x[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldaddal x[[NEG]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

  ret i64 %old
}

define void @test_atomic_load_sub_i32_noret_seq_cst(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i32_noret_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_sub_i32_noret_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    neg w0, w0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd4_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  atomicrmw sub i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: neg w[[NEG:[0-9]+]], w[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldaddal w[[NEG]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

  ret void
}

define void @test_atomic_load_sub_i64_noret_seq_cst(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i64_noret_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_sub_i64_noret_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    neg x0, x0
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldadd8_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
  atomicrmw sub i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: neg x[[NEG:[0-9]+]], x[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldaddal x[[NEG]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

  ret void
}

define i8 @test_atomic_load_xchg_i8_acq_rel(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xchg_i8_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xchg_i8_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var8
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var8
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_swp1_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw xchg i8* @var8, i8 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: swpalb w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_load_xchg_i16_acq_rel(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xchg_i16_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xchg_i16_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var16
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var16
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_swp2_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw xchg i16* @var16, i16 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: swpalh w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_load_xchg_i32_acq_rel(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xchg_i32_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xchg_i32_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_swp4_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw xchg i32* @var32, i32 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: swpal w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_load_xchg_i64_acq_rel(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xchg_i64_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xchg_i64_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_swp8_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw xchg i64* @var64, i64 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: swpal x[[OLD:[0-9]+]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define void @test_atomic_load_xchg_i32_noret_acq_rel(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xchg_i32_noret_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xchg_i32_noret_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_swp4_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw xchg i32* @var32, i32 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: swpal w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret void
}

define void @test_atomic_load_xchg_i64_noret_acq_rel(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xchg_i64_noret_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xchg_i64_noret_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_swp8_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw xchg i64* @var64, i64 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: swpal x[[OLD:[0-9]+]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret void
}

define i8 @test_atomic_load_xchg_i8_acquire(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xchg_i8_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xchg_i8_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var8
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var8
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_swp1_acq
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw xchg i8* @var8, i8 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: swpab w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_load_xchg_i16_acquire(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xchg_i16_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xchg_i16_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var16
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var16
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_swp2_acq
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw xchg i16* @var16, i16 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: swpah w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_load_xchg_i32_acquire(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xchg_i32_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xchg_i32_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_swp4_acq
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw xchg i32* @var32, i32 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: swpa w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_load_xchg_i64_acquire(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xchg_i64_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xchg_i64_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_swp8_acq
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw xchg i64* @var64, i64 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: swpa x[[OLD:[0-9]+]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define void @test_atomic_load_xchg_i32_noret_acquire(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xchg_i32_noret_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xchg_i32_noret_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_swp4_acq
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw xchg i32* @var32, i32 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: swpa w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret void
}

define void @test_atomic_load_xchg_i64_noret_acquire(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xchg_i64_noret_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xchg_i64_noret_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_swp8_acq
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw xchg i64* @var64, i64 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: swpa x[[OLD:[0-9]+]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret void
}

define i8 @test_atomic_load_xchg_i8_monotonic(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xchg_i8_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xchg_i8_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var8
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var8
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_swp1_relax
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw xchg i8* @var8, i8 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: swpb w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_load_xchg_i16_monotonic(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xchg_i16_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xchg_i16_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var16
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var16
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_swp2_relax
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw xchg i16* @var16, i16 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: swph w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_load_xchg_i32_monotonic(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xchg_i32_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xchg_i32_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_swp4_relax
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw xchg i32* @var32, i32 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: swp w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_load_xchg_i64_monotonic(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xchg_i64_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xchg_i64_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_swp8_relax
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw xchg i64* @var64, i64 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: swp x[[OLD:[0-9]+]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define void @test_atomic_load_xchg_i32_noret_monotonic(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xchg_i32_noret_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xchg_i32_noret_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_swp4_relax
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw xchg i32* @var32, i32 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: swp w[[OLD:[0-9]+]], w[[NEW:[0-9,a-z]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret void
}

define void @test_atomic_load_xchg_i64_noret_monotonic(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xchg_i64_noret_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xchg_i64_noret_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_swp8_relax
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw xchg i64* @var64, i64 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: swp x[[OLD:[0-9]+]], x[[NEW:[0-9,a-z]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret void
}

define i8 @test_atomic_load_xchg_i8_release(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xchg_i8_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xchg_i8_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var8
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var8
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_swp1_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw xchg i8* @var8, i8 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: swplb w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_load_xchg_i16_release(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xchg_i16_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xchg_i16_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var16
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var16
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_swp2_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw xchg i16* @var16, i16 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: swplh w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_load_xchg_i32_release(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xchg_i32_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xchg_i32_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_swp4_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw xchg i32* @var32, i32 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: swpl w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_load_xchg_i64_release(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xchg_i64_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xchg_i64_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_swp8_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw xchg i64* @var64, i64 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: swpl x[[OLD:[0-9]+]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define void @test_atomic_load_xchg_i32_noret_release(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xchg_i32_noret_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xchg_i32_noret_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_swp4_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw xchg i32* @var32, i32 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: swpl w[[OLD:[0-9]+]], w[[NEW:[0-9,a-z]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret void
}

define void @test_atomic_load_xchg_i64_noret_release(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xchg_i64_noret_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xchg_i64_noret_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_swp8_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw xchg i64* @var64, i64 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: swpl x[[OLD:[0-9]+]], x[[NEW:[0-9,a-z]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret void
}

define i8 @test_atomic_load_xchg_i8_seq_cst(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xchg_i8_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xchg_i8_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var8
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var8
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_swp1_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw xchg i8* @var8, i8 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: swpalb w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_load_xchg_i16_seq_cst(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xchg_i16_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xchg_i16_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var16
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var16
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_swp2_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw xchg i16* @var16, i16 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: swpalh w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_load_xchg_i32_seq_cst(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xchg_i32_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xchg_i32_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_swp4_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw xchg i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: swpal w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_load_xchg_i64_seq_cst(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xchg_i64_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xchg_i64_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_swp8_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw xchg i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: swpal x[[OLD:[0-9]+]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define void @test_atomic_load_xchg_i32_noret_seq_cst(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xchg_i32_noret_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xchg_i32_noret_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_swp4_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw xchg i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: swpal w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret void
}

define void @test_atomic_load_xchg_i64_noret_seq_cst(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xchg_i64_noret_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xchg_i64_noret_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_swp8_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw xchg i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: swpal x[[OLD:[0-9]+]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret void
}

define i8 @test_atomic_load_umax_i8_acq_rel(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umax_i8_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umax_i8_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var8
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var8
; OUTLINE-ATOMICS-NEXT:  .LBB308_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxrb w8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0, uxtb
; OUTLINE-ATOMICS-NEXT:    csel w10, w8, w0, hi
; OUTLINE-ATOMICS-NEXT:    stlxrb w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB308_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw umax i8* @var8, i8 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: ldumaxalb w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_load_umax_i16_acq_rel(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umax_i16_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umax_i16_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var16
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var16
; OUTLINE-ATOMICS-NEXT:  .LBB309_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxrh w8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0, uxth
; OUTLINE-ATOMICS-NEXT:    csel w10, w8, w0, hi
; OUTLINE-ATOMICS-NEXT:    stlxrh w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB309_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw umax i16* @var16, i16 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: ldumaxalh w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_load_umax_i32_acq_rel(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umax_i32_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umax_i32_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var32
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var32
; OUTLINE-ATOMICS-NEXT:  .LBB310_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr w8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0
; OUTLINE-ATOMICS-NEXT:    csel w10, w8, w0, hi
; OUTLINE-ATOMICS-NEXT:    stlxr w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB310_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw umax i32* @var32, i32 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldumaxal w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_load_umax_i64_acq_rel(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umax_i64_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umax_i64_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var64
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var64
; OUTLINE-ATOMICS-NEXT:  .LBB311_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr x8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp x8, x0
; OUTLINE-ATOMICS-NEXT:    csel x10, x8, x0, hi
; OUTLINE-ATOMICS-NEXT:    stlxr w11, x10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB311_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov x0, x8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw umax i64* @var64, i64 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldumaxal x[[OLD:[0-9]+]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define void @test_atomic_load_umax_i32_noret_acq_rel(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umax_i32_noret_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umax_i32_noret_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x8, var32
; OUTLINE-ATOMICS-NEXT:    add x8, x8, :lo12:var32
; OUTLINE-ATOMICS-NEXT:  .LBB312_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr w9, [x8]
; OUTLINE-ATOMICS-NEXT:    cmp w9, w0
; OUTLINE-ATOMICS-NEXT:    csel w9, w9, w0, hi
; OUTLINE-ATOMICS-NEXT:    stlxr w10, w9, [x8]
; OUTLINE-ATOMICS-NEXT:    cbnz w10, .LBB312_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw umax i32* @var32, i32 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldumaxal w0, w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_load_umax_i64_noret_acq_rel(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umax_i64_noret_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umax_i64_noret_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x8, var64
; OUTLINE-ATOMICS-NEXT:    add x8, x8, :lo12:var64
; OUTLINE-ATOMICS-NEXT:  .LBB313_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr x9, [x8]
; OUTLINE-ATOMICS-NEXT:    cmp x9, x0
; OUTLINE-ATOMICS-NEXT:    csel x9, x9, x0, hi
; OUTLINE-ATOMICS-NEXT:    stlxr w10, x9, [x8]
; OUTLINE-ATOMICS-NEXT:    cbnz w10, .LBB313_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw umax i64* @var64, i64 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldumaxal x0, x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define i8 @test_atomic_load_umax_i8_acquire(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umax_i8_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umax_i8_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var8
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var8
; OUTLINE-ATOMICS-NEXT:  .LBB314_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxrb w8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0, uxtb
; OUTLINE-ATOMICS-NEXT:    csel w10, w8, w0, hi
; OUTLINE-ATOMICS-NEXT:    stxrb w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB314_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw umax i8* @var8, i8 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: ldumaxab w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_load_umax_i16_acquire(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umax_i16_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umax_i16_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var16
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var16
; OUTLINE-ATOMICS-NEXT:  .LBB315_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxrh w8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0, uxth
; OUTLINE-ATOMICS-NEXT:    csel w10, w8, w0, hi
; OUTLINE-ATOMICS-NEXT:    stxrh w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB315_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw umax i16* @var16, i16 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: ldumaxah w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_load_umax_i32_acquire(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umax_i32_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umax_i32_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var32
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var32
; OUTLINE-ATOMICS-NEXT:  .LBB316_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr w8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0
; OUTLINE-ATOMICS-NEXT:    csel w10, w8, w0, hi
; OUTLINE-ATOMICS-NEXT:    stxr w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB316_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw umax i32* @var32, i32 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldumaxa w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_load_umax_i64_acquire(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umax_i64_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umax_i64_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var64
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var64
; OUTLINE-ATOMICS-NEXT:  .LBB317_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr x8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp x8, x0
; OUTLINE-ATOMICS-NEXT:    csel x10, x8, x0, hi
; OUTLINE-ATOMICS-NEXT:    stxr w11, x10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB317_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov x0, x8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw umax i64* @var64, i64 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldumaxa x[[OLD:[0-9]+]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define void @test_atomic_load_umax_i32_noret_acquire(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umax_i32_noret_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umax_i32_noret_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x8, var32
; OUTLINE-ATOMICS-NEXT:    add x8, x8, :lo12:var32
; OUTLINE-ATOMICS-NEXT:  .LBB318_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr w9, [x8]
; OUTLINE-ATOMICS-NEXT:    cmp w9, w0
; OUTLINE-ATOMICS-NEXT:    csel w9, w9, w0, hi
; OUTLINE-ATOMICS-NEXT:    stxr w10, w9, [x8]
; OUTLINE-ATOMICS-NEXT:    cbnz w10, .LBB318_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw umax i32* @var32, i32 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldumaxa w0, w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_load_umax_i64_noret_acquire(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umax_i64_noret_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umax_i64_noret_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x8, var64
; OUTLINE-ATOMICS-NEXT:    add x8, x8, :lo12:var64
; OUTLINE-ATOMICS-NEXT:  .LBB319_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr x9, [x8]
; OUTLINE-ATOMICS-NEXT:    cmp x9, x0
; OUTLINE-ATOMICS-NEXT:    csel x9, x9, x0, hi
; OUTLINE-ATOMICS-NEXT:    stxr w10, x9, [x8]
; OUTLINE-ATOMICS-NEXT:    cbnz w10, .LBB319_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw umax i64* @var64, i64 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldumaxa x0, x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define i8 @test_atomic_load_umax_i8_monotonic(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umax_i8_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umax_i8_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var8
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var8
; OUTLINE-ATOMICS-NEXT:  .LBB320_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldxrb w8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0, uxtb
; OUTLINE-ATOMICS-NEXT:    csel w10, w8, w0, hi
; OUTLINE-ATOMICS-NEXT:    stxrb w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB320_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw umax i8* @var8, i8 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: ldumaxb w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_load_umax_i16_monotonic(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umax_i16_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umax_i16_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var16
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var16
; OUTLINE-ATOMICS-NEXT:  .LBB321_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldxrh w8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0, uxth
; OUTLINE-ATOMICS-NEXT:    csel w10, w8, w0, hi
; OUTLINE-ATOMICS-NEXT:    stxrh w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB321_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw umax i16* @var16, i16 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: ldumaxh w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_load_umax_i32_monotonic(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umax_i32_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umax_i32_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var32
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var32
; OUTLINE-ATOMICS-NEXT:  .LBB322_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldxr w8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0
; OUTLINE-ATOMICS-NEXT:    csel w10, w8, w0, hi
; OUTLINE-ATOMICS-NEXT:    stxr w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB322_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw umax i32* @var32, i32 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldumax w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_load_umax_i64_monotonic(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umax_i64_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umax_i64_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var64
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var64
; OUTLINE-ATOMICS-NEXT:  .LBB323_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldxr x8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp x8, x0
; OUTLINE-ATOMICS-NEXT:    csel x10, x8, x0, hi
; OUTLINE-ATOMICS-NEXT:    stxr w11, x10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB323_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov x0, x8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw umax i64* @var64, i64 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldumax x[[OLD:[0-9]+]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define void @test_atomic_load_umax_i32_noret_monotonic(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umax_i32_noret_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umax_i32_noret_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x8, var32
; OUTLINE-ATOMICS-NEXT:    add x8, x8, :lo12:var32
; OUTLINE-ATOMICS-NEXT:  .LBB324_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldxr w9, [x8]
; OUTLINE-ATOMICS-NEXT:    cmp w9, w0
; OUTLINE-ATOMICS-NEXT:    csel w9, w9, w0, hi
; OUTLINE-ATOMICS-NEXT:    stxr w10, w9, [x8]
; OUTLINE-ATOMICS-NEXT:    cbnz w10, .LBB324_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw umax i32* @var32, i32 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldumax w{{[0-9]+}}, w{{[1-9][0-9]*}}, [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_load_umax_i64_noret_monotonic(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umax_i64_noret_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umax_i64_noret_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x8, var64
; OUTLINE-ATOMICS-NEXT:    add x8, x8, :lo12:var64
; OUTLINE-ATOMICS-NEXT:  .LBB325_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldxr x9, [x8]
; OUTLINE-ATOMICS-NEXT:    cmp x9, x0
; OUTLINE-ATOMICS-NEXT:    csel x9, x9, x0, hi
; OUTLINE-ATOMICS-NEXT:    stxr w10, x9, [x8]
; OUTLINE-ATOMICS-NEXT:    cbnz w10, .LBB325_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw umax i64* @var64, i64 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldumax x{{[0-9]+}}, x{{[1-9][0-9]*}}, [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define i8 @test_atomic_load_umax_i8_release(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umax_i8_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umax_i8_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var8
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var8
; OUTLINE-ATOMICS-NEXT:  .LBB326_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldxrb w8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0, uxtb
; OUTLINE-ATOMICS-NEXT:    csel w10, w8, w0, hi
; OUTLINE-ATOMICS-NEXT:    stlxrb w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB326_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw umax i8* @var8, i8 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: ldumaxlb w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_load_umax_i16_release(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umax_i16_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umax_i16_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var16
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var16
; OUTLINE-ATOMICS-NEXT:  .LBB327_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldxrh w8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0, uxth
; OUTLINE-ATOMICS-NEXT:    csel w10, w8, w0, hi
; OUTLINE-ATOMICS-NEXT:    stlxrh w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB327_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw umax i16* @var16, i16 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: ldumaxlh w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_load_umax_i32_release(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umax_i32_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umax_i32_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var32
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var32
; OUTLINE-ATOMICS-NEXT:  .LBB328_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldxr w8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0
; OUTLINE-ATOMICS-NEXT:    csel w10, w8, w0, hi
; OUTLINE-ATOMICS-NEXT:    stlxr w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB328_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw umax i32* @var32, i32 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldumaxl w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_load_umax_i64_release(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umax_i64_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umax_i64_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var64
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var64
; OUTLINE-ATOMICS-NEXT:  .LBB329_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldxr x8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp x8, x0
; OUTLINE-ATOMICS-NEXT:    csel x10, x8, x0, hi
; OUTLINE-ATOMICS-NEXT:    stlxr w11, x10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB329_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov x0, x8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw umax i64* @var64, i64 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldumaxl x[[OLD:[0-9]+]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define void @test_atomic_load_umax_i32_noret_release(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umax_i32_noret_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umax_i32_noret_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x8, var32
; OUTLINE-ATOMICS-NEXT:    add x8, x8, :lo12:var32
; OUTLINE-ATOMICS-NEXT:  .LBB330_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldxr w9, [x8]
; OUTLINE-ATOMICS-NEXT:    cmp w9, w0
; OUTLINE-ATOMICS-NEXT:    csel w9, w9, w0, hi
; OUTLINE-ATOMICS-NEXT:    stlxr w10, w9, [x8]
; OUTLINE-ATOMICS-NEXT:    cbnz w10, .LBB330_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw umax i32* @var32, i32 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldumaxl w{{[0-9]+}}, w{{[1-9][0-9]*}}, [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_load_umax_i64_noret_release(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umax_i64_noret_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umax_i64_noret_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x8, var64
; OUTLINE-ATOMICS-NEXT:    add x8, x8, :lo12:var64
; OUTLINE-ATOMICS-NEXT:  .LBB331_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldxr x9, [x8]
; OUTLINE-ATOMICS-NEXT:    cmp x9, x0
; OUTLINE-ATOMICS-NEXT:    csel x9, x9, x0, hi
; OUTLINE-ATOMICS-NEXT:    stlxr w10, x9, [x8]
; OUTLINE-ATOMICS-NEXT:    cbnz w10, .LBB331_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw umax i64* @var64, i64 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldumaxl x{{[0-9]+}}, x{{[1-9][0-9]*}}, [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define i8 @test_atomic_load_umax_i8_seq_cst(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umax_i8_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umax_i8_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var8
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var8
; OUTLINE-ATOMICS-NEXT:  .LBB332_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxrb w8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0, uxtb
; OUTLINE-ATOMICS-NEXT:    csel w10, w8, w0, hi
; OUTLINE-ATOMICS-NEXT:    stlxrb w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB332_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw umax i8* @var8, i8 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: ldumaxalb w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_load_umax_i16_seq_cst(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umax_i16_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umax_i16_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var16
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var16
; OUTLINE-ATOMICS-NEXT:  .LBB333_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxrh w8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0, uxth
; OUTLINE-ATOMICS-NEXT:    csel w10, w8, w0, hi
; OUTLINE-ATOMICS-NEXT:    stlxrh w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB333_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw umax i16* @var16, i16 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: ldumaxalh w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_load_umax_i32_seq_cst(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umax_i32_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umax_i32_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var32
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var32
; OUTLINE-ATOMICS-NEXT:  .LBB334_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr w8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0
; OUTLINE-ATOMICS-NEXT:    csel w10, w8, w0, hi
; OUTLINE-ATOMICS-NEXT:    stlxr w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB334_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw umax i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldumaxal w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_load_umax_i64_seq_cst(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umax_i64_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umax_i64_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var64
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var64
; OUTLINE-ATOMICS-NEXT:  .LBB335_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr x8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp x8, x0
; OUTLINE-ATOMICS-NEXT:    csel x10, x8, x0, hi
; OUTLINE-ATOMICS-NEXT:    stlxr w11, x10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB335_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov x0, x8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw umax i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldumaxal x[[OLD:[0-9]+]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define void @test_atomic_load_umax_i32_noret_seq_cst(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umax_i32_noret_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umax_i32_noret_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x8, var32
; OUTLINE-ATOMICS-NEXT:    add x8, x8, :lo12:var32
; OUTLINE-ATOMICS-NEXT:  .LBB336_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr w9, [x8]
; OUTLINE-ATOMICS-NEXT:    cmp w9, w0
; OUTLINE-ATOMICS-NEXT:    csel w9, w9, w0, hi
; OUTLINE-ATOMICS-NEXT:    stlxr w10, w9, [x8]
; OUTLINE-ATOMICS-NEXT:    cbnz w10, .LBB336_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw umax i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldumaxal w0, w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_load_umax_i64_noret_seq_cst(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umax_i64_noret_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umax_i64_noret_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x8, var64
; OUTLINE-ATOMICS-NEXT:    add x8, x8, :lo12:var64
; OUTLINE-ATOMICS-NEXT:  .LBB337_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr x9, [x8]
; OUTLINE-ATOMICS-NEXT:    cmp x9, x0
; OUTLINE-ATOMICS-NEXT:    csel x9, x9, x0, hi
; OUTLINE-ATOMICS-NEXT:    stlxr w10, x9, [x8]
; OUTLINE-ATOMICS-NEXT:    cbnz w10, .LBB337_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw umax i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldumaxal x0, x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define i8 @test_atomic_load_umin_i8_acq_rel(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umin_i8_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umin_i8_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var8
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var8
; OUTLINE-ATOMICS-NEXT:  .LBB338_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxrb w8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0, uxtb
; OUTLINE-ATOMICS-NEXT:    csel w10, w8, w0, ls
; OUTLINE-ATOMICS-NEXT:    stlxrb w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB338_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw umin i8* @var8, i8 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: lduminalb w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_load_umin_i16_acq_rel(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umin_i16_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umin_i16_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var16
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var16
; OUTLINE-ATOMICS-NEXT:  .LBB339_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxrh w8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0, uxth
; OUTLINE-ATOMICS-NEXT:    csel w10, w8, w0, ls
; OUTLINE-ATOMICS-NEXT:    stlxrh w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB339_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw umin i16* @var16, i16 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: lduminalh w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_load_umin_i32_acq_rel(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umin_i32_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umin_i32_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var32
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var32
; OUTLINE-ATOMICS-NEXT:  .LBB340_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr w8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0
; OUTLINE-ATOMICS-NEXT:    csel w10, w8, w0, ls
; OUTLINE-ATOMICS-NEXT:    stlxr w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB340_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw umin i32* @var32, i32 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: lduminal w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_load_umin_i64_acq_rel(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umin_i64_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umin_i64_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var64
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var64
; OUTLINE-ATOMICS-NEXT:  .LBB341_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr x8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp x8, x0
; OUTLINE-ATOMICS-NEXT:    csel x10, x8, x0, ls
; OUTLINE-ATOMICS-NEXT:    stlxr w11, x10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB341_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov x0, x8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw umin i64* @var64, i64 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: lduminal x[[OLD:[0-9]+]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define void @test_atomic_load_umin_i32_noret_acq_rel(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umin_i32_noret_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umin_i32_noret_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x8, var32
; OUTLINE-ATOMICS-NEXT:    add x8, x8, :lo12:var32
; OUTLINE-ATOMICS-NEXT:  .LBB342_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr w9, [x8]
; OUTLINE-ATOMICS-NEXT:    cmp w9, w0
; OUTLINE-ATOMICS-NEXT:    csel w9, w9, w0, ls
; OUTLINE-ATOMICS-NEXT:    stlxr w10, w9, [x8]
; OUTLINE-ATOMICS-NEXT:    cbnz w10, .LBB342_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw umin i32* @var32, i32 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: lduminal w0, w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_load_umin_i64_noret_acq_rel(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umin_i64_noret_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umin_i64_noret_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x8, var64
; OUTLINE-ATOMICS-NEXT:    add x8, x8, :lo12:var64
; OUTLINE-ATOMICS-NEXT:  .LBB343_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr x9, [x8]
; OUTLINE-ATOMICS-NEXT:    cmp x9, x0
; OUTLINE-ATOMICS-NEXT:    csel x9, x9, x0, ls
; OUTLINE-ATOMICS-NEXT:    stlxr w10, x9, [x8]
; OUTLINE-ATOMICS-NEXT:    cbnz w10, .LBB343_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw umin i64* @var64, i64 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: lduminal x0, x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define i8 @test_atomic_load_umin_i8_acquire(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umin_i8_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umin_i8_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var8
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var8
; OUTLINE-ATOMICS-NEXT:  .LBB344_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxrb w8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0, uxtb
; OUTLINE-ATOMICS-NEXT:    csel w10, w8, w0, ls
; OUTLINE-ATOMICS-NEXT:    stxrb w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB344_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw umin i8* @var8, i8 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: lduminab w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_load_umin_i16_acquire(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umin_i16_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umin_i16_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var16
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var16
; OUTLINE-ATOMICS-NEXT:  .LBB345_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxrh w8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0, uxth
; OUTLINE-ATOMICS-NEXT:    csel w10, w8, w0, ls
; OUTLINE-ATOMICS-NEXT:    stxrh w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB345_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw umin i16* @var16, i16 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: lduminah w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_load_umin_i32_acquire(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umin_i32_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umin_i32_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var32
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var32
; OUTLINE-ATOMICS-NEXT:  .LBB346_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr w8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0
; OUTLINE-ATOMICS-NEXT:    csel w10, w8, w0, ls
; OUTLINE-ATOMICS-NEXT:    stxr w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB346_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw umin i32* @var32, i32 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldumina w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_load_umin_i64_acquire(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umin_i64_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umin_i64_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var64
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var64
; OUTLINE-ATOMICS-NEXT:  .LBB347_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr x8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp x8, x0
; OUTLINE-ATOMICS-NEXT:    csel x10, x8, x0, ls
; OUTLINE-ATOMICS-NEXT:    stxr w11, x10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB347_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov x0, x8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw umin i64* @var64, i64 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldumina x[[OLD:[0-9]+]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define void @test_atomic_load_umin_i32_noret_acquire(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umin_i32_noret_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umin_i32_noret_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x8, var32
; OUTLINE-ATOMICS-NEXT:    add x8, x8, :lo12:var32
; OUTLINE-ATOMICS-NEXT:  .LBB348_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr w9, [x8]
; OUTLINE-ATOMICS-NEXT:    cmp w9, w0
; OUTLINE-ATOMICS-NEXT:    csel w9, w9, w0, ls
; OUTLINE-ATOMICS-NEXT:    stxr w10, w9, [x8]
; OUTLINE-ATOMICS-NEXT:    cbnz w10, .LBB348_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw umin i32* @var32, i32 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldumina w0, w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_load_umin_i64_noret_acquire(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umin_i64_noret_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umin_i64_noret_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x8, var64
; OUTLINE-ATOMICS-NEXT:    add x8, x8, :lo12:var64
; OUTLINE-ATOMICS-NEXT:  .LBB349_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr x9, [x8]
; OUTLINE-ATOMICS-NEXT:    cmp x9, x0
; OUTLINE-ATOMICS-NEXT:    csel x9, x9, x0, ls
; OUTLINE-ATOMICS-NEXT:    stxr w10, x9, [x8]
; OUTLINE-ATOMICS-NEXT:    cbnz w10, .LBB349_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw umin i64* @var64, i64 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldumina x0, x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define i8 @test_atomic_load_umin_i8_monotonic(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umin_i8_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umin_i8_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var8
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var8
; OUTLINE-ATOMICS-NEXT:  .LBB350_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldxrb w8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0, uxtb
; OUTLINE-ATOMICS-NEXT:    csel w10, w8, w0, ls
; OUTLINE-ATOMICS-NEXT:    stxrb w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB350_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw umin i8* @var8, i8 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: lduminb w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_load_umin_i16_monotonic(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umin_i16_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umin_i16_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var16
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var16
; OUTLINE-ATOMICS-NEXT:  .LBB351_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldxrh w8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0, uxth
; OUTLINE-ATOMICS-NEXT:    csel w10, w8, w0, ls
; OUTLINE-ATOMICS-NEXT:    stxrh w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB351_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw umin i16* @var16, i16 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: lduminh w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_load_umin_i32_monotonic(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umin_i32_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umin_i32_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var32
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var32
; OUTLINE-ATOMICS-NEXT:  .LBB352_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldxr w8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0
; OUTLINE-ATOMICS-NEXT:    csel w10, w8, w0, ls
; OUTLINE-ATOMICS-NEXT:    stxr w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB352_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw umin i32* @var32, i32 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldumin w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_load_umin_i64_monotonic(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umin_i64_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umin_i64_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var64
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var64
; OUTLINE-ATOMICS-NEXT:  .LBB353_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldxr x8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp x8, x0
; OUTLINE-ATOMICS-NEXT:    csel x10, x8, x0, ls
; OUTLINE-ATOMICS-NEXT:    stxr w11, x10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB353_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov x0, x8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw umin i64* @var64, i64 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldumin x[[OLD:[0-9]+]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define void @test_atomic_load_umin_i32_noret_monotonic(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umin_i32_noret_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umin_i32_noret_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x8, var32
; OUTLINE-ATOMICS-NEXT:    add x8, x8, :lo12:var32
; OUTLINE-ATOMICS-NEXT:  .LBB354_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldxr w9, [x8]
; OUTLINE-ATOMICS-NEXT:    cmp w9, w0
; OUTLINE-ATOMICS-NEXT:    csel w9, w9, w0, ls
; OUTLINE-ATOMICS-NEXT:    stxr w10, w9, [x8]
; OUTLINE-ATOMICS-NEXT:    cbnz w10, .LBB354_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw umin i32* @var32, i32 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldumin w{{[0-9]+}}, w{{[1-9][0-9]*}}, [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_load_umin_i64_noret_monotonic(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umin_i64_noret_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umin_i64_noret_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x8, var64
; OUTLINE-ATOMICS-NEXT:    add x8, x8, :lo12:var64
; OUTLINE-ATOMICS-NEXT:  .LBB355_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldxr x9, [x8]
; OUTLINE-ATOMICS-NEXT:    cmp x9, x0
; OUTLINE-ATOMICS-NEXT:    csel x9, x9, x0, ls
; OUTLINE-ATOMICS-NEXT:    stxr w10, x9, [x8]
; OUTLINE-ATOMICS-NEXT:    cbnz w10, .LBB355_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw umin i64* @var64, i64 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldumin x{{[0-9]+}}, x{{[1-9][0-9]*}}, [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define i8 @test_atomic_load_umin_i8_release(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umin_i8_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umin_i8_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var8
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var8
; OUTLINE-ATOMICS-NEXT:  .LBB356_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldxrb w8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0, uxtb
; OUTLINE-ATOMICS-NEXT:    csel w10, w8, w0, ls
; OUTLINE-ATOMICS-NEXT:    stlxrb w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB356_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw umin i8* @var8, i8 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: lduminlb w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_load_umin_i16_release(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umin_i16_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umin_i16_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var16
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var16
; OUTLINE-ATOMICS-NEXT:  .LBB357_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldxrh w8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0, uxth
; OUTLINE-ATOMICS-NEXT:    csel w10, w8, w0, ls
; OUTLINE-ATOMICS-NEXT:    stlxrh w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB357_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw umin i16* @var16, i16 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: lduminlh w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_load_umin_i32_release(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umin_i32_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umin_i32_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var32
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var32
; OUTLINE-ATOMICS-NEXT:  .LBB358_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldxr w8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0
; OUTLINE-ATOMICS-NEXT:    csel w10, w8, w0, ls
; OUTLINE-ATOMICS-NEXT:    stlxr w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB358_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw umin i32* @var32, i32 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: lduminl w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_load_umin_i64_release(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umin_i64_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umin_i64_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var64
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var64
; OUTLINE-ATOMICS-NEXT:  .LBB359_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldxr x8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp x8, x0
; OUTLINE-ATOMICS-NEXT:    csel x10, x8, x0, ls
; OUTLINE-ATOMICS-NEXT:    stlxr w11, x10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB359_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov x0, x8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw umin i64* @var64, i64 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: lduminl x[[OLD:[0-9]+]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define void @test_atomic_load_umin_i32_noret_release(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umin_i32_noret_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umin_i32_noret_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x8, var32
; OUTLINE-ATOMICS-NEXT:    add x8, x8, :lo12:var32
; OUTLINE-ATOMICS-NEXT:  .LBB360_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldxr w9, [x8]
; OUTLINE-ATOMICS-NEXT:    cmp w9, w0
; OUTLINE-ATOMICS-NEXT:    csel w9, w9, w0, ls
; OUTLINE-ATOMICS-NEXT:    stlxr w10, w9, [x8]
; OUTLINE-ATOMICS-NEXT:    cbnz w10, .LBB360_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw umin i32* @var32, i32 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: lduminl w{{[0-9]+}}, w{{[1-9][0-9]*}}, [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_load_umin_i64_noret_release(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umin_i64_noret_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umin_i64_noret_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x8, var64
; OUTLINE-ATOMICS-NEXT:    add x8, x8, :lo12:var64
; OUTLINE-ATOMICS-NEXT:  .LBB361_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldxr x9, [x8]
; OUTLINE-ATOMICS-NEXT:    cmp x9, x0
; OUTLINE-ATOMICS-NEXT:    csel x9, x9, x0, ls
; OUTLINE-ATOMICS-NEXT:    stlxr w10, x9, [x8]
; OUTLINE-ATOMICS-NEXT:    cbnz w10, .LBB361_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw umin i64* @var64, i64 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: lduminl x{{[0-9]+}}, x{{[1-9][0-9]*}}, [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define i8 @test_atomic_load_umin_i8_seq_cst(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umin_i8_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umin_i8_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var8
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var8
; OUTLINE-ATOMICS-NEXT:  .LBB362_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxrb w8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0, uxtb
; OUTLINE-ATOMICS-NEXT:    csel w10, w8, w0, ls
; OUTLINE-ATOMICS-NEXT:    stlxrb w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB362_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw umin i8* @var8, i8 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: lduminalb w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_load_umin_i16_seq_cst(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umin_i16_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umin_i16_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var16
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var16
; OUTLINE-ATOMICS-NEXT:  .LBB363_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxrh w8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0, uxth
; OUTLINE-ATOMICS-NEXT:    csel w10, w8, w0, ls
; OUTLINE-ATOMICS-NEXT:    stlxrh w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB363_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw umin i16* @var16, i16 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: lduminalh w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_load_umin_i32_seq_cst(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umin_i32_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umin_i32_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var32
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var32
; OUTLINE-ATOMICS-NEXT:  .LBB364_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr w8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp w8, w0
; OUTLINE-ATOMICS-NEXT:    csel w10, w8, w0, ls
; OUTLINE-ATOMICS-NEXT:    stlxr w11, w10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB364_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov w0, w8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw umin i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: lduminal w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_load_umin_i64_seq_cst(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umin_i64_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umin_i64_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x9, var64
; OUTLINE-ATOMICS-NEXT:    add x9, x9, :lo12:var64
; OUTLINE-ATOMICS-NEXT:  .LBB365_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr x8, [x9]
; OUTLINE-ATOMICS-NEXT:    cmp x8, x0
; OUTLINE-ATOMICS-NEXT:    csel x10, x8, x0, ls
; OUTLINE-ATOMICS-NEXT:    stlxr w11, x10, [x9]
; OUTLINE-ATOMICS-NEXT:    cbnz w11, .LBB365_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    mov x0, x8
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw umin i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: lduminal x[[OLD:[0-9]+]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define void @test_atomic_load_umin_i32_noret_seq_cst(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umin_i32_noret_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umin_i32_noret_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x8, var32
; OUTLINE-ATOMICS-NEXT:    add x8, x8, :lo12:var32
; OUTLINE-ATOMICS-NEXT:  .LBB366_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr w9, [x8]
; OUTLINE-ATOMICS-NEXT:    cmp w9, w0
; OUTLINE-ATOMICS-NEXT:    csel w9, w9, w0, ls
; OUTLINE-ATOMICS-NEXT:    stlxr w10, w9, [x8]
; OUTLINE-ATOMICS-NEXT:    cbnz w10, .LBB366_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw umin i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: lduminal w0, w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_load_umin_i64_noret_seq_cst(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umin_i64_noret_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_umin_i64_noret_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    adrp x8, var64
; OUTLINE-ATOMICS-NEXT:    add x8, x8, :lo12:var64
; OUTLINE-ATOMICS-NEXT:  .LBB367_1: // %atomicrmw.start
; OUTLINE-ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE-ATOMICS-NEXT:    ldaxr x9, [x8]
; OUTLINE-ATOMICS-NEXT:    cmp x9, x0
; OUTLINE-ATOMICS-NEXT:    csel x9, x9, x0, ls
; OUTLINE-ATOMICS-NEXT:    stlxr w10, x9, [x8]
; OUTLINE-ATOMICS-NEXT:    cbnz w10, .LBB367_1
; OUTLINE-ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw umin i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: lduminal x0, x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define i8 @test_atomic_load_xor_i8_acq_rel(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xor_i8_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xor_i8_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var8
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var8
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldeor1_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw xor i8* @var8, i8 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: ldeoralb w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_load_xor_i16_acq_rel(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xor_i16_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xor_i16_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var16
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var16
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldeor2_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw xor i16* @var16, i16 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: ldeoralh w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_load_xor_i32_acq_rel(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xor_i32_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xor_i32_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldeor4_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw xor i32* @var32, i32 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldeoral w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_load_xor_i64_acq_rel(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xor_i64_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xor_i64_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldeor8_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw xor i64* @var64, i64 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldeoral x[[OLD:[0-9]+]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define void @test_atomic_load_xor_i32_noret_acq_rel(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xor_i32_noret_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xor_i32_noret_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldeor4_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw xor i32* @var32, i32 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldeoral w0, w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_load_xor_i64_noret_acq_rel(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xor_i64_noret_acq_rel:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xor_i64_noret_acq_rel:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldeor8_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw xor i64* @var64, i64 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldeoral x0, x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define i8 @test_atomic_load_xor_i8_acquire(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xor_i8_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xor_i8_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var8
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var8
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldeor1_acq
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw xor i8* @var8, i8 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: ldeorab w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_load_xor_i16_acquire(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xor_i16_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xor_i16_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var16
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var16
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldeor2_acq
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw xor i16* @var16, i16 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: ldeorah w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_load_xor_i32_acquire(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xor_i32_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xor_i32_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldeor4_acq
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw xor i32* @var32, i32 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldeora w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_load_xor_i64_acquire(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xor_i64_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xor_i64_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldeor8_acq
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw xor i64* @var64, i64 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldeora x[[OLD:[0-9]+]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define void @test_atomic_load_xor_i32_noret_acquire(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xor_i32_noret_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xor_i32_noret_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldeor4_acq
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw xor i32* @var32, i32 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldeora w0, w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_load_xor_i64_noret_acquire(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xor_i64_noret_acquire:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xor_i64_noret_acquire:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldeor8_acq
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw xor i64* @var64, i64 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldeora x0, x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define i8 @test_atomic_load_xor_i8_monotonic(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xor_i8_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xor_i8_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var8
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var8
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldeor1_relax
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw xor i8* @var8, i8 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: ldeorb w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_load_xor_i16_monotonic(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xor_i16_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xor_i16_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var16
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var16
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldeor2_relax
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw xor i16* @var16, i16 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: ldeorh w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_load_xor_i32_monotonic(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xor_i32_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xor_i32_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldeor4_relax
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw xor i32* @var32, i32 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldeor w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_load_xor_i64_monotonic(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xor_i64_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xor_i64_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldeor8_relax
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw xor i64* @var64, i64 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldeor x[[OLD:[0-9]+]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define void @test_atomic_load_xor_i32_noret_monotonic(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xor_i32_noret_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xor_i32_noret_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldeor4_relax
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw xor i32* @var32, i32 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldeor w{{[0-9]+}}, w{{[1-9][0-9]*}}, [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_load_xor_i64_noret_monotonic(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xor_i64_noret_monotonic:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xor_i64_noret_monotonic:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldeor8_relax
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw xor i64* @var64, i64 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldeor x{{[0-9]+}}, x{{[1-9][0-9]*}}, [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define i8 @test_atomic_load_xor_i8_release(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xor_i8_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xor_i8_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var8
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var8
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldeor1_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw xor i8* @var8, i8 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: ldeorlb w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_load_xor_i16_release(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xor_i16_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xor_i16_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var16
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var16
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldeor2_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw xor i16* @var16, i16 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: ldeorlh w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_load_xor_i32_release(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xor_i32_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xor_i32_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldeor4_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw xor i32* @var32, i32 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldeorl w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_load_xor_i64_release(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xor_i64_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xor_i64_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldeor8_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw xor i64* @var64, i64 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldeorl x[[OLD:[0-9]+]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define void @test_atomic_load_xor_i32_noret_release(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xor_i32_noret_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xor_i32_noret_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldeor4_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw xor i32* @var32, i32 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldeorl w{{[0-9]+}}, w{{[1-9][0-9]*}}, [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_load_xor_i64_noret_release(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xor_i64_noret_release:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xor_i64_noret_release:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldeor8_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw xor i64* @var64, i64 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldeorl x{{[0-9]+}}, x{{[1-9][0-9]*}}, [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define i8 @test_atomic_load_xor_i8_seq_cst(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xor_i8_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xor_i8_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var8
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var8
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldeor1_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw xor i8* @var8, i8 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: ldeoralb w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_load_xor_i16_seq_cst(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xor_i16_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xor_i16_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var16
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var16
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldeor2_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw xor i16* @var16, i16 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: ldeoralh w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_load_xor_i32_seq_cst(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xor_i32_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xor_i32_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldeor4_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw xor i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldeoral w[[OLD:[0-9]+]], w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_load_xor_i64_seq_cst(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xor_i64_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xor_i64_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldeor8_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   %old = atomicrmw xor i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldeoral x[[OLD:[0-9]+]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define void @test_atomic_load_xor_i32_noret_seq_cst(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xor_i32_noret_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xor_i32_noret_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var32
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldeor4_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw xor i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: ldeoral w0, w[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_load_xor_i64_noret_seq_cst(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xor_i64_noret_seq_cst:
; OUTLINE-ATOMICS-LABEL: test_atomic_load_xor_i64_noret_seq_cst:
; OUTLINE-ATOMICS:       // %bb.0:
; OUTLINE-ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE-ATOMICS-NEXT:    adrp x1, var64
; OUTLINE-ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE-ATOMICS-NEXT:    bl __aarch64_ldeor8_acq_rel
; OUTLINE-ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE-ATOMICS-NEXT:    ret
   atomicrmw xor i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldeoral x0, x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}


