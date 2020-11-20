; RUN: llc -mtriple=aarch64-none-linux-gnu -disable-post-ra -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-none-linux-gnu -disable-post-ra -verify-machineinstrs < %s | FileCheck %s --check-prefix=CHECK-REG
; RUN: llc -mtriple=aarch64-none-linux-gnu -disable-post-ra -verify-machineinstrs -mattr=+outline-atomics < %s | FileCheck %s --check-prefix=OUTLINE_ATOMICS


; Point of CHECK-REG is to make sure UNPREDICTABLE instructions aren't created
; (i.e. reusing a register for status & data in store exclusive).
; CHECK-REG-NOT: stlxrb w[[NEW:[0-9]+]], w[[NEW]], [x{{[0-9]+}}]
; CHECK-REG-NOT: stlxrb w[[NEW:[0-9]+]], x[[NEW]], [x{{[0-9]+}}]

@var8 = global i8 0
@var16 = global i16 0
@var32 = global i32 0
@var64 = global i64 0

define i8 @test_atomic_load_add_i8(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i8:
; OUTLINE_ATOMICS-LABEL: test_atomic_load_add_i8:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE_ATOMICS-NEXT:    adrp x1, var8
; OUTLINE_ATOMICS-NEXT:    add x1, x1, :lo12:var8
; OUTLINE_ATOMICS-NEXT:    bl __aarch64_ldadd1_acq_rel
; OUTLINE_ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE_ATOMICS-NEXT:    ret
   %old = atomicrmw add i8* @var8, i8 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldaxrb w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: add [[NEW:w[0-9]+]], w[[OLD]], w0
; CHECK-NEXT: stlxrb [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov {{[xw]}}0, {{[xw]}}[[OLD]]
   ret i8 %old
}

define i16 @test_atomic_load_add_i16(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i16:
; OUTLINE_ATOMICS-LABEL: test_atomic_load_add_i16:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE_ATOMICS-NEXT:    adrp x1, var16
; OUTLINE_ATOMICS-NEXT:    add x1, x1, :lo12:var16
; OUTLINE_ATOMICS-NEXT:    bl __aarch64_ldadd2_acq
; OUTLINE_ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE_ATOMICS-NEXT:    ret
   %old = atomicrmw add i16* @var16, i16 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: .LBB{{[0-9]+}}_1:
; ; CHECK: ldaxrh w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: add [[NEW:w[0-9]+]], w[[OLD]], w0
; CHECK-NEXT: stxrh [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov {{[xw]}}0, {{[xw]}}[[OLD]]
   ret i16 %old
}

define i32 @test_atomic_load_add_i32(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i32:
; OUTLINE_ATOMICS-LABEL: test_atomic_load_add_i32:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE_ATOMICS-NEXT:    adrp x1, var32
; OUTLINE_ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE_ATOMICS-NEXT:    bl __aarch64_ldadd4_rel
; OUTLINE_ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE_ATOMICS-NEXT:    ret
   %old = atomicrmw add i32* @var32, i32 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: .LBB{{[0-9]+}}_1:
; ; CHECK: ldxr w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: add [[NEW:w[0-9]+]], w[[OLD]], w0
; CHECK-NEXT: stlxr [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov {{[xw]}}0, {{[xw]}}[[OLD]]
   ret i32 %old
}

define i64 @test_atomic_load_add_i64(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i64:
; OUTLINE_ATOMICS-LABEL: test_atomic_load_add_i64:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE_ATOMICS-NEXT:    adrp x1, var64
; OUTLINE_ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE_ATOMICS-NEXT:    bl __aarch64_ldadd8_relax
; OUTLINE_ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE_ATOMICS-NEXT:    ret
   %old = atomicrmw add i64* @var64, i64 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: .LBB{{[0-9]+}}_1:
; ; CHECK: ldxr x[[OLD:[0-9]+]], [x[[ADDR]]]
  ; x0 below is a reasonable guess but could change: it certainly comes into the
  ; function there.
; CHECK-NEXT: add [[NEW:x[0-9]+]], x[[OLD]], x0
; CHECK-NEXT: stxr [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i64 %old
}

define i8 @test_atomic_load_sub_i8(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i8:
; OUTLINE_ATOMICS-LABEL: test_atomic_load_sub_i8:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE_ATOMICS-NEXT:    neg w0, w0
; OUTLINE_ATOMICS-NEXT:    adrp x1, var8
; OUTLINE_ATOMICS-NEXT:    add x1, x1, :lo12:var8
; OUTLINE_ATOMICS-NEXT:    bl __aarch64_ldadd1_relax
; OUTLINE_ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE_ATOMICS-NEXT:    ret
   %old = atomicrmw sub i8* @var8, i8 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: .LBB{{[0-9]+}}_1:
; ; CHECK: ldxrb w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: sub [[NEW:w[0-9]+]], w[[OLD]], w0
; CHECK-NEXT: stxrb [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov {{[xw]}}0, {{[xw]}}[[OLD]]
   ret i8 %old
}

define i16 @test_atomic_load_sub_i16(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i16:
; OUTLINE_ATOMICS-LABEL: test_atomic_load_sub_i16:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE_ATOMICS-NEXT:    neg w0, w0
; OUTLINE_ATOMICS-NEXT:    adrp x1, var16
; OUTLINE_ATOMICS-NEXT:    add x1, x1, :lo12:var16
; OUTLINE_ATOMICS-NEXT:    bl __aarch64_ldadd2_rel
; OUTLINE_ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE_ATOMICS-NEXT:    ret
   %old = atomicrmw sub i16* @var16, i16 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: .LBB{{[0-9]+}}_1:
; ; CHECK: ldxrh w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: sub [[NEW:w[0-9]+]], w[[OLD]], w0
; CHECK-NEXT: stlxrh [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov {{[xw]}}0, {{[xw]}}[[OLD]]
   ret i16 %old
}

define i32 @test_atomic_load_sub_i32(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i32:
; OUTLINE_ATOMICS-LABEL: test_atomic_load_sub_i32:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE_ATOMICS-NEXT:    neg w0, w0
; OUTLINE_ATOMICS-NEXT:    adrp x1, var32
; OUTLINE_ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE_ATOMICS-NEXT:    bl __aarch64_ldadd4_acq
; OUTLINE_ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE_ATOMICS-NEXT:    ret
   %old = atomicrmw sub i32* @var32, i32 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: .LBB{{[0-9]+}}_1:
; ; CHECK: ldaxr w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: sub [[NEW:w[0-9]+]], w[[OLD]], w0
; CHECK-NEXT: stxr [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov {{[xw]}}0, {{[xw]}}[[OLD]]
   ret i32 %old
}

define i64 @test_atomic_load_sub_i64(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i64:
; OUTLINE_ATOMICS-LABEL: test_atomic_load_sub_i64:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE_ATOMICS-NEXT:    neg x0, x0
; OUTLINE_ATOMICS-NEXT:    adrp x1, var64
; OUTLINE_ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE_ATOMICS-NEXT:    bl __aarch64_ldadd8_acq_rel
; OUTLINE_ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE_ATOMICS-NEXT:    ret
   %old = atomicrmw sub i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: .LBB{{[0-9]+}}_1:
; ; CHECK: ldaxr x[[OLD:[0-9]+]], [x[[ADDR]]]
  ; x0 below is a reasonable guess but could change: it certainly comes into the
  ; function there.
; CHECK-NEXT: sub [[NEW:x[0-9]+]], x[[OLD]], x0
; CHECK-NEXT: stlxr [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i64 %old
}

define i8 @test_atomic_load_and_i8(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i8:
; OUTLINE_ATOMICS-LABEL: test_atomic_load_and_i8:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE_ATOMICS-NEXT:    mvn w0, w0
; OUTLINE_ATOMICS-NEXT:    adrp x1, var8
; OUTLINE_ATOMICS-NEXT:    add x1, x1, :lo12:var8
; OUTLINE_ATOMICS-NEXT:    bl __aarch64_ldclr1_rel
; OUTLINE_ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE_ATOMICS-NEXT:    ret
   %old = atomicrmw and i8* @var8, i8 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: .LBB{{[0-9]+}}_1:
; ; CHECK: ldxrb w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: and [[NEW:w[0-9]+]], w[[OLD]], w0
; CHECK-NEXT: stlxrb [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov {{[xw]}}0, {{[xw]}}[[OLD]]
   ret i8 %old
}

define i16 @test_atomic_load_and_i16(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i16:
; OUTLINE_ATOMICS-LABEL: test_atomic_load_and_i16:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE_ATOMICS-NEXT:    mvn w0, w0
; OUTLINE_ATOMICS-NEXT:    adrp x1, var16
; OUTLINE_ATOMICS-NEXT:    add x1, x1, :lo12:var16
; OUTLINE_ATOMICS-NEXT:    bl __aarch64_ldclr2_relax
; OUTLINE_ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE_ATOMICS-NEXT:    ret
   %old = atomicrmw and i16* @var16, i16 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: .LBB{{[0-9]+}}_1:
; ; CHECK: ldxrh w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: and [[NEW:w[0-9]+]], w[[OLD]], w0
; CHECK-NEXT: stxrh [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov {{[xw]}}0, {{[xw]}}[[OLD]]
   ret i16 %old
}

define i32 @test_atomic_load_and_i32(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i32:
; OUTLINE_ATOMICS-LABEL: test_atomic_load_and_i32:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE_ATOMICS-NEXT:    mvn w0, w0
; OUTLINE_ATOMICS-NEXT:    adrp x1, var32
; OUTLINE_ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE_ATOMICS-NEXT:    bl __aarch64_ldclr4_acq_rel
; OUTLINE_ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE_ATOMICS-NEXT:    ret
   %old = atomicrmw and i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: .LBB{{[0-9]+}}_1:
; ; CHECK: ldaxr w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: and [[NEW:w[0-9]+]], w[[OLD]], w0
; CHECK-NEXT: stlxr [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov {{[xw]}}0, {{[xw]}}[[OLD]]
   ret i32 %old
}

define i64 @test_atomic_load_and_i64(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i64:
; OUTLINE_ATOMICS-LABEL: test_atomic_load_and_i64:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE_ATOMICS-NEXT:    mvn x0, x0
; OUTLINE_ATOMICS-NEXT:    adrp x1, var64
; OUTLINE_ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE_ATOMICS-NEXT:    bl __aarch64_ldclr8_acq
; OUTLINE_ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE_ATOMICS-NEXT:    ret
   %old = atomicrmw and i64* @var64, i64 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: .LBB{{[0-9]+}}_1:
; ; CHECK: ldaxr x[[OLD:[0-9]+]], [x[[ADDR]]]
  ; x0 below is a reasonable guess but could change: it certainly comes into the
  ; function there.
; CHECK-NEXT: and [[NEW:x[0-9]+]], x[[OLD]], x0
; CHECK-NEXT: stxr [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i64 %old
}

define i8 @test_atomic_load_or_i8(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_or_i8:
; OUTLINE_ATOMICS-LABEL: test_atomic_load_or_i8:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE_ATOMICS-NEXT:    adrp x1, var8
; OUTLINE_ATOMICS-NEXT:    add x1, x1, :lo12:var8
; OUTLINE_ATOMICS-NEXT:    bl __aarch64_ldset1_acq_rel
; OUTLINE_ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE_ATOMICS-NEXT:    ret
   %old = atomicrmw or i8* @var8, i8 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: .LBB{{[0-9]+}}_1:
; ; CHECK: ldaxrb w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: orr [[NEW:w[0-9]+]], w[[OLD]], w0
; CHECK-NEXT: stlxrb [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov {{[xw]}}0, {{[xw]}}[[OLD]]
   ret i8 %old
}

define i16 @test_atomic_load_or_i16(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_or_i16:
; OUTLINE_ATOMICS-LABEL: test_atomic_load_or_i16:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE_ATOMICS-NEXT:    adrp x1, var16
; OUTLINE_ATOMICS-NEXT:    add x1, x1, :lo12:var16
; OUTLINE_ATOMICS-NEXT:    bl __aarch64_ldset2_relax
; OUTLINE_ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE_ATOMICS-NEXT:    ret
   %old = atomicrmw or i16* @var16, i16 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: .LBB{{[0-9]+}}_1:
; ; CHECK: ldxrh w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: orr [[NEW:w[0-9]+]], w[[OLD]], w0
; CHECK-NEXT: stxrh [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov {{[xw]}}0, {{[xw]}}[[OLD]]
   ret i16 %old
}

define i32 @test_atomic_load_or_i32(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_or_i32:
; OUTLINE_ATOMICS-LABEL: test_atomic_load_or_i32:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE_ATOMICS-NEXT:    adrp x1, var32
; OUTLINE_ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE_ATOMICS-NEXT:    bl __aarch64_ldset4_acq
; OUTLINE_ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE_ATOMICS-NEXT:    ret
   %old = atomicrmw or i32* @var32, i32 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: .LBB{{[0-9]+}}_1:
; ; CHECK: ldaxr w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: orr [[NEW:w[0-9]+]], w[[OLD]], w0
; CHECK-NEXT: stxr [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov {{[xw]}}0, {{[xw]}}[[OLD]]
   ret i32 %old
}

define i64 @test_atomic_load_or_i64(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_or_i64:
; OUTLINE_ATOMICS-LABEL: test_atomic_load_or_i64:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE_ATOMICS-NEXT:    adrp x1, var64
; OUTLINE_ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE_ATOMICS-NEXT:    bl __aarch64_ldset8_rel
; OUTLINE_ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE_ATOMICS-NEXT:    ret
   %old = atomicrmw or i64* @var64, i64 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: .LBB{{[0-9]+}}_1:
; ; CHECK: ldxr x[[OLD:[0-9]+]], [x[[ADDR]]]
  ; x0 below is a reasonable guess but could change: it certainly comes into the
  ; function there.
; CHECK-NEXT: orr [[NEW:x[0-9]+]], x[[OLD]], x0
; CHECK-NEXT: stlxr [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i64 %old
}

define i8 @test_atomic_load_xor_i8(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xor_i8:
; OUTLINE_ATOMICS-LABEL: test_atomic_load_xor_i8:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE_ATOMICS-NEXT:    adrp x1, var8
; OUTLINE_ATOMICS-NEXT:    add x1, x1, :lo12:var8
; OUTLINE_ATOMICS-NEXT:    bl __aarch64_ldeor1_acq
; OUTLINE_ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE_ATOMICS-NEXT:    ret
   %old = atomicrmw xor i8* @var8, i8 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: .LBB{{[0-9]+}}_1:
; ; CHECK: ldaxrb w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: eor [[NEW:w[0-9]+]], w[[OLD]], w0
; CHECK-NEXT: stxrb [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov {{[xw]}}0, {{[xw]}}[[OLD]]
   ret i8 %old
}

define i16 @test_atomic_load_xor_i16(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xor_i16:
; OUTLINE_ATOMICS-LABEL: test_atomic_load_xor_i16:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE_ATOMICS-NEXT:    adrp x1, var16
; OUTLINE_ATOMICS-NEXT:    add x1, x1, :lo12:var16
; OUTLINE_ATOMICS-NEXT:    bl __aarch64_ldeor2_rel
; OUTLINE_ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE_ATOMICS-NEXT:    ret
   %old = atomicrmw xor i16* @var16, i16 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: .LBB{{[0-9]+}}_1:
; ; CHECK: ldxrh w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: eor [[NEW:w[0-9]+]], w[[OLD]], w0
; CHECK-NEXT: stlxrh [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov {{[xw]}}0, {{[xw]}}[[OLD]]
   ret i16 %old
}

define i32 @test_atomic_load_xor_i32(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xor_i32:
; OUTLINE_ATOMICS-LABEL: test_atomic_load_xor_i32:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE_ATOMICS-NEXT:    adrp x1, var32
; OUTLINE_ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE_ATOMICS-NEXT:    bl __aarch64_ldeor4_acq_rel
; OUTLINE_ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE_ATOMICS-NEXT:    ret
   %old = atomicrmw xor i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: .LBB{{[0-9]+}}_1:
; ; CHECK: ldaxr w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: eor [[NEW:w[0-9]+]], w[[OLD]], w0
; CHECK-NEXT: stlxr [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov {{[xw]}}0, {{[xw]}}[[OLD]]
   ret i32 %old
}

define i64 @test_atomic_load_xor_i64(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xor_i64:
; OUTLINE_ATOMICS-LABEL: test_atomic_load_xor_i64:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE_ATOMICS-NEXT:    adrp x1, var64
; OUTLINE_ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE_ATOMICS-NEXT:    bl __aarch64_ldeor8_relax
; OUTLINE_ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE_ATOMICS-NEXT:    ret
   %old = atomicrmw xor i64* @var64, i64 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: .LBB{{[0-9]+}}_1:
; ; CHECK: ldxr x[[OLD:[0-9]+]], [x[[ADDR]]]
  ; x0 below is a reasonable guess but could change: it certainly comes into the
  ; function there.
; CHECK-NEXT: eor [[NEW:x[0-9]+]], x[[OLD]], x0
; CHECK-NEXT: stxr [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i64 %old
}

define i8 @test_atomic_load_xchg_i8(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xchg_i8:
; OUTLINE_ATOMICS-LABEL: test_atomic_load_xchg_i8:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE_ATOMICS-NEXT:    adrp x1, var8
; OUTLINE_ATOMICS-NEXT:    add x1, x1, :lo12:var8
; OUTLINE_ATOMICS-NEXT:    bl __aarch64_swp1_relax
; OUTLINE_ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE_ATOMICS-NEXT:    ret
   %old = atomicrmw xchg i8* @var8, i8 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: .LBB{{[0-9]+}}_1:
; ; CHECK: ldxrb w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ; function there.
; CHECK-NEXT: stxrb [[STATUS:w[0-9]+]], w0, [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov {{[xw]}}0, {{[xw]}}[[OLD]]
   ret i8 %old
}

define i16 @test_atomic_load_xchg_i16(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xchg_i16:
; OUTLINE_ATOMICS-LABEL: test_atomic_load_xchg_i16:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE_ATOMICS-NEXT:    adrp x1, var16
; OUTLINE_ATOMICS-NEXT:    add x1, x1, :lo12:var16
; OUTLINE_ATOMICS-NEXT:    bl __aarch64_swp2_acq_rel
; OUTLINE_ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE_ATOMICS-NEXT:    ret
   %old = atomicrmw xchg i16* @var16, i16 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: .LBB{{[0-9]+}}_1:
; ; CHECK: ldaxrh w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ; function there.
; CHECK-NEXT: stlxrh [[STATUS:w[0-9]+]], w0, [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov {{[xw]}}0, {{[xw]}}[[OLD]]
   ret i16 %old
}

define i32 @test_atomic_load_xchg_i32(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xchg_i32:
; CHECK: mov {{[xw]}}8, w[[OLD:[0-9]+]]
; OUTLINE_ATOMICS-LABEL: test_atomic_load_xchg_i32:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE_ATOMICS-NEXT:    adrp x1, var32
; OUTLINE_ATOMICS-NEXT:    add x1, x1, :lo12:var32
; OUTLINE_ATOMICS-NEXT:    bl __aarch64_swp4_rel
; OUTLINE_ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE_ATOMICS-NEXT:    ret
   %old = atomicrmw xchg i32* @var32, i32 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: .LBB{{[0-9]+}}_1:
; ; CHECK: ldxr {{[xw]}}[[OLD]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: stlxr [[STATUS:w[0-9]+]], w8, [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb
   ret i32 %old
}

define i64 @test_atomic_load_xchg_i64(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xchg_i64:
; OUTLINE_ATOMICS-LABEL: test_atomic_load_xchg_i64:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE_ATOMICS-NEXT:    adrp x1, var64
; OUTLINE_ATOMICS-NEXT:    add x1, x1, :lo12:var64
; OUTLINE_ATOMICS-NEXT:    bl __aarch64_swp8_acq
; OUTLINE_ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE_ATOMICS-NEXT:    ret
   %old = atomicrmw xchg i64* @var64, i64 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: .LBB{{[0-9]+}}_1:
; ; CHECK: ldaxr x[[OLD:[0-9]+]], [x[[ADDR]]]
  ; x0 below is a reasonable guess but could change: it certainly comes into the
  ; function there.
; CHECK-NEXT: stxr [[STATUS:w[0-9]+]], x0, [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i64 %old
}


define i8 @test_atomic_load_min_i8(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_min_i8:
; OUTLINE_ATOMICS-LABEL: test_atomic_load_min_i8:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    adrp x9, var8
; OUTLINE_ATOMICS-NEXT:    add x9, x9, :lo12:var8
; OUTLINE_ATOMICS-NEXT:  .LBB24_1: // %atomicrmw.start
; OUTLINE_ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE_ATOMICS-NEXT:    ldaxrb w10, [x9]
; OUTLINE_ATOMICS-NEXT:    sxtb w8, w10
; OUTLINE_ATOMICS-NEXT:    cmp w8, w0, sxtb
; OUTLINE_ATOMICS-NEXT:    csel w10, w10, w0, le
; OUTLINE_ATOMICS-NEXT:    stxrb w11, w10, [x9]
; OUTLINE_ATOMICS-NEXT:    cbnz w11, .LBB24_1
; OUTLINE_ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE_ATOMICS-NEXT:    mov w0, w8
; OUTLINE_ATOMICS-NEXT:    ret
   %old = atomicrmw min i8* @var8, i8 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldaxrb w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.

; CHECK-NEXT: sxtb w[[OLD_EXT:[0-9]+]], w[[OLD]]
; CHECK-NEXT: cmp w[[OLD_EXT]], w0, sxtb
; CHECK-NEXT: csel [[NEW:w[0-9]+]], w[[OLD]], w0, le

; CHECK-NEXT: stxrb [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov {{[xw]}}0, {{[xw]}}[[OLD_EXT]]
   ret i8 %old
}

define i16 @test_atomic_load_min_i16(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_min_i16:
; OUTLINE_ATOMICS-LABEL: test_atomic_load_min_i16:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    adrp x9, var16
; OUTLINE_ATOMICS-NEXT:    add x9, x9, :lo12:var16
; OUTLINE_ATOMICS-NEXT:  .LBB25_1: // %atomicrmw.start
; OUTLINE_ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE_ATOMICS-NEXT:    ldxrh w10, [x9]
; OUTLINE_ATOMICS-NEXT:    sxth w8, w10
; OUTLINE_ATOMICS-NEXT:    cmp w8, w0, sxth
; OUTLINE_ATOMICS-NEXT:    csel w10, w10, w0, le
; OUTLINE_ATOMICS-NEXT:    stlxrh w11, w10, [x9]
; OUTLINE_ATOMICS-NEXT:    cbnz w11, .LBB25_1
; OUTLINE_ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE_ATOMICS-NEXT:    mov w0, w8
; OUTLINE_ATOMICS-NEXT:    ret
   %old = atomicrmw min i16* @var16, i16 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldxrh w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.

; CHECK-NEXT: sxth w[[OLD_EXT:[0-9]+]], w[[OLD]]
; CHECK-NEXT: cmp w[[OLD_EXT]], w0, sxth
; CHECK-NEXT: csel [[NEW:w[0-9]+]], w[[OLD]], w0, le


; CHECK-NEXT: stlxrh [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov {{[xw]}}0, {{[xw]}}[[OLD_EXT]]
   ret i16 %old
}

define i32 @test_atomic_load_min_i32(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_min_i32:
; OUTLINE_ATOMICS-LABEL: test_atomic_load_min_i32:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    adrp x9, var32
; OUTLINE_ATOMICS-NEXT:    add x9, x9, :lo12:var32
; OUTLINE_ATOMICS-NEXT:  .LBB26_1: // %atomicrmw.start
; OUTLINE_ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE_ATOMICS-NEXT:    ldxr w8, [x9]
; OUTLINE_ATOMICS-NEXT:    cmp w8, w0
; OUTLINE_ATOMICS-NEXT:    csel w10, w8, w0, le
; OUTLINE_ATOMICS-NEXT:    stxr w11, w10, [x9]
; OUTLINE_ATOMICS-NEXT:    cbnz w11, .LBB26_1
; OUTLINE_ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE_ATOMICS-NEXT:    mov w0, w8
; OUTLINE_ATOMICS-NEXT:    ret
   %old = atomicrmw min i32* @var32, i32 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldxr w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.

; CHECK-NEXT: cmp w[[OLD]], w0
; CHECK-NEXT: csel [[NEW:w[0-9]+]], w[[OLD]], w0, le


; CHECK-NEXT: stxr [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov {{[xw]}}0, {{[xw]}}[[OLD]]
   ret i32 %old
}

define i64 @test_atomic_load_min_i64(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_min_i64:
; OUTLINE_ATOMICS-LABEL: test_atomic_load_min_i64:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    adrp x9, var64
; OUTLINE_ATOMICS-NEXT:    add x9, x9, :lo12:var64
; OUTLINE_ATOMICS-NEXT:  .LBB27_1: // %atomicrmw.start
; OUTLINE_ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE_ATOMICS-NEXT:    ldaxr x8, [x9]
; OUTLINE_ATOMICS-NEXT:    cmp x8, x0
; OUTLINE_ATOMICS-NEXT:    csel x10, x8, x0, le
; OUTLINE_ATOMICS-NEXT:    stlxr w11, x10, [x9]
; OUTLINE_ATOMICS-NEXT:    cbnz w11, .LBB27_1
; OUTLINE_ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE_ATOMICS-NEXT:    mov x0, x8
; OUTLINE_ATOMICS-NEXT:    ret
   %old = atomicrmw min i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldaxr x[[OLD:[0-9]+]], [x[[ADDR]]]
  ; x0 below is a reasonable guess but could change: it certainly comes into the
  ; function there.

; CHECK-NEXT: cmp x[[OLD]], x0
; CHECK-NEXT: csel [[NEW:x[0-9]+]], x[[OLD]], x0, le


; CHECK-NEXT: stlxr [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i64 %old
}

define i8 @test_atomic_load_max_i8(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_max_i8:
; OUTLINE_ATOMICS-LABEL: test_atomic_load_max_i8:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    adrp x9, var8
; OUTLINE_ATOMICS-NEXT:    add x9, x9, :lo12:var8
; OUTLINE_ATOMICS-NEXT:  .LBB28_1: // %atomicrmw.start
; OUTLINE_ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE_ATOMICS-NEXT:    ldaxrb w10, [x9]
; OUTLINE_ATOMICS-NEXT:    sxtb w8, w10
; OUTLINE_ATOMICS-NEXT:    cmp w8, w0, sxtb
; OUTLINE_ATOMICS-NEXT:    csel w10, w10, w0, gt
; OUTLINE_ATOMICS-NEXT:    stlxrb w11, w10, [x9]
; OUTLINE_ATOMICS-NEXT:    cbnz w11, .LBB28_1
; OUTLINE_ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE_ATOMICS-NEXT:    mov w0, w8
; OUTLINE_ATOMICS-NEXT:    ret
   %old = atomicrmw max i8* @var8, i8 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldaxrb w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.

; CHECK-NEXT: sxtb w[[OLD_EXT:[0-9]+]], w[[OLD]]
; CHECK-NEXT: cmp w[[OLD_EXT]], w0, sxtb
; CHECK-NEXT: csel [[NEW:w[0-9]+]], w[[OLD]], w0, gt


; CHECK-NEXT: stlxrb [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov {{[xw]}}0, {{[xw]}}[[OLD_EXT]]
   ret i8 %old
}

define i16 @test_atomic_load_max_i16(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_max_i16:
; OUTLINE_ATOMICS-LABEL: test_atomic_load_max_i16:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    adrp x9, var16
; OUTLINE_ATOMICS-NEXT:    add x9, x9, :lo12:var16
; OUTLINE_ATOMICS-NEXT:  .LBB29_1: // %atomicrmw.start
; OUTLINE_ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE_ATOMICS-NEXT:    ldaxrh w10, [x9]
; OUTLINE_ATOMICS-NEXT:    sxth w8, w10
; OUTLINE_ATOMICS-NEXT:    cmp w8, w0, sxth
; OUTLINE_ATOMICS-NEXT:    csel w10, w10, w0, gt
; OUTLINE_ATOMICS-NEXT:    stxrh w11, w10, [x9]
; OUTLINE_ATOMICS-NEXT:    cbnz w11, .LBB29_1
; OUTLINE_ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE_ATOMICS-NEXT:    mov w0, w8
; OUTLINE_ATOMICS-NEXT:    ret
   %old = atomicrmw max i16* @var16, i16 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldaxrh w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.

; CHECK-NEXT: sxth w[[OLD_EXT:[0-9]+]], w[[OLD]]
; CHECK-NEXT: cmp w[[OLD_EXT]], w0, sxth
; CHECK-NEXT: csel [[NEW:w[0-9]+]], w[[OLD]], w0, gt


; CHECK-NEXT: stxrh [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov {{[xw]}}0, {{[xw]}}[[OLD_EXT]]
   ret i16 %old
}

define i32 @test_atomic_load_max_i32(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_max_i32:
; OUTLINE_ATOMICS-LABEL: test_atomic_load_max_i32:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    adrp x9, var32
; OUTLINE_ATOMICS-NEXT:    add x9, x9, :lo12:var32
; OUTLINE_ATOMICS-NEXT:  .LBB30_1: // %atomicrmw.start
; OUTLINE_ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE_ATOMICS-NEXT:    ldxr w8, [x9]
; OUTLINE_ATOMICS-NEXT:    cmp w8, w0
; OUTLINE_ATOMICS-NEXT:    csel w10, w8, w0, gt
; OUTLINE_ATOMICS-NEXT:    stlxr w11, w10, [x9]
; OUTLINE_ATOMICS-NEXT:    cbnz w11, .LBB30_1
; OUTLINE_ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE_ATOMICS-NEXT:    mov w0, w8
; OUTLINE_ATOMICS-NEXT:    ret
   %old = atomicrmw max i32* @var32, i32 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldxr w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.

; CHECK-NEXT: cmp w[[OLD]], w0
; CHECK-NEXT: csel [[NEW:w[0-9]+]], w[[OLD]], w0, gt


; CHECK-NEXT: stlxr [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov {{[xw]}}0, {{[xw]}}[[OLD]]
   ret i32 %old
}

define i64 @test_atomic_load_max_i64(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_max_i64:
; OUTLINE_ATOMICS-LABEL: test_atomic_load_max_i64:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    adrp x9, var64
; OUTLINE_ATOMICS-NEXT:    add x9, x9, :lo12:var64
; OUTLINE_ATOMICS-NEXT:  .LBB31_1: // %atomicrmw.start
; OUTLINE_ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE_ATOMICS-NEXT:    ldxr x8, [x9]
; OUTLINE_ATOMICS-NEXT:    cmp x8, x0
; OUTLINE_ATOMICS-NEXT:    csel x10, x8, x0, gt
; OUTLINE_ATOMICS-NEXT:    stxr w11, x10, [x9]
; OUTLINE_ATOMICS-NEXT:    cbnz w11, .LBB31_1
; OUTLINE_ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE_ATOMICS-NEXT:    mov x0, x8
; OUTLINE_ATOMICS-NEXT:    ret
   %old = atomicrmw max i64* @var64, i64 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldxr x[[OLD:[0-9]+]], [x[[ADDR]]]
  ; x0 below is a reasonable guess but could change: it certainly comes into the
  ; function there.

; CHECK-NEXT: cmp x[[OLD]], x0
; CHECK-NEXT: csel [[NEW:x[0-9]+]], x[[OLD]], x0, gt


; CHECK-NEXT: stxr [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i64 %old
}

define i8 @test_atomic_load_umin_i8(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umin_i8:
; OUTLINE_ATOMICS-LABEL: test_atomic_load_umin_i8:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    adrp x9, var8
; OUTLINE_ATOMICS-NEXT:    add x9, x9, :lo12:var8
; OUTLINE_ATOMICS-NEXT:  .LBB32_1: // %atomicrmw.start
; OUTLINE_ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE_ATOMICS-NEXT:    ldxrb w8, [x9]
; OUTLINE_ATOMICS-NEXT:    cmp w8, w0, uxtb
; OUTLINE_ATOMICS-NEXT:    csel w10, w8, w0, ls
; OUTLINE_ATOMICS-NEXT:    stxrb w11, w10, [x9]
; OUTLINE_ATOMICS-NEXT:    cbnz w11, .LBB32_1
; OUTLINE_ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE_ATOMICS-NEXT:    mov w0, w8
; OUTLINE_ATOMICS-NEXT:    ret
   %old = atomicrmw umin i8* @var8, i8 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldxrb w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.

; CHECK-NEXT: cmp w[[OLD]], w0, uxtb
; CHECK-NEXT: csel [[NEW:w[0-9]+]], w[[OLD]], w0, ls


; CHECK-NEXT: stxrb [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov {{[xw]}}0, {{[xw]}}[[OLD]]
   ret i8 %old
}

define i16 @test_atomic_load_umin_i16(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umin_i16:
; OUTLINE_ATOMICS-LABEL: test_atomic_load_umin_i16:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    adrp x9, var16
; OUTLINE_ATOMICS-NEXT:    add x9, x9, :lo12:var16
; OUTLINE_ATOMICS-NEXT:  .LBB33_1: // %atomicrmw.start
; OUTLINE_ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE_ATOMICS-NEXT:    ldaxrh w8, [x9]
; OUTLINE_ATOMICS-NEXT:    cmp w8, w0, uxth
; OUTLINE_ATOMICS-NEXT:    csel w10, w8, w0, ls
; OUTLINE_ATOMICS-NEXT:    stxrh w11, w10, [x9]
; OUTLINE_ATOMICS-NEXT:    cbnz w11, .LBB33_1
; OUTLINE_ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE_ATOMICS-NEXT:    mov w0, w8
; OUTLINE_ATOMICS-NEXT:    ret
   %old = atomicrmw umin i16* @var16, i16 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldaxrh w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.

; CHECK-NEXT: cmp w[[OLD]], w0, uxth
; CHECK-NEXT: csel [[NEW:w[0-9]+]], w[[OLD]], w0, ls


; CHECK-NEXT: stxrh [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov {{[xw]}}0, {{[xw]}}[[OLD]]
   ret i16 %old
}

define i32 @test_atomic_load_umin_i32(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umin_i32:
; OUTLINE_ATOMICS-LABEL: test_atomic_load_umin_i32:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    adrp x9, var32
; OUTLINE_ATOMICS-NEXT:    add x9, x9, :lo12:var32
; OUTLINE_ATOMICS-NEXT:  .LBB34_1: // %atomicrmw.start
; OUTLINE_ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE_ATOMICS-NEXT:    ldaxr w8, [x9]
; OUTLINE_ATOMICS-NEXT:    cmp w8, w0
; OUTLINE_ATOMICS-NEXT:    csel w10, w8, w0, ls
; OUTLINE_ATOMICS-NEXT:    stlxr w11, w10, [x9]
; OUTLINE_ATOMICS-NEXT:    cbnz w11, .LBB34_1
; OUTLINE_ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE_ATOMICS-NEXT:    mov w0, w8
; OUTLINE_ATOMICS-NEXT:    ret
   %old = atomicrmw umin i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldaxr w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.

; CHECK-NEXT: cmp w[[OLD]], w0
; CHECK-NEXT: csel [[NEW:w[0-9]+]], w[[OLD]], w0, ls


; CHECK-NEXT: stlxr [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov {{[xw]}}0, {{[xw]}}[[OLD]]
   ret i32 %old
}

define i64 @test_atomic_load_umin_i64(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umin_i64:
; OUTLINE_ATOMICS-LABEL: test_atomic_load_umin_i64:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    adrp x9, var64
; OUTLINE_ATOMICS-NEXT:    add x9, x9, :lo12:var64
; OUTLINE_ATOMICS-NEXT:  .LBB35_1: // %atomicrmw.start
; OUTLINE_ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE_ATOMICS-NEXT:    ldaxr x8, [x9]
; OUTLINE_ATOMICS-NEXT:    cmp x8, x0
; OUTLINE_ATOMICS-NEXT:    csel x10, x8, x0, ls
; OUTLINE_ATOMICS-NEXT:    stlxr w11, x10, [x9]
; OUTLINE_ATOMICS-NEXT:    cbnz w11, .LBB35_1
; OUTLINE_ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE_ATOMICS-NEXT:    mov x0, x8
; OUTLINE_ATOMICS-NEXT:    ret
   %old = atomicrmw umin i64* @var64, i64 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldaxr x[[OLD:[0-9]+]], [x[[ADDR]]]
  ; x0 below is a reasonable guess but could change: it certainly comes into the
  ; function there.

; CHECK-NEXT: cmp x[[OLD]], x0
; CHECK-NEXT: csel [[NEW:x[0-9]+]], x[[OLD]], x0, ls


; CHECK-NEXT: stlxr [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i64 %old
}

define i8 @test_atomic_load_umax_i8(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umax_i8:
; OUTLINE_ATOMICS-LABEL: test_atomic_load_umax_i8:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    adrp x9, var8
; OUTLINE_ATOMICS-NEXT:    add x9, x9, :lo12:var8
; OUTLINE_ATOMICS-NEXT:  .LBB36_1: // %atomicrmw.start
; OUTLINE_ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE_ATOMICS-NEXT:    ldaxrb w8, [x9]
; OUTLINE_ATOMICS-NEXT:    cmp w8, w0, uxtb
; OUTLINE_ATOMICS-NEXT:    csel w10, w8, w0, hi
; OUTLINE_ATOMICS-NEXT:    stlxrb w11, w10, [x9]
; OUTLINE_ATOMICS-NEXT:    cbnz w11, .LBB36_1
; OUTLINE_ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE_ATOMICS-NEXT:    mov w0, w8
; OUTLINE_ATOMICS-NEXT:    ret
   %old = atomicrmw umax i8* @var8, i8 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldaxrb w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.

; CHECK-NEXT: cmp w[[OLD]], w0, uxtb
; CHECK-NEXT: csel [[NEW:w[0-9]+]], w[[OLD]], w0, hi


; CHECK-NEXT: stlxrb [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov {{[xw]}}0, {{[xw]}}[[OLD]]
   ret i8 %old
}

define i16 @test_atomic_load_umax_i16(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umax_i16:
; OUTLINE_ATOMICS-LABEL: test_atomic_load_umax_i16:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    adrp x9, var16
; OUTLINE_ATOMICS-NEXT:    add x9, x9, :lo12:var16
; OUTLINE_ATOMICS-NEXT:  .LBB37_1: // %atomicrmw.start
; OUTLINE_ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE_ATOMICS-NEXT:    ldxrh w8, [x9]
; OUTLINE_ATOMICS-NEXT:    cmp w8, w0, uxth
; OUTLINE_ATOMICS-NEXT:    csel w10, w8, w0, hi
; OUTLINE_ATOMICS-NEXT:    stxrh w11, w10, [x9]
; OUTLINE_ATOMICS-NEXT:    cbnz w11, .LBB37_1
; OUTLINE_ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE_ATOMICS-NEXT:    mov w0, w8
; OUTLINE_ATOMICS-NEXT:    ret
   %old = atomicrmw umax i16* @var16, i16 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldxrh w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.

; CHECK-NEXT: cmp w[[OLD]], w0, uxth
; CHECK-NEXT: csel [[NEW:w[0-9]+]], w[[OLD]], w0, hi


; CHECK-NEXT: stxrh [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov {{[xw]}}0, {{[xw]}}[[OLD]]
   ret i16 %old
}

define i32 @test_atomic_load_umax_i32(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umax_i32:
; OUTLINE_ATOMICS-LABEL: test_atomic_load_umax_i32:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    adrp x9, var32
; OUTLINE_ATOMICS-NEXT:    add x9, x9, :lo12:var32
; OUTLINE_ATOMICS-NEXT:  .LBB38_1: // %atomicrmw.start
; OUTLINE_ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE_ATOMICS-NEXT:    ldaxr w8, [x9]
; OUTLINE_ATOMICS-NEXT:    cmp w8, w0
; OUTLINE_ATOMICS-NEXT:    csel w10, w8, w0, hi
; OUTLINE_ATOMICS-NEXT:    stlxr w11, w10, [x9]
; OUTLINE_ATOMICS-NEXT:    cbnz w11, .LBB38_1
; OUTLINE_ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE_ATOMICS-NEXT:    mov w0, w8
; OUTLINE_ATOMICS-NEXT:    ret
   %old = atomicrmw umax i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldaxr w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.

; CHECK-NEXT: cmp w[[OLD]], w0
; CHECK-NEXT: csel [[NEW:w[0-9]+]], w[[OLD]], w0, hi


; CHECK-NEXT: stlxr [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov {{[xw]}}0, {{[xw]}}[[OLD]]
   ret i32 %old
}

define i64 @test_atomic_load_umax_i64(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umax_i64:
; OUTLINE_ATOMICS-LABEL: test_atomic_load_umax_i64:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    adrp x9, var64
; OUTLINE_ATOMICS-NEXT:    add x9, x9, :lo12:var64
; OUTLINE_ATOMICS-NEXT:  .LBB39_1: // %atomicrmw.start
; OUTLINE_ATOMICS-NEXT:    // =>This Inner Loop Header: Depth=1
; OUTLINE_ATOMICS-NEXT:    ldxr x8, [x9]
; OUTLINE_ATOMICS-NEXT:    cmp x8, x0
; OUTLINE_ATOMICS-NEXT:    csel x10, x8, x0, hi
; OUTLINE_ATOMICS-NEXT:    stlxr w11, x10, [x9]
; OUTLINE_ATOMICS-NEXT:    cbnz w11, .LBB39_1
; OUTLINE_ATOMICS-NEXT:  // %bb.2: // %atomicrmw.end
; OUTLINE_ATOMICS-NEXT:    mov x0, x8
; OUTLINE_ATOMICS-NEXT:    ret
   %old = atomicrmw umax i64* @var64, i64 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldxr x[[OLD:[0-9]+]], [x[[ADDR]]]
  ; x0 below is a reasonable guess but could change: it certainly comes into the
  ; function there.

; CHECK-NEXT: cmp x[[OLD]], x0
; CHECK-NEXT: csel [[NEW:x[0-9]+]], x[[OLD]], x0, hi


; CHECK-NEXT: stlxr [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i64 %old
}

define i8 @test_atomic_cmpxchg_i8(i8 %wanted, i8 %new) nounwind {
; CHECK-LABEL: test_atomic_cmpxchg_i8:
; OUTLINE_ATOMICS-LABEL: test_atomic_cmpxchg_i8:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE_ATOMICS-NEXT:    adrp x2, var8
; OUTLINE_ATOMICS-NEXT:    add x2, x2, :lo12:var8
; OUTLINE_ATOMICS-NEXT:    bl __aarch64_cas1_acq
; OUTLINE_ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE_ATOMICS-NEXT:    ret
   %pair = cmpxchg i8* @var8, i8 %wanted, i8 %new acquire acquire
   %old = extractvalue { i8, i1 } %pair, 0

; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: [[STARTAGAIN:.LBB[0-9]+_[0-9]+]]:
; CHECK: ldaxrb w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: cmp w[[OLD]], w0
; CHECK-NEXT: b.ne [[GET_OUT:.LBB[0-9]+_[0-9]+]]
; CHECK: stxrb [[STATUS:w[0-9]+]], {{w[0-9]+}}, [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], [[STARTAGAIN]]
; CHECK: [[GET_OUT]]:
; CHECK: clrex
; CHECK-NOT: dmb

; CHECK: mov {{[xw]}}0, {{[xw]}}[[OLD]]
   ret i8 %old
}

define i16 @test_atomic_cmpxchg_i16(i16 %wanted, i16 %new) nounwind {
; CHECK-LABEL: test_atomic_cmpxchg_i16:
; OUTLINE_ATOMICS-LABEL: test_atomic_cmpxchg_i16:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE_ATOMICS-NEXT:    adrp x2, var16
; OUTLINE_ATOMICS-NEXT:    add x2, x2, :lo12:var16
; OUTLINE_ATOMICS-NEXT:    bl __aarch64_cas2_acq_rel
; OUTLINE_ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE_ATOMICS-NEXT:    ret
   %pair = cmpxchg i16* @var16, i16 %wanted, i16 %new seq_cst seq_cst
   %old = extractvalue { i16, i1 } %pair, 0

; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: [[STARTAGAIN:.LBB[0-9]+_[0-9]+]]:
; CHECK: ldaxrh w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: cmp w[[OLD]], w0
; CHECK-NEXT: b.ne [[GET_OUT:.LBB[0-9]+_[0-9]+]]
; CHECK: stlxrh [[STATUS:w[0-9]+]], {{w[0-9]+}}, [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], [[STARTAGAIN]]
; CHECK: [[GET_OUT]]:
; CHECK: clrex
; CHECK-NOT: dmb

; CHECK: mov {{[xw]}}0, {{[xw]}}[[OLD]]
   ret i16 %old
}

define i32 @test_atomic_cmpxchg_i32(i32 %wanted, i32 %new) nounwind {
; CHECK-LABEL: test_atomic_cmpxchg_i32:
; OUTLINE_ATOMICS-LABEL: test_atomic_cmpxchg_i32:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    str x30, [sp, #-16]! // 8-byte Folded Spill
; OUTLINE_ATOMICS-NEXT:    adrp x2, var32
; OUTLINE_ATOMICS-NEXT:    add x2, x2, :lo12:var32
; OUTLINE_ATOMICS-NEXT:    bl __aarch64_cas4_rel
; OUTLINE_ATOMICS-NEXT:    ldr x30, [sp], #16 // 8-byte Folded Reload
; OUTLINE_ATOMICS-NEXT:    ret
   %pair = cmpxchg i32* @var32, i32 %wanted, i32 %new release monotonic
   %old = extractvalue { i32, i1 } %pair, 0

; CHECK: mov {{[xw]}}[[WANTED:[0-9]+]], {{[xw]}}0

; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: [[STARTAGAIN:.LBB[0-9]+_[0-9]+]]:
; CHECK: ldxr w[[OLD:[0-9]+]], [x[[ADDR]]]
; CHECK-NEXT: cmp w[[OLD]], w[[WANTED]]
; CHECK-NEXT: b.ne [[GET_OUT:.LBB[0-9]+_[0-9]+]]
; CHECK: stlxr [[STATUS:w[0-9]+]], {{w[0-9]+}}, [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], [[STARTAGAIN]]
; CHECK: [[GET_OUT]]:
; CHECK: clrex
; CHECK-NOT: dmb
   ret i32 %old
}

define void @test_atomic_cmpxchg_i64(i64 %wanted, i64 %new) nounwind {
; CHECK-LABEL: test_atomic_cmpxchg_i64:
; OUTLINE_ATOMICS-LABEL: test_atomic_cmpxchg_i64:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    stp x30, x19, [sp, #-16]! // 16-byte Folded Spill
; OUTLINE_ATOMICS-NEXT:    adrp x19, var64
; OUTLINE_ATOMICS-NEXT:    add x19, x19, :lo12:var64
; OUTLINE_ATOMICS-NEXT:    mov x2, x19
; OUTLINE_ATOMICS-NEXT:    bl __aarch64_cas8_relax
; OUTLINE_ATOMICS-NEXT:    str x0, [x19]
; OUTLINE_ATOMICS-NEXT:    ldp x30, x19, [sp], #16 // 16-byte Folded Reload
; OUTLINE_ATOMICS-NEXT:    ret
   %pair = cmpxchg i64* @var64, i64 %wanted, i64 %new monotonic monotonic
   %old = extractvalue { i64, i1 } %pair, 0

; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: [[STARTAGAIN:.LBB[0-9]+_[0-9]+]]:
; CHECK: ldxr x[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: cmp x[[OLD]], x0
; CHECK-NEXT: b.ne [[GET_OUT:.LBB[0-9]+_[0-9]+]]
  ; As above, w1 is a reasonable guess.
; CHECK: stxr [[STATUS:w[0-9]+]], x1, [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], [[STARTAGAIN]]
; CHECK: [[GET_OUT]]:
; CHECK: clrex
; CHECK-NOT: dmb

; CHECK: str x[[OLD]],
   store i64 %old, i64* @var64
   ret void
}

define i8 @test_atomic_load_monotonic_i8() nounwind {
; CHECK-LABEL: test_atomic_load_monotonic_i8:
; OUTLINE_ATOMICS-LABEL: test_atomic_load_monotonic_i8:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    adrp x8, var8
; OUTLINE_ATOMICS-NEXT:    ldrb w0, [x8, :lo12:var8]
; OUTLINE_ATOMICS-NEXT:    ret
  %val = load atomic i8, i8* @var8 monotonic, align 1
; CHECK-NOT: dmb
; CHECK: adrp x[[HIADDR:[0-9]+]], var8
; CHECK: ldrb w0, [x[[HIADDR]], {{#?}}:lo12:var8]
; CHECK-NOT: dmb

  ret i8 %val
}

define i8 @test_atomic_load_monotonic_regoff_i8(i64 %base, i64 %off) nounwind {
; CHECK-LABEL: test_atomic_load_monotonic_regoff_i8:
; OUTLINE_ATOMICS-LABEL: test_atomic_load_monotonic_regoff_i8:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    ldrb w0, [x0, x1]
; OUTLINE_ATOMICS-NEXT:    ret
  %addr_int = add i64 %base, %off
  %addr = inttoptr i64 %addr_int to i8*

  %val = load atomic i8, i8* %addr monotonic, align 1
; CHECK-NOT: dmb
; CHECK: ldrb w0, [x0, x1]
; CHECK-NOT: dmb

  ret i8 %val
}

define i8 @test_atomic_load_acquire_i8() nounwind {
; CHECK-LABEL: test_atomic_load_acquire_i8:
; OUTLINE_ATOMICS-LABEL: test_atomic_load_acquire_i8:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    adrp x8, var8
; OUTLINE_ATOMICS-NEXT:    add x8, x8, :lo12:var8
; OUTLINE_ATOMICS-NEXT:    ldarb w0, [x8]
; OUTLINE_ATOMICS-NEXT:    ret
  %val = load atomic i8, i8* @var8 acquire, align 1
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK-NOT: dmb
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8
; CHECK-NOT: dmb
; CHECK: ldarb w0, [x[[ADDR]]]
; CHECK-NOT: dmb
  ret i8 %val
}

define i8 @test_atomic_load_seq_cst_i8() nounwind {
; CHECK-LABEL: test_atomic_load_seq_cst_i8:
; OUTLINE_ATOMICS-LABEL: test_atomic_load_seq_cst_i8:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    adrp x8, var8
; OUTLINE_ATOMICS-NEXT:    add x8, x8, :lo12:var8
; OUTLINE_ATOMICS-NEXT:    ldarb w0, [x8]
; OUTLINE_ATOMICS-NEXT:    ret
  %val = load atomic i8, i8* @var8 seq_cst, align 1
; CHECK-NOT: dmb
; CHECK: adrp [[HIADDR:x[0-9]+]], var8
; CHECK-NOT: dmb
; CHECK: add x[[ADDR:[0-9]+]], [[HIADDR]], {{#?}}:lo12:var8
; CHECK-NOT: dmb
; CHECK: ldarb w0, [x[[ADDR]]]
; CHECK-NOT: dmb
  ret i8 %val
}

define i16 @test_atomic_load_monotonic_i16() nounwind {
; CHECK-LABEL: test_atomic_load_monotonic_i16:
; OUTLINE_ATOMICS-LABEL: test_atomic_load_monotonic_i16:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    adrp x8, var16
; OUTLINE_ATOMICS-NEXT:    ldrh w0, [x8, :lo12:var16]
; OUTLINE_ATOMICS-NEXT:    ret
  %val = load atomic i16, i16* @var16 monotonic, align 2
; CHECK-NOT: dmb
; CHECK: adrp x[[HIADDR:[0-9]+]], var16
; CHECK-NOT: dmb
; CHECK: ldrh w0, [x[[HIADDR]], {{#?}}:lo12:var16]
; CHECK-NOT: dmb

  ret i16 %val
}

define i32 @test_atomic_load_monotonic_regoff_i32(i64 %base, i64 %off) nounwind {
; CHECK-LABEL: test_atomic_load_monotonic_regoff_i32:
; OUTLINE_ATOMICS-LABEL: test_atomic_load_monotonic_regoff_i32:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    ldr w0, [x0, x1]
; OUTLINE_ATOMICS-NEXT:    ret
  %addr_int = add i64 %base, %off
  %addr = inttoptr i64 %addr_int to i32*

  %val = load atomic i32, i32* %addr monotonic, align 4
; CHECK-NOT: dmb
; CHECK: ldr w0, [x0, x1]
; CHECK-NOT: dmb

  ret i32 %val
}

define i64 @test_atomic_load_seq_cst_i64() nounwind {
; CHECK-LABEL: test_atomic_load_seq_cst_i64:
; OUTLINE_ATOMICS-LABEL: test_atomic_load_seq_cst_i64:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    adrp x8, var64
; OUTLINE_ATOMICS-NEXT:    add x8, x8, :lo12:var64
; OUTLINE_ATOMICS-NEXT:    ldar x0, [x8]
; OUTLINE_ATOMICS-NEXT:    ret
  %val = load atomic i64, i64* @var64 seq_cst, align 8
; CHECK-NOT: dmb
; CHECK: adrp [[HIADDR:x[0-9]+]], var64
; CHECK-NOT: dmb
; CHECK: add x[[ADDR:[0-9]+]], [[HIADDR]], {{#?}}:lo12:var64
; CHECK-NOT: dmb
; CHECK: ldar x0, [x[[ADDR]]]
; CHECK-NOT: dmb
  ret i64 %val
}

define void @test_atomic_store_monotonic_i8(i8 %val) nounwind {
; CHECK-LABEL: test_atomic_store_monotonic_i8:
; OUTLINE_ATOMICS-LABEL: test_atomic_store_monotonic_i8:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    adrp x8, var8
; OUTLINE_ATOMICS-NEXT:    strb w0, [x8, :lo12:var8]
; OUTLINE_ATOMICS-NEXT:    ret
  store atomic i8 %val, i8* @var8 monotonic, align 1
; CHECK: adrp x[[HIADDR:[0-9]+]], var8
; CHECK: strb w0, [x[[HIADDR]], {{#?}}:lo12:var8]

  ret void
}

define void @test_atomic_store_monotonic_regoff_i8(i64 %base, i64 %off, i8 %val) nounwind {
; CHECK-LABEL: test_atomic_store_monotonic_regoff_i8:
; OUTLINE_ATOMICS-LABEL: test_atomic_store_monotonic_regoff_i8:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    strb w2, [x0, x1]
; OUTLINE_ATOMICS-NEXT:    ret
  %addr_int = add i64 %base, %off
  %addr = inttoptr i64 %addr_int to i8*

  store atomic i8 %val, i8* %addr monotonic, align 1
; CHECK: strb w2, [x0, x1]

  ret void
}
define void @test_atomic_store_release_i8(i8 %val) nounwind {
; CHECK-LABEL: test_atomic_store_release_i8:
; OUTLINE_ATOMICS-LABEL: test_atomic_store_release_i8:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    adrp x8, var8
; OUTLINE_ATOMICS-NEXT:    add x8, x8, :lo12:var8
; OUTLINE_ATOMICS-NEXT:    stlrb w0, [x8]
; OUTLINE_ATOMICS-NEXT:    ret
  store atomic i8 %val, i8* @var8 release, align 1
; CHECK-NOT: dmb
; CHECK: adrp [[HIADDR:x[0-9]+]], var8
; CHECK-NOT: dmb
; CHECK: add x[[ADDR:[0-9]+]], [[HIADDR]], {{#?}}:lo12:var8
; CHECK-NOT: dmb
; CHECK: stlrb w0, [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_store_seq_cst_i8(i8 %val) nounwind {
; CHECK-LABEL: test_atomic_store_seq_cst_i8:
; OUTLINE_ATOMICS-LABEL: test_atomic_store_seq_cst_i8:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    adrp x8, var8
; OUTLINE_ATOMICS-NEXT:    add x8, x8, :lo12:var8
; OUTLINE_ATOMICS-NEXT:    stlrb w0, [x8]
; OUTLINE_ATOMICS-NEXT:    ret
  store atomic i8 %val, i8* @var8 seq_cst, align 1
; CHECK-NOT: dmb
; CHECK: adrp [[HIADDR:x[0-9]+]], var8
; CHECK-NOT: dmb
; CHECK: add x[[ADDR:[0-9]+]], [[HIADDR]], {{#?}}:lo12:var8
; CHECK-NOT: dmb
; CHECK: stlrb w0, [x[[ADDR]]]
; CHECK-NOT: dmb

  ret void
}

define void @test_atomic_store_monotonic_i16(i16 %val) nounwind {
; CHECK-LABEL: test_atomic_store_monotonic_i16:
; OUTLINE_ATOMICS-LABEL: test_atomic_store_monotonic_i16:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    adrp x8, var16
; OUTLINE_ATOMICS-NEXT:    strh w0, [x8, :lo12:var16]
; OUTLINE_ATOMICS-NEXT:    ret
  store atomic i16 %val, i16* @var16 monotonic, align 2
; CHECK-NOT: dmb
; CHECK: adrp x[[HIADDR:[0-9]+]], var16
; CHECK-NOT: dmb
; CHECK: strh w0, [x[[HIADDR]], {{#?}}:lo12:var16]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_store_monotonic_regoff_i32(i64 %base, i64 %off, i32 %val) nounwind {
; CHECK-LABEL: test_atomic_store_monotonic_regoff_i32:
; OUTLINE_ATOMICS-LABEL: test_atomic_store_monotonic_regoff_i32:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    str w2, [x0, x1]
; OUTLINE_ATOMICS-NEXT:    ret
  %addr_int = add i64 %base, %off
  %addr = inttoptr i64 %addr_int to i32*

  store atomic i32 %val, i32* %addr monotonic, align 4
; CHECK-NOT: dmb
; CHECK: str w2, [x0, x1]
; CHECK-NOT: dmb

  ret void
}

define void @test_atomic_store_release_i64(i64 %val) nounwind {
; CHECK-LABEL: test_atomic_store_release_i64:
; OUTLINE_ATOMICS-LABEL: test_atomic_store_release_i64:
; OUTLINE_ATOMICS:       // %bb.0:
; OUTLINE_ATOMICS-NEXT:    adrp x8, var64
; OUTLINE_ATOMICS-NEXT:    add x8, x8, :lo12:var64
; OUTLINE_ATOMICS-NEXT:    stlr x0, [x8]
; OUTLINE_ATOMICS-NEXT:    ret
  store atomic i64 %val, i64* @var64 release, align 8
; CHECK-NOT: dmb
; CHECK: adrp [[HIADDR:x[0-9]+]], var64
; CHECK-NOT: dmb
; CHECK: add x[[ADDR:[0-9]+]], [[HIADDR]], {{#?}}:lo12:var64
; CHECK-NOT: dmb
; CHECK: stlr x0, [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}
