; RUN: llc -mtriple=aarch64-none-linux-gnu -disable-post-ra -verify-machineinstrs -mattr=+lse < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-none-linux-gnu -disable-post-ra -verify-machineinstrs -mattr=+lse < %s | FileCheck %s --check-prefix=CHECK-REG

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
   %pair = cmpxchg i8* @var8, i8 %wanted, i8 %new acquire acquire
   %old = extractvalue { i8, i1 } %pair, 0

; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8

; CHECK: casalb w[[NEW:[0-9]+]], w[[OLD:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i8 %old
}

define i16 @test_atomic_cmpxchg_i16(i16 %wanted, i16 %new) nounwind {
; CHECK-LABEL: test_atomic_cmpxchg_i16:
   %pair = cmpxchg i16* @var16, i16 %wanted, i16 %new acquire acquire
   %old = extractvalue { i16, i1 } %pair, 0

; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var16

; CHECK: casalh w0, w1, [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i16 %old
}

define i32 @test_atomic_cmpxchg_i32(i32 %wanted, i32 %new) nounwind {
; CHECK-LABEL: test_atomic_cmpxchg_i32:
   %pair = cmpxchg i32* @var32, i32 %wanted, i32 %new acquire acquire
   %old = extractvalue { i32, i1 } %pair, 0

; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var32

; CHECK: casal w0, w1, [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i32 %old
}

define i64 @test_atomic_cmpxchg_i64(i64 %wanted, i64 %new) nounwind {
; CHECK-LABEL: test_atomic_cmpxchg_i64:
   %pair = cmpxchg i64* @var64, i64 %wanted, i64 %new acquire acquire
   %old = extractvalue { i64, i1 } %pair, 0

; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: casal x0, x1, [x[[ADDR]]]
; CHECK-NOT: dmb

   ret i64 %old
}

define i8 @test_atomic_load_sub_i8(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i8:
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
  atomicrmw sub i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: neg x[[NEG:[0-9]+]], x[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldaddal x[[NEG]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb

  ret void
}

define i8 @test_atomic_load_and_i8(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i8:
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
  %old = atomicrmw and i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: mvn x[[NOT:[0-9]+]], x[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldclral x[[NOT]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret i64 %old
}

define void @test_atomic_load_and_i32_noret(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i32_noret:
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
  atomicrmw and i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: mvn x[[NOT:[0-9]+]], x[[OLD:[0-9]+]]
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldclral x[[NOT]], x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}
