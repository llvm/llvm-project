; RUN: llc -mtriple=aarch64-none-linux-gnu -disable-post-ra -verify-machineinstrs -mattr=+lse < %s | FileCheck %s
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
; CHECK-NEXT: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var8
; CHECK-NEXT: casab w0, w1, [x[[ADDR]]]
; CHECK-NEXT: ret

   ret i8 %old
}

define i1 @test_atomic_cmpxchg_i8_1(i8 %wanted, i8 %new) nounwind {
; CHECK-LABEL: test_atomic_cmpxchg_i8_1:
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

define i8 @test_atomic_load_sub_i8_neg_imm() nounwind {
; CHECK-LABEL: test_atomic_load_sub_i8_neg_imm:
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

define i8 @test_atomic_load_and_i8_inv_imm() nounwind {
; CHECK-LABEL: test_atomic_load_and_i8_inv_imm:
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

define i8 @test_atomic_load_add_i8_acq_rel(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i8_acq_rel:
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
   atomicrmw xor i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], {{#?}}:lo12:var64

; CHECK: ldeoral x0, x[[NEW:[0-9]+]], [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}


