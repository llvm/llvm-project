; RUN: llc -mtriple=aarch64-none-linux-gnu -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-none-linux-gnu -verify-machineinstrs < %s | FileCheck --check-prefix=CHECK-REG %s

@var8 = global i8 0
@var16 = global i16 0
@var32 = global i32 0
@var64 = global i64 0

define i8 @test_atomic_load_add_i8(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i8:
   %old = atomicrmw add i8* @var8, i8 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], #:lo12:var8

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK-NEXT: ldaxrb w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: add [[NEW:w[0-9]+]], w[[OLD]], w0
; CHECK-REG: add w[[NEW:[0-9]+]], w{{[0-9]+}}, w0
; CHECK-REG-NOT: stlxrb w[[NEW]], w[[NEW]], [x{{[0-9]+}}]
; CHECK-NEXT: stlxrb [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i8 %old
}

define i16 @test_atomic_load_add_i16(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i16:
   %old = atomicrmw add i16* @var16, i16 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], #:lo12:var16

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK-NEXT: ldaxrh w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: add [[NEW:w[0-9]+]], w[[OLD]], w0
; CHECK-REG: add w[[NEW:[0-9]+]], w{{[0-9]+}}, w0
; CHECK-REG-NOT: stxrh w[[NEW]], w[[NEW]], [x{{[0-9]+}}]
; CHECK-NEXT: stxrh [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i16 %old
}

define i32 @test_atomic_load_add_i32(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i32:
   %old = atomicrmw add i32* @var32, i32 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], #:lo12:var32

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK-NEXT: ldxr w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: add [[NEW:w[0-9]+]], w[[OLD]], w0
; CHECK-REG: add w[[NEW:[0-9]+]], w{{[0-9]+}}, w0
; CHECK-REG-NOT: stlxr w[[NEW]], w[[NEW]], [x{{[0-9]+}}]
; CHECK-NEXT: stlxr [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i32 %old
}

define i64 @test_atomic_load_add_i64(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i64:
   %old = atomicrmw add i64* @var64, i64 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], #:lo12:var64

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK-NEXT: ldxr x[[OLD:[0-9]+]], [x[[ADDR]]]
  ; x0 below is a reasonable guess but could change: it certainly comes into the
  ; function there.
; CHECK-NEXT: add [[NEW:x[0-9]+]], x[[OLD]], x0
; CHECK-REG: add x[[NEW:[0-9]+]], x{{[0-9]+}}, x0
; CHECK-REG-NOT: stxr w[[NEW]], x[[NEW]], [x{{[0-9]+}}]
; CHECK-NEXT: stxr [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i64 %old
}

define i8 @test_atomic_load_sub_i8(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i8:
   %old = atomicrmw sub i8* @var8, i8 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], #:lo12:var8

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK-NEXT: ldxrb w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: sub [[NEW:w[0-9]+]], w[[OLD]], w0
; CHECK-REG: sub w[[NEW:[0-9]+]], w{{[0-9]+}}, w0
; CHECK-REG-NOT: stxrb w[[NEW]], w[[NEW]], [x{{[0-9]+}}]
; CHECK-NEXT: stxrb [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i8 %old
}

define i16 @test_atomic_load_sub_i16(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i16:
   %old = atomicrmw sub i16* @var16, i16 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], #:lo12:var16

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK-NEXT: ldxrh w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: sub [[NEW:w[0-9]+]], w[[OLD]], w0
; CHECK-REG: sub w[[NEW:[0-9]+]], w{{[0-9]+}}, w0
; CHECK-REG-NOT: stlxrh w[[NEW]], w[[NEW]], [x{{[0-9]+}}]
; CHECK-NEXT: stlxrh [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i16 %old
}

define i32 @test_atomic_load_sub_i32(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i32:
   %old = atomicrmw sub i32* @var32, i32 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], #:lo12:var32

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK-NEXT: ldaxr w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: sub [[NEW:w[0-9]+]], w[[OLD]], w0
; CHECK-REG: sub w[[NEW:[0-9]+]], w{{[0-9]+}}, w0
; CHECK-REG-NOT: stxr w[[NEW]], w[[NEW]], [x{{[0-9]+}}]
; CHECK-NEXT: stxr [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i32 %old
}

define i64 @test_atomic_load_sub_i64(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i64:
   %old = atomicrmw sub i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], #:lo12:var64

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK-NEXT: ldaxr x[[OLD:[0-9]+]], [x[[ADDR]]]
  ; x0 below is a reasonable guess but could change: it certainly comes into the
  ; function there.
; CHECK-NEXT: sub [[NEW:x[0-9]+]], x[[OLD]], x0
; CHECK-REG: sub x[[NEW:[0-9]+]], x{{[0-9]+}}, x0
; CHECK-REG-NOT: stlxr w[[NEW]], x[[NEW]], [x{{[0-9]+}}]
; CHECK-NEXT: stlxr [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i64 %old
}

define i8 @test_atomic_load_and_i8(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i8:
   %old = atomicrmw and i8* @var8, i8 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], #:lo12:var8

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK-NEXT: ldxrb w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: and [[NEW:w[0-9]+]], w[[OLD]], w0
; CHECK-REG: and w[[NEW:[0-9]+]], w{{[0-9]+}}, w0
; CHECK-REG-NOT: stlxrb w[[NEW]], w[[NEW]], [x{{[0-9]+}}]
; CHECK-NEXT: stlxrb [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i8 %old
}

define i16 @test_atomic_load_and_i16(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i16:
   %old = atomicrmw and i16* @var16, i16 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], #:lo12:var16

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK-NEXT: ldxrh w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: and [[NEW:w[0-9]+]], w[[OLD]], w0
; CHECK-REG: and w[[NEW:[0-9]+]], w{{[0-9]+}}, w0
; CHECK-REG-NOT: stxrh w[[NEW]], w[[NEW]], [x{{[0-9]+}}]
; CHECK-NEXT: stxrh [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i16 %old
}

define i32 @test_atomic_load_and_i32(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i32:
   %old = atomicrmw and i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], #:lo12:var32

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK-NEXT: ldaxr w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: and [[NEW:w[0-9]+]], w[[OLD]], w0
; CHECK-REG: and w[[NEW:[0-9]+]], w{{[0-9]+}}, w0
; CHECK-REG-NOT: stlxr w[[NEW]], w[[NEW]], [x{{[0-9]+}}]
; CHECK-NEXT: stlxr [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i32 %old
}

define i64 @test_atomic_load_and_i64(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i64:
   %old = atomicrmw and i64* @var64, i64 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], #:lo12:var64

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK-NEXT: ldaxr x[[OLD:[0-9]+]], [x[[ADDR]]]
  ; x0 below is a reasonable guess but could change: it certainly comes into the
  ; function there.
; CHECK-NEXT: and [[NEW:x[0-9]+]], x[[OLD]], x0
; CHECK-REG: and x[[NEW:[0-9]+]], x{{[0-9]+}}, x0
; CHECK-REG-NOT: stxr w[[NEW]], x[[NEW]], [x{{[0-9]+}}]
; CHECK-NEXT: stxr [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i64 %old
}

define i8 @test_atomic_load_or_i8(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_or_i8:
   %old = atomicrmw or i8* @var8, i8 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], #:lo12:var8

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK-NEXT: ldaxrb w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: orr [[NEW:w[0-9]+]], w[[OLD]], w0
; CHECK-REG: orr w[[NEW:[0-9]+]], w{{[0-9]+}}, w0
; CHECK-REG-NOT: stlxrb w[[NEW]], w[[NEW]], [x{{[0-9]+}}]
; CHECK-NEXT: stlxrb [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i8 %old
}

define i16 @test_atomic_load_or_i16(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_or_i16:
   %old = atomicrmw or i16* @var16, i16 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], #:lo12:var16

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK-NEXT: ldxrh w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: orr [[NEW:w[0-9]+]], w[[OLD]], w0
; CHECK-REG: orr w[[NEW:[0-9]+]], w{{[0-9]+}}, w0
; CHECK-REG-NOT: stxrh w[[NEW]], w[[NEW]], [x{{[0-9]+}}]
; CHECK-NEXT: stxrh [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i16 %old
}

define i32 @test_atomic_load_or_i32(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_or_i32:
   %old = atomicrmw or i32* @var32, i32 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], #:lo12:var32

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK-NEXT: ldaxr w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: orr [[NEW:w[0-9]+]], w[[OLD]], w0
; CHECK-REG: orr w[[NEW:[0-9]+]], w{{[0-9]+}}, w0
; CHECK-REG-NOT: stxr w[[NEW]], w[[NEW]], [x{{[0-9]+}}]
; CHECK-NEXT: stxr [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i32 %old
}

define i64 @test_atomic_load_or_i64(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_or_i64:
   %old = atomicrmw or i64* @var64, i64 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], #:lo12:var64

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK-NEXT: ldxr x[[OLD:[0-9]+]], [x[[ADDR]]]
  ; x0 below is a reasonable guess but could change: it certainly comes into the
  ; function there.
; CHECK-NEXT: orr [[NEW:x[0-9]+]], x[[OLD]], x0
; CHECK-REG: orr x[[NEW:[0-9]+]], x{{[0-9]+}}, x0
; CHECK-REG-NOT: stlxr w[[NEW]], x[[NEW]], [x{{[0-9]+}}]
; CHECK-NEXT: stlxr [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i64 %old
}

define i8 @test_atomic_load_xor_i8(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xor_i8:
   %old = atomicrmw xor i8* @var8, i8 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], #:lo12:var8

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK-NEXT: ldaxrb w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: eor [[NEW:w[0-9]+]], w[[OLD]], w0
; CHECK-REG: eor w[[NEW:[0-9]+]], w{{[0-9]+}}, w0
; CHECK-REG-NOT: stxrb w[[NEW]], w[[NEW]], [x{{[0-9]+}}]
; CHECK-NEXT: stxrb [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i8 %old
}

define i16 @test_atomic_load_xor_i16(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xor_i16:
   %old = atomicrmw xor i16* @var16, i16 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], #:lo12:var16

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK-NEXT: ldxrh w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: eor [[NEW:w[0-9]+]], w[[OLD]], w0
; CHECK-REG: eor w[[NEW:[0-9]+]], w{{[0-9]+}}, w0
; CHECK-REG-NOT: stxrh w[[NEW]], w[[NEW]], [x{{[0-9]+}}]
; CHECK-NEXT: stlxrh [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i16 %old
}

define i32 @test_atomic_load_xor_i32(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xor_i32:
   %old = atomicrmw xor i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], #:lo12:var32

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK-NEXT: ldaxr w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: eor [[NEW:w[0-9]+]], w[[OLD]], w0
; CHECK-REG: eor w[[NEW:[0-9]+]], w{{[0-9]+}}, w0
; CHECK-REG-NOT: stlxr w[[NEW]], w[[NEW]], [x{{[0-9]+}}]
; CHECK-NEXT: stlxr [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i32 %old
}

define i64 @test_atomic_load_xor_i64(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xor_i64:
   %old = atomicrmw xor i64* @var64, i64 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], #:lo12:var64

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK-NEXT: ldxr x[[OLD:[0-9]+]], [x[[ADDR]]]
  ; x0 below is a reasonable guess but could change: it certainly comes into the
  ; function there.
; CHECK-NEXT: eor [[NEW:x[0-9]+]], x[[OLD]], x0
; CHECK-REG: eor x[[NEW:[0-9]+]], x{{[0-9]+}}, x0
; CHECK-REG-NOT: stxr w[[NEW]], x[[NEW]], [x{{[0-9]+}}]
; CHECK-NEXT: stxr [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i64 %old
}

define i8 @test_atomic_load_xchg_i8(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xchg_i8:
   %old = atomicrmw xchg i8* @var8, i8 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], #:lo12:var8

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK-NEXT: ldxrb w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ; function there.
; CHECK-REG-NOT: stxrb w0, w0, [x{{[0-9]+}}]
; CHECK-NEXT: stxrb [[STATUS:w[0-9]+]], w0, [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i8 %old
}

define i16 @test_atomic_load_xchg_i16(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xchg_i16:
   %old = atomicrmw xchg i16* @var16, i16 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], #:lo12:var16

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK-NEXT: ldaxrh w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ; function there.
; CHECK-REG-NOT: stlxrh w0, w0, [x{{[0-9]+}}]
; CHECK-NEXT: stlxrh [[STATUS:w[0-9]+]], w0, [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i16 %old
}

define i32 @test_atomic_load_xchg_i32(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xchg_i32:
   %old = atomicrmw xchg i32* @var32, i32 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], #:lo12:var32

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK-NEXT: ldxr w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-REG-NOT: stlxr w0, w0, [x{{[0-9]+}}]
; CHECK-NEXT: stlxr [[STATUS:w[0-9]+]], w0, [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i32 %old
}

define i64 @test_atomic_load_xchg_i64(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xchg_i64:
   %old = atomicrmw xchg i64* @var64, i64 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], #:lo12:var64

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK-NEXT: ldaxr x[[OLD:[0-9]+]], [x[[ADDR]]]
  ; x0 below is a reasonable guess but could change: it certainly comes into the
  ; function there.
; CHECK-REG-NOT: stxr w0, x0, [x{{[0-9]+}}]
; CHECK-NEXT: stxr [[STATUS:w[0-9]+]], x0, [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i64 %old
}


define i8 @test_atomic_load_min_i8(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_min_i8:
   %old = atomicrmw min i8* @var8, i8 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], #:lo12:var8

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK-NEXT: ldaxrb w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: cmp w0, w[[OLD]], sxtb
; CHECK-NEXT: csel [[NEW:w[0-9]+]], w[[OLD]], w0, gt
; CHECK-REG: csel w[[NEW:[0-9]+]], w{{[0-9]+}}, w0, gt
; CHECK-REG-NOT: stxrb w[[NEW]], w[[NEW]], [x{{[0-9]+}}]
; CHECK-NEXT: stxrb [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i8 %old
}

define i16 @test_atomic_load_min_i16(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_min_i16:
   %old = atomicrmw min i16* @var16, i16 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], #:lo12:var16

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK-NEXT: ldxrh w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: cmp w0, w[[OLD]], sxth
; CHECK-NEXT: csel [[NEW:w[0-9]+]], w[[OLD]], w0, gt
; CHECK-REG: csel w[[NEW:[0-9]+]], w{{[0-9]+}}, w0, gt
; CHECK-REG-NOT: stlxrh w[[NEW]], w[[NEW]], [x{{[0-9]+}}]
; CHECK-NEXT: stlxrh [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i16 %old
}

define i32 @test_atomic_load_min_i32(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_min_i32:
   %old = atomicrmw min i32* @var32, i32 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], #:lo12:var32

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK-NEXT: ldxr w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: cmp w0, w[[OLD]]
; CHECK-NEXT: csel [[NEW:w[0-9]+]], w[[OLD]], w0, gt
; CHECK-REG: csel w[[NEW:[0-9]+]], w{{[0-9]+}}, w0, gt
; CHECK-REG-NOT: stxr w[[NEW]], w[[NEW]], [x{{[0-9]+}}]
; CHECK-NEXT: stxr [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i32 %old
}

define i64 @test_atomic_load_min_i64(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_min_i64:
   %old = atomicrmw min i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], #:lo12:var64

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK-NEXT: ldaxr x[[OLD:[0-9]+]], [x[[ADDR]]]
  ; x0 below is a reasonable guess but could change: it certainly comes into the
  ; function there.
; CHECK-NEXT: cmp x0, x[[OLD]]
; CHECK-NEXT: csel [[NEW:x[0-9]+]], x[[OLD]], x0, gt
; CHECK-REG: csel x[[NEW:[0-9]+]], x{{[0-9]+}}, x0, gt
; CHECK-REG-NOT: stlxr w[[NEW]], x[[NEW]], [x{{[0-9]+}}]
; CHECK-NEXT: stlxr [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i64 %old
}

define i8 @test_atomic_load_max_i8(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_max_i8:
   %old = atomicrmw max i8* @var8, i8 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], #:lo12:var8

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK-NEXT: ldaxrb w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: cmp w0, w[[OLD]], sxtb
; CHECK-NEXT: csel [[NEW:w[0-9]+]], w[[OLD]], w0, lt
; CHECK-REG: csel w[[NEW:[0-9]+]], w{{[0-9]+}}, w0, lt
; CHECK-REG-NOT: stlxrb w[[NEW]], w[[NEW]], [x{{[0-9]+}}]
; CHECK-NEXT: stlxrb [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i8 %old
}

define i16 @test_atomic_load_max_i16(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_max_i16:
   %old = atomicrmw max i16* @var16, i16 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], #:lo12:var16

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK-NEXT: ldaxrh w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: cmp w0, w[[OLD]], sxth
; CHECK-NEXT: csel [[NEW:w[0-9]+]], w[[OLD]], w0, lt
; CHECK-REG: csel w[[NEW:[0-9]+]], w{{[0-9]+}}, w0, lt
; CHECK-REG-NOT: stxrh w[[NEW]], w[[NEW]], [x{{[0-9]+}}]
; CHECK-NEXT: stxrh [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i16 %old
}

define i32 @test_atomic_load_max_i32(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_max_i32:
   %old = atomicrmw max i32* @var32, i32 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], #:lo12:var32

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK-NEXT: ldxr w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: cmp w0, w[[OLD]]
; CHECK-NEXT: csel [[NEW:w[0-9]+]], w[[OLD]], w0, lt
; CHECK-REG: csel w[[NEW:[0-9]+]], w{{[0-9]+}}, w0, lt
; CHECK-REG-NOT: stlxr w[[NEW]], w[[NEW]], [x{{[0-9]+}}]
; CHECK-NEXT: stlxr [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i32 %old
}

define i64 @test_atomic_load_max_i64(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_max_i64:
   %old = atomicrmw max i64* @var64, i64 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], #:lo12:var64

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK-NEXT: ldxr x[[OLD:[0-9]+]], [x[[ADDR]]]
  ; x0 below is a reasonable guess but could change: it certainly comes into the
  ; function there.
; CHECK-NEXT: cmp x0, x[[OLD]]
; CHECK-NEXT: csel [[NEW:x[0-9]+]], x[[OLD]], x0, lt
; CHECK-REG: csel x[[NEW:[0-9]+]], x{{[0-9]+}}, x0, lt
; CHECK-REG-NOT: stlxr w[[NEW]], x[[NEW]], [x{{[0-9]+}}]
; CHECK-NEXT: stxr [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i64 %old
}

define i8 @test_atomic_load_umin_i8(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umin_i8:
   %old = atomicrmw umin i8* @var8, i8 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], #:lo12:var8

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK-NEXT: ldxrb w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: cmp w0, w[[OLD]], uxtb
; CHECK-NEXT: csel [[NEW:w[0-9]+]], w[[OLD]], w0, hi
; CHECK-REG: csel w[[NEW:[0-9]+]], w{{[0-9]+}}, w0, hi
; CHECK-REG-NOT: stlxr w[[NEW]], w[[NEW]], [x{{[0-9]+}}]
; CHECK-NEXT: stxrb [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i8 %old
}

define i16 @test_atomic_load_umin_i16(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umin_i16:
   %old = atomicrmw umin i16* @var16, i16 %offset acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], #:lo12:var16

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK-NEXT: ldaxrh w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: cmp w0, w[[OLD]], uxth
; CHECK-NEXT: csel [[NEW:w[0-9]+]], w[[OLD]], w0, hi
; CHECK-REG: csel w[[NEW:[0-9]+]], w{{[0-9]+}}, w0, hi
; CHECK-REG-NOT: stxrh w[[NEW]], w[[NEW]], [x{{[0-9]+}}]
; CHECK-NEXT: stxrh [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i16 %old
}

define i32 @test_atomic_load_umin_i32(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umin_i32:
   %old = atomicrmw umin i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], #:lo12:var32

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK-NEXT: ldaxr w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: cmp w0, w[[OLD]]
; CHECK-NEXT: csel [[NEW:w[0-9]+]], w[[OLD]], w0, hi
; CHECK-REG: csel w[[NEW:[0-9]+]], w{{[0-9]+}}, w0, hi
; CHECK-REG-NOT: stlxr w[[NEW]], w[[NEW]], [x{{[0-9]+}}]
; CHECK-NEXT: stlxr [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i32 %old
}

define i64 @test_atomic_load_umin_i64(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umin_i64:
   %old = atomicrmw umin i64* @var64, i64 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], #:lo12:var64

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK-NEXT: ldaxr x[[OLD:[0-9]+]], [x[[ADDR]]]
  ; x0 below is a reasonable guess but could change: it certainly comes into the
  ; function there.
; CHECK-NEXT: cmp x0, x[[OLD]]
; CHECK-NEXT: csel [[NEW:x[0-9]+]], x[[OLD]], x0, hi
; CHECK-REG: csel x[[NEW:[0-9]+]], x{{[0-9]+}}, x0, hi
; CHECK-REG-NOT: stlxr w[[NEW]], x[[NEW]], [x{{[0-9]+}}]
; CHECK-NEXT: stlxr [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i64 %old
}

define i8 @test_atomic_load_umax_i8(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umax_i8:
   %old = atomicrmw umax i8* @var8, i8 %offset acq_rel
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], #:lo12:var8

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK-NEXT: ldaxrb w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: cmp w0, w[[OLD]], uxtb
; CHECK-NEXT: csel [[NEW:w[0-9]+]], w[[OLD]], w0, lo
; CHECK-REG: csel w[[NEW:[0-9]+]], w{{[0-9]+}}, w0, lo
; CHECK-REG-NOT: stlxrb w[[NEW]], w[[NEW]], [x{{[0-9]+}}]
; CHECK-NEXT: stlxrb [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i8 %old
}

define i16 @test_atomic_load_umax_i16(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umax_i16:
   %old = atomicrmw umax i16* @var16, i16 %offset monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], #:lo12:var16

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK-NEXT: ldxrh w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: cmp w0, w[[OLD]], uxth
; CHECK-NEXT: csel [[NEW:w[0-9]+]], w[[OLD]], w0, lo
; CHECK-REG: csel w[[NEW:[0-9]+]], w{{[0-9]+}}, w0, lo
; CHECK-REG-NOT: stxrh w[[NEW]], w[[NEW]], [x{{[0-9]+}}]
; CHECK-NEXT: stxrh [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i16 %old
}

define i32 @test_atomic_load_umax_i32(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umax_i32:
   %old = atomicrmw umax i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], #:lo12:var32

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK-NEXT: ldaxr w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: cmp w0, w[[OLD]]
; CHECK-NEXT: csel [[NEW:w[0-9]+]], w[[OLD]], w0, lo
; CHECK-REG: csel w[[NEW:[0-9]+]], w{{[0-9]+}}, w0, lo
; CHECK-REG-NOT: stlxr w[[NEW]], w[[NEW]], [x{{[0-9]+}}]
; CHECK-NEXT: stlxr [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i32 %old
}

define i64 @test_atomic_load_umax_i64(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umax_i64:
   %old = atomicrmw umax i64* @var64, i64 %offset release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], #:lo12:var64

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK-NEXT: ldxr x[[OLD:[0-9]+]], [x[[ADDR]]]
  ; x0 below is a reasonable guess but could change: it certainly comes into the
  ; function there.
; CHECK-NEXT: cmp x0, x[[OLD]]
; CHECK-NEXT: csel [[NEW:x[0-9]+]], x[[OLD]], x0, lo
; CHECK-REG: csel x[[NEW:[0-9]+]], x{{[0-9]+}}, x0, lo
; CHECK-REG-NOT: stlxr w[[NEW]], x[[NEW]], [x{{[0-9]+}}]
; CHECK-NEXT: stlxr [[STATUS:w[0-9]+]], [[NEW]], [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i64 %old
}

define i8 @test_atomic_cmpxchg_i8(i8 %wanted, i8 %new) nounwind {
; CHECK-LABEL: test_atomic_cmpxchg_i8:
   %old = cmpxchg i8* @var8, i8 %wanted, i8 %new acquire
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], #:lo12:var8

; CHECK: [[STARTAGAIN:.LBB[0-9]+_[0-9]+]]:
; CHECK-NEXT: ldaxrb w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: cmp w[[OLD]], w0
; CHECK-NEXT: b.ne [[GET_OUT:.LBB[0-9]+_[0-9]+]]
  ; As above, w1 is a reasonable guess.
; CHECK-REG-NOT: stxrb w1, w1, [x{{[0-9]+}}]
; CHECK: stxrb [[STATUS:w[0-9]+]], w1, [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], [[STARTAGAIN]]
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i8 %old
}

define i16 @test_atomic_cmpxchg_i16(i16 %wanted, i16 %new) nounwind {
; CHECK-LABEL: test_atomic_cmpxchg_i16:
   %old = cmpxchg i16* @var16, i16 %wanted, i16 %new seq_cst
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var16
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], #:lo12:var16

; CHECK: [[STARTAGAIN:.LBB[0-9]+_[0-9]+]]:
; CHECK-NEXT: ldaxrh w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: cmp w[[OLD]], w0
; CHECK-NEXT: b.ne [[GET_OUT:.LBB[0-9]+_[0-9]+]]
  ; As above, w1 is a reasonable guess.
; CHECK-REG-NOT: stlxrh w1, w1, [x{{[0-9]+}}]
; CHECK: stlxrh [[STATUS:w[0-9]+]], w1, [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], [[STARTAGAIN]]
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i16 %old
}

define i32 @test_atomic_cmpxchg_i32(i32 %wanted, i32 %new) nounwind {
; CHECK-LABEL: test_atomic_cmpxchg_i32:
   %old = cmpxchg i32* @var32, i32 %wanted, i32 %new release
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var32
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], #:lo12:var32

; CHECK: [[STARTAGAIN:.LBB[0-9]+_[0-9]+]]:
; CHECK-NEXT: ldxr w[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: cmp w[[OLD]], w0
; CHECK-NEXT: b.ne [[GET_OUT:.LBB[0-9]+_[0-9]+]]
  ; As above, w1 is a reasonable guess.
; CHECK-REG-NOT: stlxr w1, w1, [x{{[0-9]+}}]
; CHECK: stlxr [[STATUS:w[0-9]+]], w1, [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], [[STARTAGAIN]]
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i32 %old
}

define i64 @test_atomic_cmpxchg_i64(i64 %wanted, i64 %new) nounwind {
; CHECK-LABEL: test_atomic_cmpxchg_i64:
   %old = cmpxchg i64* @var64, i64 %wanted, i64 %new monotonic
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var64
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], #:lo12:var64

; CHECK: [[STARTAGAIN:.LBB[0-9]+_[0-9]+]]:
; CHECK-NEXT: ldxr x[[OLD:[0-9]+]], [x[[ADDR]]]
  ; w0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: cmp x[[OLD]], x0
; CHECK-NEXT: b.ne [[GET_OUT:.LBB[0-9]+_[0-9]+]]
  ; As above, w1 is a reasonable guess.
; CHECK-REG-NOT: stxr w1, x1, [x{{[0-9]+}}]
; CHECK: stxr [[STATUS:w[0-9]+]], x1, [x[[ADDR]]]
; CHECK-NEXT: cbnz [[STATUS]], [[STARTAGAIN]]
; CHECK-NOT: dmb

; CHECK: mov x0, x[[OLD]]
   ret i64 %old
}

define i8 @test_atomic_load_monotonic_i8() nounwind {
; CHECK-LABEL: test_atomic_load_monotonic_i8:
  %val = load atomic i8* @var8 monotonic, align 1
; CHECK-NOT: dmb
; CHECK: adrp x[[HIADDR:[0-9]+]], var8
; CHECK: ldrb w0, [x[[HIADDR]], #:lo12:var8]
; CHECK-NOT: dmb

  ret i8 %val
}

define i8 @test_atomic_load_monotonic_regoff_i8(i64 %base, i64 %off) nounwind {
; CHECK-LABEL: test_atomic_load_monotonic_regoff_i8:
  %addr_int = add i64 %base, %off
  %addr = inttoptr i64 %addr_int to i8*

  %val = load atomic i8* %addr monotonic, align 1
; CHECK-NOT: dmb
; CHECK: ldrb w0, [x0, x1]
; CHECK-NOT: dmb

  ret i8 %val
}

define i8 @test_atomic_load_acquire_i8() nounwind {
; CHECK-LABEL: test_atomic_load_acquire_i8:
  %val = load atomic i8* @var8 acquire, align 1
; CHECK-NOT: dmb
; CHECK: adrp [[TMPADDR:x[0-9]+]], var8
; CHECK-NOT: dmb
; CHECK: add x[[ADDR:[0-9]+]], [[TMPADDR]], #:lo12:var8
; CHECK-NOT: dmb
; CHECK: ldarb w0, [x[[ADDR]]]
; CHECK-NOT: dmb
  ret i8 %val
}

define i8 @test_atomic_load_seq_cst_i8() nounwind {
; CHECK-LABEL: test_atomic_load_seq_cst_i8:
  %val = load atomic i8* @var8 seq_cst, align 1
; CHECK-NOT: dmb
; CHECK: adrp [[HIADDR:x[0-9]+]], var8
; CHECK-NOT: dmb
; CHECK: add x[[ADDR:[0-9]+]], [[HIADDR]], #:lo12:var8
; CHECK-NOT: dmb
; CHECK: ldarb w0, [x[[ADDR]]]
; CHECK-NOT: dmb
  ret i8 %val
}

define i16 @test_atomic_load_monotonic_i16() nounwind {
; CHECK-LABEL: test_atomic_load_monotonic_i16:
  %val = load atomic i16* @var16 monotonic, align 2
; CHECK-NOT: dmb
; CHECK: adrp x[[HIADDR:[0-9]+]], var16
; CHECK-NOT: dmb
; CHECK: ldrh w0, [x[[HIADDR]], #:lo12:var16]
; CHECK-NOT: dmb

  ret i16 %val
}

define i32 @test_atomic_load_monotonic_regoff_i32(i64 %base, i64 %off) nounwind {
; CHECK-LABEL: test_atomic_load_monotonic_regoff_i32:
  %addr_int = add i64 %base, %off
  %addr = inttoptr i64 %addr_int to i32*

  %val = load atomic i32* %addr monotonic, align 4
; CHECK-NOT: dmb
; CHECK: ldr w0, [x0, x1]
; CHECK-NOT: dmb

  ret i32 %val
}

define i64 @test_atomic_load_seq_cst_i64() nounwind {
; CHECK-LABEL: test_atomic_load_seq_cst_i64:
  %val = load atomic i64* @var64 seq_cst, align 8
; CHECK-NOT: dmb
; CHECK: adrp [[HIADDR:x[0-9]+]], var64
; CHECK-NOT: dmb
; CHECK: add x[[ADDR:[0-9]+]], [[HIADDR]], #:lo12:var64
; CHECK-NOT: dmb
; CHECK: ldar x0, [x[[ADDR]]]
; CHECK-NOT: dmb
  ret i64 %val
}

define void @test_atomic_store_monotonic_i8(i8 %val) nounwind {
; CHECK-LABEL: test_atomic_store_monotonic_i8:
  store atomic i8 %val, i8* @var8 monotonic, align 1
; CHECK: adrp x[[HIADDR:[0-9]+]], var8
; CHECK: strb w0, [x[[HIADDR]], #:lo12:var8]

  ret void
}

define void @test_atomic_store_monotonic_regoff_i8(i64 %base, i64 %off, i8 %val) nounwind {
; CHECK-LABEL: test_atomic_store_monotonic_regoff_i8:

  %addr_int = add i64 %base, %off
  %addr = inttoptr i64 %addr_int to i8*

  store atomic i8 %val, i8* %addr monotonic, align 1
; CHECK: strb w2, [x0, x1]

  ret void
}
define void @test_atomic_store_release_i8(i8 %val) nounwind {
; CHECK-LABEL: test_atomic_store_release_i8:
  store atomic i8 %val, i8* @var8 release, align 1
; CHECK-NOT: dmb
; CHECK: adrp [[HIADDR:x[0-9]+]], var8
; CHECK-NOT: dmb
; CHECK: add x[[ADDR:[0-9]+]], [[HIADDR]], #:lo12:var8
; CHECK-NOT: dmb
; CHECK: stlrb w0, [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_store_seq_cst_i8(i8 %val) nounwind {
; CHECK-LABEL: test_atomic_store_seq_cst_i8:
  store atomic i8 %val, i8* @var8 seq_cst, align 1
; CHECK-NOT: dmb
; CHECK: adrp [[HIADDR:x[0-9]+]], var8
; CHECK-NOT: dmb
; CHECK: add x[[ADDR:[0-9]+]], [[HIADDR]], #:lo12:var8
; CHECK-NOT: dmb
; CHECK: stlrb w0, [x[[ADDR]]]
; CHECK-NOT: dmb

  ret void
}

define void @test_atomic_store_monotonic_i16(i16 %val) nounwind {
; CHECK-LABEL: test_atomic_store_monotonic_i16:
  store atomic i16 %val, i16* @var16 monotonic, align 2
; CHECK-NOT: dmb
; CHECK: adrp x[[HIADDR:[0-9]+]], var16
; CHECK-NOT: dmb
; CHECK: strh w0, [x[[HIADDR]], #:lo12:var16]
; CHECK-NOT: dmb
  ret void
}

define void @test_atomic_store_monotonic_regoff_i32(i64 %base, i64 %off, i32 %val) nounwind {
; CHECK-LABEL: test_atomic_store_monotonic_regoff_i32:

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
  store atomic i64 %val, i64* @var64 release, align 8
; CHECK-NOT: dmb
; CHECK: adrp [[HIADDR:x[0-9]+]], var64
; CHECK-NOT: dmb
; CHECK: add x[[ADDR:[0-9]+]], [[HIADDR]], #:lo12:var64
; CHECK-NOT: dmb
; CHECK: stlr x0, [x[[ADDR]]]
; CHECK-NOT: dmb
  ret void
}
