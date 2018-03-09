; RUN: llc -mtriple=armv8-none-linux-gnu -verify-machineinstrs < %s | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-LE --check-prefix=CHECK-ARM --check-prefix=CHECK-ARM-LE
; RUN: llc -mtriple=armebv8-none-linux-gnu -verify-machineinstrs < %s | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-BE --check-prefix=CHECK-ARM --check-prefix=CHECK-ARM-BE
; RUN: llc -mtriple=thumbv8-none-linux-gnu -verify-machineinstrs < %s | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-LE --check-prefix=CHECK-THUMB --check-prefix=CHECK-THUMB-LE
; RUN: llc -mtriple=thumbebv8-none-linux-gnu -verify-machineinstrs < %s | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-BE --check-prefix=CHECK-THUMB --check-prefix=CHECK-THUMB-BE

@var8 = global i8 0
@var16 = global i16 0
@var32 = global i32 0
@var64 = global i64 0

define i8 @test_atomic_load_add_i8(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i8:
   %old = atomicrmw add i8* @var8, i8 %offset seq_cst
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw r[[ADDR:[0-9]+]], :lower16:var8
; CHECK: movt r[[ADDR]], :upper16:var8

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldaexb r[[OLD:[0-9]+]], [r[[ADDR]]]
  ; r0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: add{{s?}} [[NEW:r[0-9]+]], r[[OLD]], r0
; CHECK-NEXT: stlexb [[STATUS:r[0-9]+]], [[NEW]], [r[[ADDR]]]
; CHECK-NEXT: cmp [[STATUS]], #0
; CHECK-NEXT: bne .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb
; CHECK-NOT: mcr

; CHECK: mov r0, r[[OLD]]
   ret i8 %old
}

define i16 @test_atomic_load_add_i16(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i16:
   %old = atomicrmw add i16* @var16, i16 %offset acquire
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw r[[ADDR:[0-9]+]], :lower16:var16
; CHECK: movt r[[ADDR]], :upper16:var16

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldaexh r[[OLD:[0-9]+]], [r[[ADDR]]]
  ; r0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: add{{s?}} [[NEW:r[0-9]+]], r[[OLD]], r0
; CHECK-NEXT: strexh [[STATUS:r[0-9]+]], [[NEW]], [r[[ADDR]]]
; CHECK-NEXT: cmp [[STATUS]], #0
; CHECK-NEXT: bne .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb
; CHECK-NOT: mcr

; CHECK: mov r0, r[[OLD]]
   ret i16 %old
}

define i32 @test_atomic_load_add_i32(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i32:
   %old = atomicrmw add i32* @var32, i32 %offset release
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw r[[ADDR:[0-9]+]], :lower16:var32
; CHECK: movt r[[ADDR]], :upper16:var32

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldrex r[[OLD:[0-9]+]], [r[[ADDR]]]
  ; r0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: add{{s?}} [[NEW:r[0-9]+]], r[[OLD]], r0
; CHECK-NEXT: stlex [[STATUS:r[0-9]+]], [[NEW]], [r[[ADDR]]]
; CHECK-NEXT: cmp [[STATUS]], #0
; CHECK-NEXT: bne .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb
; CHECK-NOT: mcr

; CHECK: mov r0, r[[OLD]]
   ret i32 %old
}

define void @test_atomic_load_add_i64(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i64:
   %old = atomicrmw add i64* @var64, i64 %offset monotonic
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw r[[ADDR:[0-9]+]], :lower16:var64
; CHECK: movt r[[ADDR]], :upper16:var64

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldrexd r[[OLD1:[0-9]+]], r[[OLD2:[0-9]+]], [r[[ADDR]]]
  ; r0, r1 below is a reasonable guess but could change: it certainly comes into the
  ; function there.
; CHECK-LE-NEXT: adds{{(\.w)?}} [[NEW1:r[0-9]+|lr]], r[[OLD1]], r0
; CHECK-LE-NEXT: adc{{(\.w)?}}  [[NEW2:r[0-9]+]], r[[OLD2]], r1
; CHECK-BE-NEXT: adds{{(\.w)?}} [[NEW2:r[0-9]+|lr]], r[[OLD2]], r1
; CHECK-BE-NEXT: adc{{(\.w)?}}  [[NEW1:r[0-9]+]], r[[OLD1]], r0
; CHECK-NEXT: strexd [[STATUS:r[0-9]+]], [[NEW1]], [[NEW2]], [r[[ADDR]]]
; CHECK-NEXT: cmp [[STATUS]], #0
; CHECK-NEXT: bne .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb
; CHECK-NOT: mcr

; CHECK: strd r[[OLD1]], r[[OLD2]], [r[[ADDR]]]
  store i64 %old, i64* @var64
   ret void
}

define i8 @test_atomic_load_sub_i8(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i8:
   %old = atomicrmw sub i8* @var8, i8 %offset monotonic
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw r[[ADDR:[0-9]+]], :lower16:var8
; CHECK: movt r[[ADDR]], :upper16:var8

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldrexb r[[OLD:[0-9]+]], [r[[ADDR]]]
  ; r0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: sub{{s?}} [[NEW:r[0-9]+]], r[[OLD]], r0
; CHECK-NEXT: strexb [[STATUS:r[0-9]+]], [[NEW]], [r[[ADDR]]]
; CHECK-NEXT: cmp [[STATUS]], #0
; CHECK-NEXT: bne .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb
; CHECK-NOT: mcr

; CHECK: mov r0, r[[OLD]]
   ret i8 %old
}

define i16 @test_atomic_load_sub_i16(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i16:
   %old = atomicrmw sub i16* @var16, i16 %offset release
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw r[[ADDR:[0-9]+]], :lower16:var16
; CHECK: movt r[[ADDR]], :upper16:var16

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldrexh r[[OLD:[0-9]+]], [r[[ADDR]]]
  ; r0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: sub{{s?}} [[NEW:r[0-9]+]], r[[OLD]], r0
; CHECK-NEXT: stlexh [[STATUS:r[0-9]+]], [[NEW]], [r[[ADDR]]]
; CHECK-NEXT: cmp [[STATUS]], #0
; CHECK-NEXT: bne .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb
; CHECK-NOT: mcr

; CHECK: mov r0, r[[OLD]]
   ret i16 %old
}

define i32 @test_atomic_load_sub_i32(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i32:
   %old = atomicrmw sub i32* @var32, i32 %offset acquire
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw r[[ADDR:[0-9]+]], :lower16:var32
; CHECK: movt r[[ADDR]], :upper16:var32

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldaex r[[OLD:[0-9]+]], [r[[ADDR]]]
  ; r0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: sub{{s?}} [[NEW:r[0-9]+]], r[[OLD]], r0
; CHECK-NEXT: strex [[STATUS:r[0-9]+]], [[NEW]], [r[[ADDR]]]
; CHECK-NEXT: cmp [[STATUS]], #0
; CHECK-NEXT: bne .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb
; CHECK-NOT: mcr

; CHECK: mov r0, r[[OLD]]
   ret i32 %old
}

define void @test_atomic_load_sub_i64(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_sub_i64:
   %old = atomicrmw sub i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw r[[ADDR:[0-9]+]], :lower16:var64
; CHECK: movt r[[ADDR]], :upper16:var64

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldaexd r[[OLD1:[0-9]+]], r[[OLD2:[0-9]+]], [r[[ADDR]]]
  ; r0, r1 below is a reasonable guess but could change: it certainly comes into the
  ; function there.
; CHECK-LE-NEXT: subs{{(\.w)?}} [[NEW1:r[0-9]+|lr]], r[[OLD1]], r0
; CHECK-LE-NEXT: sbc{{(\.w)?}}  [[NEW2:r[0-9]+]], r[[OLD2]], r1
; CHECK-BE-NEXT: subs{{(\.w)?}} [[NEW2:r[0-9]+|lr]], r[[OLD2]], r1
; CHECK-BE-NEXT: sbc{{(\.w)?}}  [[NEW1:r[0-9]+]], r[[OLD1]], r0
; CHECK-NEXT: stlexd [[STATUS:r[0-9]+]], [[NEW1]], [[NEW2]], [r[[ADDR]]]
; CHECK-NEXT: cmp [[STATUS]], #0
; CHECK-NEXT: bne .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb
; CHECK-NOT: mcr

; CHECK: strd r[[OLD1]], r[[OLD2]], [r[[ADDR]]]
   store i64 %old, i64* @var64
   ret void
}

define i8 @test_atomic_load_and_i8(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i8:
   %old = atomicrmw and i8* @var8, i8 %offset release
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw r[[ADDR:[0-9]+]], :lower16:var8
; CHECK: movt r[[ADDR]], :upper16:var8

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldrexb r[[OLD:[0-9]+]], [r[[ADDR]]]
  ; r0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: and{{(\.w)?}} [[NEW:r[0-9]+]], r[[OLD]], r0
; CHECK-NEXT: stlexb [[STATUS:r[0-9]+]], [[NEW]], [r[[ADDR]]]
; CHECK-NEXT: cmp [[STATUS]], #0
; CHECK-NEXT: bne .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb
; CHECK-NOT: mcr

; CHECK: mov r0, r[[OLD]]
   ret i8 %old
}

define i16 @test_atomic_load_and_i16(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i16:
   %old = atomicrmw and i16* @var16, i16 %offset monotonic
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw r[[ADDR:[0-9]+]], :lower16:var16
; CHECK: movt r[[ADDR]], :upper16:var16

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldrexh r[[OLD:[0-9]+]], [r[[ADDR]]]
  ; r0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: and{{(\.w)?}} [[NEW:r[0-9]+]], r[[OLD]], r0
; CHECK-NEXT: strexh [[STATUS:r[0-9]+]], [[NEW]], [r[[ADDR]]]
; CHECK-NEXT: cmp [[STATUS]], #0
; CHECK-NEXT: bne .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb
; CHECK-NOT: mcr

; CHECK: mov r0, r[[OLD]]
   ret i16 %old
}

define i32 @test_atomic_load_and_i32(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i32:
   %old = atomicrmw and i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw r[[ADDR:[0-9]+]], :lower16:var32
; CHECK: movt r[[ADDR]], :upper16:var32

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldaex r[[OLD:[0-9]+]], [r[[ADDR]]]
  ; r0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: and{{(\.w)?}} [[NEW:r[0-9]+]], r[[OLD]], r0
; CHECK-NEXT: stlex [[STATUS:r[0-9]+]], [[NEW]], [r[[ADDR]]]
; CHECK-NEXT: cmp [[STATUS]], #0
; CHECK-NEXT: bne .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb
; CHECK-NOT: mcr

; CHECK: mov r0, r[[OLD]]
   ret i32 %old
}

define void @test_atomic_load_and_i64(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_and_i64:
   %old = atomicrmw and i64* @var64, i64 %offset acquire
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw r[[ADDR:[0-9]+]], :lower16:var64
; CHECK: movt r[[ADDR]], :upper16:var64

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldaexd r[[OLD1:[0-9]+]], r[[OLD2:[0-9]+]], [r[[ADDR]]]
  ; r0, r1 below is a reasonable guess but could change: it certainly comes into the
  ; function there.
; CHECK-LE-DAG: and{{(\.w)?}} [[NEW1:r[0-9]+|lr]], r[[OLD1]], r0
; CHECK-LE-DAG: and{{(\.w)?}} [[NEW2:r[0-9]+|lr]], r[[OLD2]], r1
; CHECK-BE-DAG: and{{(\.w)?}} [[NEW2:r[0-9]+|lr]], r[[OLD2]], r1
; CHECK-BE-DAG: and{{(\.w)?}} [[NEW1:r[0-9]+|lr]], r[[OLD1]], r0
; CHECK: strexd [[STATUS:r[0-9]+]], [[NEW1]], [[NEW2]], [r[[ADDR]]]
; CHECK-NEXT: cmp [[STATUS]], #0
; CHECK-NEXT: bne .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb
; CHECK-NOT: mcr

; CHECK: strd r[[OLD1]], r[[OLD2]], [r[[ADDR]]]
   store i64 %old, i64* @var64
   ret void
}

define i8 @test_atomic_load_or_i8(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_or_i8:
   %old = atomicrmw or i8* @var8, i8 %offset seq_cst
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw r[[ADDR:[0-9]+]], :lower16:var8
; CHECK: movt r[[ADDR]], :upper16:var8

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldaexb r[[OLD:[0-9]+]], [r[[ADDR]]]
  ; r0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: orr{{(\.w)?}} [[NEW:r[0-9]+]], r[[OLD]], r0
; CHECK-NEXT: stlexb [[STATUS:r[0-9]+]], [[NEW]], [r[[ADDR]]]
; CHECK-NEXT: cmp [[STATUS]], #0
; CHECK-NEXT: bne .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb
; CHECK-NOT: mcr

; CHECK: mov r0, r[[OLD]]
   ret i8 %old
}

define i16 @test_atomic_load_or_i16(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_or_i16:
   %old = atomicrmw or i16* @var16, i16 %offset monotonic
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw r[[ADDR:[0-9]+]], :lower16:var16
; CHECK: movt r[[ADDR]], :upper16:var16

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldrexh r[[OLD:[0-9]+]], [r[[ADDR]]]
  ; r0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: orr{{(\.w)?}} [[NEW:r[0-9]+]], r[[OLD]], r0
; CHECK-NEXT: strexh [[STATUS:r[0-9]+]], [[NEW]], [r[[ADDR]]]
; CHECK-NEXT: cmp [[STATUS]], #0
; CHECK-NEXT: bne .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb
; CHECK-NOT: mcr

; CHECK: mov r0, r[[OLD]]
   ret i16 %old
}

define i32 @test_atomic_load_or_i32(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_or_i32:
   %old = atomicrmw or i32* @var32, i32 %offset acquire
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw r[[ADDR:[0-9]+]], :lower16:var32
; CHECK: movt r[[ADDR]], :upper16:var32

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldaex r[[OLD:[0-9]+]], [r[[ADDR]]]
  ; r0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: orr{{(\.w)?}} [[NEW:r[0-9]+]], r[[OLD]], r0
; CHECK-NEXT: strex [[STATUS:r[0-9]+]], [[NEW]], [r[[ADDR]]]
; CHECK-NEXT: cmp [[STATUS]], #0
; CHECK-NEXT: bne .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb
; CHECK-NOT: mcr

; CHECK: mov r0, r[[OLD]]
   ret i32 %old
}

define void @test_atomic_load_or_i64(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_or_i64:
   %old = atomicrmw or i64* @var64, i64 %offset release
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw r[[ADDR:[0-9]+]], :lower16:var64
; CHECK: movt r[[ADDR]], :upper16:var64

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldrexd r[[OLD1:[0-9]+]], r[[OLD2:[0-9]+]], [r[[ADDR]]]
  ; r0, r1 below is a reasonable guess but could change: it certainly comes into the
  ; function there.
; CHECK-LE-DAG: orr{{(\.w)?}} [[NEW1:r[0-9]+|lr]], r[[OLD1]], r0
; CHECK-LE-DAG: orr{{(\.w)?}} [[NEW2:r[0-9]+|lr]], r[[OLD2]], r1
; CHECK-BE-DAG: orr{{(\.w)?}} [[NEW2:r[0-9]+|lr]], r[[OLD2]], r1
; CHECK-BE-DAG: orr{{(\.w)?}} [[NEW1:r[0-9]+|lr]], r[[OLD1]], r0
; CHECK: stlexd [[STATUS:r[0-9]+]], [[NEW1]], [[NEW2]], [r[[ADDR]]]
; CHECK-NEXT: cmp [[STATUS]], #0
; CHECK-NEXT: bne .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb
; CHECK-NOT: mcr

; CHECK: strd r[[OLD1]], r[[OLD2]], [r[[ADDR]]]
   store i64 %old, i64* @var64
   ret void
}

define i8 @test_atomic_load_xor_i8(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xor_i8:
   %old = atomicrmw xor i8* @var8, i8 %offset acquire
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw r[[ADDR:[0-9]+]], :lower16:var8
; CHECK: movt r[[ADDR]], :upper16:var8

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldaexb r[[OLD:[0-9]+]], [r[[ADDR]]]
  ; r0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: eor{{(\.w)?}} [[NEW:r[0-9]+]], r[[OLD]], r0
; CHECK-NEXT: strexb [[STATUS:r[0-9]+]], [[NEW]], [r[[ADDR]]]
; CHECK-NEXT: cmp [[STATUS]], #0
; CHECK-NEXT: bne .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb
; CHECK-NOT: mcr

; CHECK: mov r0, r[[OLD]]
   ret i8 %old
}

define i16 @test_atomic_load_xor_i16(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xor_i16:
   %old = atomicrmw xor i16* @var16, i16 %offset release
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw r[[ADDR:[0-9]+]], :lower16:var16
; CHECK: movt r[[ADDR]], :upper16:var16

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldrexh r[[OLD:[0-9]+]], [r[[ADDR]]]
  ; r0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: eor{{(\.w)?}} [[NEW:r[0-9]+]], r[[OLD]], r0
; CHECK-NEXT: stlexh [[STATUS:r[0-9]+]], [[NEW]], [r[[ADDR]]]
; CHECK-NEXT: cmp [[STATUS]], #0
; CHECK-NEXT: bne .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb
; CHECK-NOT: mcr

; CHECK: mov r0, r[[OLD]]
   ret i16 %old
}

define i32 @test_atomic_load_xor_i32(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xor_i32:
   %old = atomicrmw xor i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw r[[ADDR:[0-9]+]], :lower16:var32
; CHECK: movt r[[ADDR]], :upper16:var32

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldaex r[[OLD:[0-9]+]], [r[[ADDR]]]
  ; r0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: eor{{(\.w)?}} [[NEW:r[0-9]+]], r[[OLD]], r0
; CHECK-NEXT: stlex [[STATUS:r[0-9]+]], [[NEW]], [r[[ADDR]]]
; CHECK-NEXT: cmp [[STATUS]], #0
; CHECK-NEXT: bne .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb
; CHECK-NOT: mcr

; CHECK: mov r0, r[[OLD]]
   ret i32 %old
}

define void @test_atomic_load_xor_i64(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xor_i64:
   %old = atomicrmw xor i64* @var64, i64 %offset monotonic
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw r[[ADDR:[0-9]+]], :lower16:var64
; CHECK: movt r[[ADDR]], :upper16:var64

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldrexd r[[OLD1:[0-9]+]], r[[OLD2:[0-9]+]], [r[[ADDR]]]
  ; r0, r1 below is a reasonable guess but could change: it certainly comes into the
  ; function there.
; CHECK-LE-DAG: eor{{(\.w)?}} [[NEW1:r[0-9]+|lr]], r[[OLD1]], r0
; CHECK-LE-DAG: eor{{(\.w)?}} [[NEW2:r[0-9]+|lr]], r[[OLD2]], r1
; CHECK-BE-DAG: eor{{(\.w)?}} [[NEW2:r[0-9]+|lr]], r[[OLD2]], r1
; CHECK-BE-DAG: eor{{(\.w)?}} [[NEW1:r[0-9]+|lr]], r[[OLD1]], r0
; CHECK: strexd [[STATUS:r[0-9]+]], [[NEW1]], [[NEW2]], [r[[ADDR]]]
; CHECK-NEXT: cmp [[STATUS]], #0
; CHECK-NEXT: bne .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb
; CHECK-NOT: mcr

; CHECK: strd r[[OLD1]], r[[OLD2]], [r[[ADDR]]]
   store i64 %old, i64* @var64
   ret void
}

define i8 @test_atomic_load_xchg_i8(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xchg_i8:
   %old = atomicrmw xchg i8* @var8, i8 %offset monotonic
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw r[[ADDR:[0-9]+]], :lower16:var8
; CHECK: movt r[[ADDR]], :upper16:var8

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldrexb r[[OLD:[0-9]+]], [r[[ADDR]]]
  ; r0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: strexb [[STATUS:r[0-9]+]], r0, [r[[ADDR]]]
; CHECK-NEXT: cmp [[STATUS]], #0
; CHECK-NEXT: bne .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb
; CHECK-NOT: mcr

; CHECK: mov r0, r[[OLD]]
   ret i8 %old
}

define i16 @test_atomic_load_xchg_i16(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xchg_i16:
   %old = atomicrmw xchg i16* @var16, i16 %offset seq_cst
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw r[[ADDR:[0-9]+]], :lower16:var16
; CHECK: movt r[[ADDR]], :upper16:var16

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldaexh r[[OLD:[0-9]+]], [r[[ADDR]]]
  ; r0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: stlexh [[STATUS:r[0-9]+]], r0, [r[[ADDR]]]
; CHECK-NEXT: cmp [[STATUS]], #0
; CHECK-NEXT: bne .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb
; CHECK-NOT: mcr

; CHECK: mov r0, r[[OLD]]
   ret i16 %old
}

define i32 @test_atomic_load_xchg_i32(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xchg_i32:
   %old = atomicrmw xchg i32* @var32, i32 %offset release
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw r[[ADDR:[0-9]+]], :lower16:var32
; CHECK: movt r[[ADDR]], :upper16:var32

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldrex r[[OLD:[0-9]+]], [r[[ADDR]]]
  ; r0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: stlex [[STATUS:r[0-9]+]], r0, [r[[ADDR]]]
; CHECK-NEXT: cmp [[STATUS]], #0
; CHECK-NEXT: bne .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb
; CHECK-NOT: mcr

; CHECK: mov r0, r[[OLD]]
   ret i32 %old
}

define void @test_atomic_load_xchg_i64(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_xchg_i64:
   %old = atomicrmw xchg i64* @var64, i64 %offset acquire
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw r[[ADDR:[0-9]+]], :lower16:var64
; CHECK: movt r[[ADDR]], :upper16:var64

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldaexd [[OLD1:r[0-9]+]], [[OLD2:r[0-9]+|lr]], [r[[ADDR]]]
  ; r0, r1 below is a reasonable guess but could change: it certainly comes into the
  ; function there.
; CHECK-NEXT: strexd [[STATUS:r[0-9]+]], r0, r1, [r[[ADDR]]]
; CHECK-NEXT: cmp [[STATUS]], #0
; CHECK-NEXT: bne .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb
; CHECK-NOT: mcr

; CHECK: strd [[OLD1]], [[OLD2]], [r[[ADDR]]]
   store i64 %old, i64* @var64
   ret void
}

define i8 @test_atomic_load_min_i8(i8 signext %offset) nounwind {
; CHECK-LABEL: test_atomic_load_min_i8:
   %old = atomicrmw min i8* @var8, i8 %offset acquire
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK-DAG: movw [[ADDR:r[0-9]+|lr]], :lower16:var8
; CHECK-DAG: movt [[ADDR]], :upper16:var8

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldaexb r[[OLD:[0-9]+]], {{.*}}[[ADDR]]
; CHECK-NEXT: sxtb r[[OLDX:[0-9]+]], r[[OLD]]
  ; r0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: cmp r[[OLDX]], r0
; Thumb mode: it le
; CHECK:      movle r[[OLDX]], r[[OLD]]
; CHECK-NEXT: strexb [[STATUS:r[0-9]+]], r[[OLDX]], {{.*}}[[ADDR]]]
; CHECK-NEXT: cmp [[STATUS]], #0
; CHECK-NEXT: bne .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb
; CHECK-NOT: mcr

; CHECK: mov r0, r[[OLD]]
   ret i8 %old
}

define i16 @test_atomic_load_min_i16(i16 signext %offset) nounwind {
; CHECK-LABEL: test_atomic_load_min_i16:
   %old = atomicrmw min i16* @var16, i16 %offset release
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw [[ADDR:r[0-9]+|lr]], :lower16:var16
; CHECK: movt [[ADDR]], :upper16:var16

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldrexh r[[OLD:[0-9]+]], {{.*}}[[ADDR]]
; CHECK-NEXT: sxth r[[OLDX:[0-9]+]], r[[OLD]]
  ; r0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: cmp r[[OLDX]], r0
; Thumb mode: it le
; CHECK:      movle r[[OLDX]], r[[OLD]]
; CHECK-NEXT: stlexh [[STATUS:r[0-9]+]], r[[OLDX]], {{.*}}[[ADDR]]
; CHECK-NEXT: cmp [[STATUS]], #0
; CHECK-NEXT: bne .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb
; CHECK-NOT: mcr

; CHECK: mov r0, r[[OLD]]
   ret i16 %old
}

define i32 @test_atomic_load_min_i32(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_min_i32:
   %old = atomicrmw min i32* @var32, i32 %offset monotonic
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw r[[ADDR:[0-9]+]], :lower16:var32
; CHECK: movt r[[ADDR]], :upper16:var32

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldrex r[[OLD:[0-9]+]], [r[[ADDR]]]
  ; r0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: mov r[[NEW:[0-9]+]], r0
; CHECK-NEXT: cmp r[[OLD]], r0
; Thumb mode: it le
; CHECK:      movle r[[NEW]], r[[OLD]]
; CHECK-NEXT: strex [[STATUS:r[0-9]+]], r[[NEW]], [r[[ADDR]]]
; CHECK-NEXT: cmp [[STATUS]], #0
; CHECK-NEXT: bne .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb
; CHECK-NOT: mcr

; CHECK: mov r0, r[[OLD]]
   ret i32 %old
}

define void @test_atomic_load_min_i64(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_min_i64:
   %old = atomicrmw min i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw r[[ADDR:[0-9]+]], :lower16:var64
; CHECK: movt r[[ADDR]], :upper16:var64

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldaexd [[OLD1:r[0-9]+|lr]], [[OLD2:r[0-9]+|lr]], [r[[ADDR]]]
  ; r0, r1 below is a reasonable guess but could change: it certainly comes into the
  ; function there.
; CHECK-ARM: mov [[MINHI:r[0-9]+]], r1
; CHECK-ARM-LE: subs {{[^,]+}}, r0, [[OLD1]]
; CHECK-ARM-LE: sbcs {{[^,]+}}, r1, [[OLD2]]
; CHECK-ARM-BE: subs {{[^,]+}}, r1, [[OLD2]]
; CHECK-ARM-BE: sbcs {{[^,]+}}, r0, [[OLD1]]
; CHECK-ARM: mov [[CMP:r[0-9]+|lr]], #0
; CHECK-ARM: movwge [[CMP:r[0-9]+|lr]], #1
; CHECK-ARM: cmp [[CMP:r[0-9]+|lr]], #0
; CHECK-ARM: movne [[MINHI]], [[OLD2]]
; CHECK-ARM: mov [[MINLO:r[0-9]+]], r0
; CHECK-ARM: movne [[MINLO]], [[OLD1]]
; CHECK-ARM: stlexd [[STATUS:r[0-9]+]], [[MINLO]], [[MINHI]], [r[[ADDR]]]
; CHECK-THUMB: stlexd [[STATUS:r[0-9]+]], {{r[0-9]+}}, {{r[0-9]+}}, [r[[ADDR]]]
; CHECK-NEXT: cmp [[STATUS]], #0
; CHECK-NEXT: bne .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb
; CHECK-NOT: mcr

; CHECK-ARM: strd [[OLD1]], [[OLD2]], [r[[ADDR]]]
   store i64 %old, i64* @var64
   ret void
}

define i8 @test_atomic_load_max_i8(i8 signext %offset) nounwind {
; CHECK-LABEL: test_atomic_load_max_i8:
   %old = atomicrmw max i8* @var8, i8 %offset seq_cst
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw [[ADDR:r[0-9]+|lr]], :lower16:var8
; CHECK: movt [[ADDR]], :upper16:var8

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldaexb r[[OLD:[0-9]+]], {{.*}}[[ADDR]]
; CHECK-NEXT: sxtb r[[OLDX:[0-9]+]], r[[OLD]]
  ; r0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: cmp r[[OLDX]], r0
; Thumb mode: it gt
; CHECK:      movgt r[[OLDX]], r[[OLD]]
; CHECK-NEXT: stlexb [[STATUS:r[0-9]+]], r[[OLDX]], {{.*}}[[ADDR]]
; CHECK-NEXT: cmp [[STATUS]], #0
; CHECK-NEXT: bne .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb
; CHECK-NOT: mcr

; CHECK: mov r0, r[[OLD]]
   ret i8 %old
}

define i16 @test_atomic_load_max_i16(i16 signext %offset) nounwind {
; CHECK-LABEL: test_atomic_load_max_i16:
   %old = atomicrmw max i16* @var16, i16 %offset acquire
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw r[[ADDR:[0-9]+]], :lower16:var16
; CHECK: movt r[[ADDR]], :upper16:var16

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldaexh r[[OLD:[0-9]+]], [r[[ADDR]]]
; CHECK-NEXT: sxth r[[OLDX:[0-9]+]], r[[OLD]]
  ; r0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: cmp r[[OLDX]], r0
; Thumb mode: it gt
; CHECK:      movgt r[[OLDX]], r[[OLD]]
; CHECK-NEXT: strexh [[STATUS:r[0-9]+]], r[[OLDX]], [r[[ADDR]]]
; CHECK-NEXT: cmp [[STATUS]], #0
; CHECK-NEXT: bne .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb
; CHECK-NOT: mcr

; CHECK: mov r0, r[[OLD]]
   ret i16 %old
}

define i32 @test_atomic_load_max_i32(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_max_i32:
   %old = atomicrmw max i32* @var32, i32 %offset release
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw r[[ADDR:[0-9]+]], :lower16:var32
; CHECK: movt r[[ADDR]], :upper16:var32

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldrex r[[OLD:[0-9]+]], [r[[ADDR]]]
  ; r0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: mov r[[NEW:[0-9]+]], r0
; CHECK-NEXT: cmp r[[OLD]], r0
; Thumb mode: it gt
; CHECK:      movgt r[[NEW]], r[[OLD]]
; CHECK-NEXT: stlex [[STATUS:r[0-9]+]], r[[NEW]], [r[[ADDR]]]
; CHECK-NEXT: cmp [[STATUS]], #0
; CHECK-NEXT: bne .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb
; CHECK-NOT: mcr

; CHECK: mov r0, r[[OLD]]
   ret i32 %old
}

define void @test_atomic_load_max_i64(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_max_i64:
   %old = atomicrmw max i64* @var64, i64 %offset monotonic
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw r[[ADDR:[0-9]+]], :lower16:var64
; CHECK: movt r[[ADDR]], :upper16:var64

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldrexd [[OLD1:r[0-9]+]], [[OLD2:r[0-9]+|lr]], [r[[ADDR]]]
  ; r0, r1 below is a reasonable guess but could change: it certainly comes into the
  ; function there.
; CHECK-ARM: mov [[MINHI:r[0-9]+]], r1
; CHECK-ARM-LE: subs {{[^,]+}}, r0, [[OLD1]]
; CHECK-ARM-LE: sbcs {{[^,]+}}, r1, [[OLD2]]
; CHECK-ARM-BE: subs {{[^,]+}}, r1, [[OLD2]]
; CHECK-ARM-BE: sbcs {{[^,]+}}, r0, [[OLD1]]
; CHECK-ARM: mov [[CMP:r[0-9]+|lr]], #0
; CHECK-ARM: movwlt [[CMP:r[0-9]+|lr]], #1
; CHECK-ARM: cmp [[CMP:r[0-9]+|lr]], #0
; CHECK-ARM: movne [[MINHI]], [[OLD2]]
; CHECK-ARM: mov [[MINLO:r[0-9]+]], r0
; CHECK-ARM: movne [[MINLO]], [[OLD1]]
; CHECK-ARM: strexd [[STATUS:r[0-9]+]], [[MINLO]], [[MINHI]], [r[[ADDR]]]
; CHECK-THUMB: strexd [[STATUS:r[0-9]+]], {{r[0-9]+}}, {{r[0-9]+}}, [r[[ADDR]]]
; CHECK-NEXT: cmp [[STATUS]], #0
; CHECK-NEXT: bne .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb
; CHECK-NOT: mcr

; CHECK-ARM: strd [[OLD1]], [[OLD2]], [r[[ADDR]]]
   store i64 %old, i64* @var64
   ret void
}

define i8 @test_atomic_load_umin_i8(i8 zeroext %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umin_i8:
   %old = atomicrmw umin i8* @var8, i8 %offset monotonic
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw [[ADDR:r[0-9]+|lr]], :lower16:var8
; CHECK: movt [[ADDR]], :upper16:var8

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldrexb r[[OLD:[0-9]+]], {{.*}}[[ADDR]]
  ; r0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: mov r[[NEW:[0-9]+]], r0
; CHECK-NEXT: cmp r[[OLD]], r0
; Thumb mode: it ls
; CHECK:      movls r[[NEW]], r[[OLD]]
; CHECK-NEXT: strexb [[STATUS:r[0-9]+]], r[[NEW]], {{.*}}[[ADDR]]
; CHECK-NEXT: cmp [[STATUS]], #0
; CHECK-NEXT: bne .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb
; CHECK-NOT: mcr

; CHECK: mov r0, r[[OLD]]
   ret i8 %old
}

define i16 @test_atomic_load_umin_i16(i16 zeroext %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umin_i16:
   %old = atomicrmw umin i16* @var16, i16 %offset acquire
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw [[ADDR:r[0-9]+|lr]], :lower16:var16
; CHECK: movt [[ADDR]], :upper16:var16

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldaexh r[[OLD:[0-9]+]], {{.*}}[[ADDR]]
  ; r0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: mov r[[NEW:[0-9]+]], r0
; CHECK-NEXT: cmp r[[OLD]], r0
; Thumb mode: it ls
; CHECK:      movls r[[NEW]], r[[OLD]]
; CHECK-NEXT: strexh [[STATUS:r[0-9]+]], r[[NEW]], {{.*}}[[ADDR]]
; CHECK-NEXT: cmp [[STATUS]], #0
; CHECK-NEXT: bne .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb
; CHECK-NOT: mcr

; CHECK: mov r0, r[[OLD]]
   ret i16 %old
}

define i32 @test_atomic_load_umin_i32(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umin_i32:
   %old = atomicrmw umin i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw r[[ADDR:[0-9]+]], :lower16:var32
; CHECK: movt r[[ADDR]], :upper16:var32

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldaex r[[OLD:[0-9]+]], [r[[ADDR]]]
  ; r0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: mov r[[NEW:[0-9]+]], r0
; CHECK-NEXT: cmp r[[OLD]], r0
; Thumb mode: it ls
; CHECK:      movls r[[NEW]], r[[OLD]]
; CHECK-NEXT: stlex [[STATUS:r[0-9]+]], r[[NEW]], [r[[ADDR]]]
; CHECK-NEXT: cmp [[STATUS]], #0
; CHECK-NEXT: bne .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb
; CHECK-NOT: mcr

; CHECK: mov r0, r[[OLD]]
   ret i32 %old
}

define void @test_atomic_load_umin_i64(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umin_i64:
   %old = atomicrmw umin i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw r[[ADDR:[0-9]+]], :lower16:var64
; CHECK: movt r[[ADDR]], :upper16:var64

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldaexd [[OLD1:r[0-9]+|lr]], [[OLD2:r[0-9]+|lr]], [r[[ADDR]]]
  ; r0, r1 below is a reasonable guess but could change: it certainly comes into the
  ; function there.
; CHECK-ARM: mov [[MINHI:r[0-9]+]], r1
; CHECK-ARM-LE: subs {{[^,]+}}, r0, [[OLD1]]
; CHECK-ARM-LE: sbcs {{[^,]+}}, r1, [[OLD2]]
; CHECK-ARM-BE: subs {{[^,]+}}, r1, [[OLD2]]
; CHECK-ARM-BE: sbcs {{[^,]+}}, r0, [[OLD1]]
; CHECK-ARM: mov [[CMP:r[0-9]+|lr]], #0
; CHECK-ARM: movwhs [[CMP:r[0-9]+|lr]], #1
; CHECK-ARM: cmp [[CMP:r[0-9]+|lr]], #0
; CHECK-ARM: movne [[MINHI]], [[OLD2]]
; CHECK-ARM: mov [[MINLO:r[0-9]+]], r0
; CHECK-ARM: movne [[MINLO]], [[OLD1]]
; CHECK-ARM: stlexd [[STATUS:r[0-9]+]], [[MINLO]], [[MINHI]], [r[[ADDR]]]
; CHECK-THUMB: stlexd [[STATUS:r[0-9]+]], {{r[0-9]+}}, {{r[0-9]+}}, [r[[ADDR]]]
; CHECK-NEXT: cmp [[STATUS]], #0
; CHECK-NEXT: bne .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb
; CHECK-NOT: mcr

; CHECK-ARM: strd [[OLD1]], [[OLD2]], [r[[ADDR]]]
   store i64 %old, i64* @var64
   ret void
}

define i8 @test_atomic_load_umax_i8(i8 zeroext %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umax_i8:
   %old = atomicrmw umax i8* @var8, i8 %offset acq_rel
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw [[ADDR:r[0-9]+|lr]], :lower16:var8
; CHECK: movt [[ADDR]], :upper16:var8

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldaexb r[[OLD:[0-9]+]], {{.*}}[[ADDR]]
  ; r0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: mov r[[NEW:[0-9]+]], r0
; CHECK-NEXT: cmp r[[OLD]], r0
; Thumb mode: it hi
; CHECK:      movhi r[[NEW]], r[[OLD]]
; CHECK-NEXT: stlexb [[STATUS:r[0-9]+]], r[[NEW]], {{.*}}[[ADDR]]
; CHECK-NEXT: cmp [[STATUS]], #0
; CHECK-NEXT: bne .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb
; CHECK-NOT: mcr

; CHECK: mov r0, r[[OLD]]
   ret i8 %old
}

define i16 @test_atomic_load_umax_i16(i16 zeroext %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umax_i16:
   %old = atomicrmw umax i16* @var16, i16 %offset monotonic
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw [[ADDR:r[0-9]+|lr]], :lower16:var16
; CHECK: movt [[ADDR]], :upper16:var16

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldrexh r[[OLD:[0-9]+]], {{.*}}[[ADDR]]
  ; r0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: mov r[[NEW:[0-9]+]], r0
; CHECK-NEXT: cmp r[[OLD]], r0
; Thumb mode: it hi
; CHECK:      movhi r[[NEW]], r[[OLD]]
; CHECK-NEXT: strexh [[STATUS:r[0-9]+]], r[[NEW]], {{.*}}[[ADDR]]
; CHECK-NEXT: cmp [[STATUS]], #0
; CHECK-NEXT: bne .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb
; CHECK-NOT: mcr

; CHECK: mov r0, r[[OLD]]
   ret i16 %old
}

define i32 @test_atomic_load_umax_i32(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umax_i32:
   %old = atomicrmw umax i32* @var32, i32 %offset seq_cst
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw r[[ADDR:[0-9]+]], :lower16:var32
; CHECK: movt r[[ADDR]], :upper16:var32

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldaex r[[OLD:[0-9]+]], [r[[ADDR]]]
  ; r0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: mov r[[NEW:[0-9]+]], r0
; CHECK-NEXT: cmp r[[OLD]], r0
; Thumb mode: it hi
; CHECK:      movhi r[[NEW]], r[[OLD]]
; CHECK-NEXT: stlex [[STATUS:r[0-9]+]], r[[NEW]], [r[[ADDR]]]
; CHECK-NEXT: cmp [[STATUS]], #0
; CHECK-NEXT: bne .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb
; CHECK-NOT: mcr

; CHECK: mov r0, r[[OLD]]
   ret i32 %old
}

define void @test_atomic_load_umax_i64(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_umax_i64:
   %old = atomicrmw umax i64* @var64, i64 %offset seq_cst
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw r[[ADDR:[0-9]+]], :lower16:var64
; CHECK: movt r[[ADDR]], :upper16:var64

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldaexd [[OLD1:r[0-9]+|lr]], [[OLD2:r[0-9]+|lr]], [r[[ADDR]]]
  ; r0, r1 below is a reasonable guess but could change: it certainly comes into the
  ; function there.
; CHECK-ARM: mov [[MINHI:r[0-9]+]], r1
; CHECK-ARM-LE: subs {{[^,]+}}, r0, [[OLD1]]
; CHECK-ARM-LE: sbcs {{[^,]+}}, r1, [[OLD2]]
; CHECK-ARM-BE: subs {{[^,]+}}, r1, [[OLD2]]
; CHECK-ARM-BE: sbcs {{[^,]+}}, r0, [[OLD1]]
; CHECK-ARM: mov [[CMP:r[0-9]+|lr]], #0
; CHECK-ARM: movwlo [[CMP:r[0-9]+|lr]], #1
; CHECK-ARM: cmp [[CMP:r[0-9]+|lr]], #0
; CHECK-ARM: movne [[MINHI]], [[OLD2]]
; CHECK-ARM: mov [[MINLO:r[0-9]+]], r0
; CHECK-ARM: movne [[MINLO]], [[OLD1]]
; CHECK-ARM: stlexd [[STATUS:r[0-9]+]], [[MINLO]], [[MINHI]], [r[[ADDR]]]
; CHECK-THUMB: stlexd [[STATUS:r[0-9]+]], {{r[0-9]+}}, {{r[0-9]+}}, [r[[ADDR]]]
; CHECK-NEXT: cmp [[STATUS]], #0
; CHECK-NEXT: bne .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb
; CHECK-NOT: mcr

; CHECK-ARM: strd [[OLD1]], [[OLD2]], [r[[ADDR]]]
   store i64 %old, i64* @var64
   ret void
}

define i8 @test_atomic_cmpxchg_i8(i8 zeroext %wanted, i8 zeroext %new) nounwind {
; CHECK-LABEL: test_atomic_cmpxchg_i8:
   %pair = cmpxchg i8* @var8, i8 %wanted, i8 %new acquire acquire
   %old = extractvalue { i8, i1 } %pair, 0
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK-DAG: movw r[[ADDR:[0-9]+]], :lower16:var8
; CHECK-DAG: movt r[[ADDR]], :upper16:var8
; CHECK-THUMB-DAG: mov r[[WANTED:[0-9]+]], r0

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldaexb r[[OLD:[0-9]+]], [r[[ADDR]]]
  ; r0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-ARM-NEXT:   cmp r[[OLD]], r0
; CHECK-THUMB-NEXT: cmp r[[OLD]], r[[WANTED]]
; CHECK-NEXT: bne .LBB{{[0-9]+}}_4
; CHECK-NEXT: %bb.2:
  ; As above, r1 is a reasonable guess.
; CHECK: strexb [[STATUS:r[0-9]+]], r1, [r[[ADDR]]]
; CHECK-NEXT: cmp [[STATUS]], #0
; CHECK-NEXT: bne .LBB{{[0-9]+}}_1
; CHECK-ARM: mov r0, r[[OLD]]
; CHECK: bx lr
; CHECK-NEXT: .LBB{{[0-9]+}}_4:
; CHECK-NEXT: clrex
; CHECK-NOT: dmb
; CHECK-NOT: mcr

; CHECK-ARM: mov r0, r[[OLD]]
; CHECK-ARM-NEXT: bx lr
   ret i8 %old
}

define i16 @test_atomic_cmpxchg_i16(i16 zeroext %wanted, i16 zeroext %new) nounwind {
; CHECK-LABEL: test_atomic_cmpxchg_i16:
   %pair = cmpxchg i16* @var16, i16 %wanted, i16 %new seq_cst seq_cst
   %old = extractvalue { i16, i1 } %pair, 0
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK-DAG: movw r[[ADDR:[0-9]+]], :lower16:var16
; CHECK-DAG: movt r[[ADDR]], :upper16:var16
; CHECK-THUMB-DAG: mov r[[WANTED:[0-9]+]], r0

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldaexh r[[OLD:[0-9]+]], [r[[ADDR]]]
  ; r0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-ARM-NEXT:   cmp r[[OLD]], r0
; CHECK-THUMB-NEXT: cmp r[[OLD]], r[[WANTED]]
; CHECK-NEXT: bne .LBB{{[0-9]+}}_4
; CHECK-NEXT: %bb.2:
  ; As above, r1 is a reasonable guess.
; CHECK: stlexh [[STATUS:r[0-9]+]], r1, [r[[ADDR]]]
; CHECK-NEXT: cmp [[STATUS]], #0
; CHECK-NEXT: bne .LBB{{[0-9]+}}_1
; CHECK-ARM: mov r0, r[[OLD]]
; CHECK: bx lr
; CHECK-NEXT: .LBB{{[0-9]+}}_4:
; CHECK-NEXT: clrex
; CHECK-NOT: dmb
; CHECK-NOT: mcr

; CHECK-ARM: mov r0, r[[OLD]]
; CHECK-ARM-NEXT: bx lr
   ret i16 %old
}

define void @test_atomic_cmpxchg_i32(i32 %wanted, i32 %new) nounwind {
; CHECK-LABEL: test_atomic_cmpxchg_i32:
   %pair = cmpxchg i32* @var32, i32 %wanted, i32 %new release monotonic
   %old = extractvalue { i32, i1 } %pair, 0
   store i32 %old, i32* @var32
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw r[[ADDR:[0-9]+]], :lower16:var32
; CHECK: movt r[[ADDR]], :upper16:var32

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldrex r[[OLD:[0-9]+]], [r[[ADDR]]]
  ; r0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: cmp r[[OLD]], r0
; CHECK-NEXT: bne .LBB{{[0-9]+}}_4
; CHECK-NEXT: %bb.2:
  ; As above, r1 is a reasonable guess.
; CHECK: stlex [[STATUS:r[0-9]+]], r1, [r[[ADDR]]]
; CHECK-NEXT: cmp [[STATUS]], #0
; CHECK-NEXT: bne .LBB{{[0-9]+}}_1
; CHECK: str{{(.w)?}} r[[OLD]],
; CHECK-NEXT: bx lr
; CHECK-NEXT: .LBB{{[0-9]+}}_4:
; CHECK-NEXT: clrex
; CHECK-NOT: dmb
; CHECK-NOT: mcr

; CHECK: str{{(.w)?}} r[[OLD]],
; CHECK-ARM-NEXT: bx lr
   ret void
}

define void @test_atomic_cmpxchg_i64(i64 %wanted, i64 %new) nounwind {
; CHECK-LABEL: test_atomic_cmpxchg_i64:
   %pair = cmpxchg i64* @var64, i64 %wanted, i64 %new monotonic monotonic
   %old = extractvalue { i64, i1 } %pair, 0
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw r[[ADDR:[0-9]+]], :lower16:var64
; CHECK: movt r[[ADDR]], :upper16:var64

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldrexd [[OLD1:r[0-9]+|lr]], [[OLD2:r[0-9]+|lr]], [r[[ADDR]]]
  ; r0, r1 below is a reasonable guess but could change: it certainly comes into the
  ; function there.
; CHECK-LE-DAG: eor{{(\.w)?}} [[MISMATCH_LO:r[0-9]+|lr]], [[OLD1]], r0
; CHECK-LE-DAG: eor{{(\.w)?}} [[MISMATCH_HI:r[0-9]+|lr]], [[OLD2]], r1
; CHECK-ARM-LE: orrs{{(\.w)?}} {{r[0-9]+}}, [[MISMATCH_LO]], [[MISMATCH_HI]]
; CHECK-THUMB-LE: orrs{{(\.w)?}} {{(r[0-9]+, )?}}[[MISMATCH_HI]], [[MISMATCH_LO]]
; CHECK-BE-DAG: eor{{(\.w)?}} [[MISMATCH_HI:r[0-9]+|lr]], [[OLD2]], r1
; CHECK-BE-DAG: eor{{(\.w)?}} [[MISMATCH_LO:r[0-9]+|lr]], [[OLD1]], r0
; CHECK-ARM-BE: orrs{{(\.w)?}} {{r[0-9]+}}, [[MISMATCH_HI]], [[MISMATCH_LO]]
; CHECK-THUMB-BE: orrs{{(\.w)?}} {{(r[0-9]+, )?}}[[MISMATCH_LO]], [[MISMATCH_HI]]
; CHECK-NEXT: bne .LBB{{[0-9]+}}_4
; CHECK-NEXT: %bb.2:
  ; As above, r2, r3 is a reasonable guess.
; CHECK: strexd [[STATUS:r[0-9]+]], r2, r3, [r[[ADDR]]]
; CHECK-NEXT: cmp [[STATUS]], #0
; CHECK-NEXT: bne .LBB{{[0-9]+}}_1
; CHECK: strd [[OLD1]], [[OLD2]], [r[[ADDR]]]
; CHECK-NEXT: pop
; CHECK-NEXT: .LBB{{[0-9]+}}_4:
; CHECK-NEXT: clrex
; CHECK-NOT: dmb
; CHECK-NOT: mcr

; CHECK-ARM: strd [[OLD1]], [[OLD2]], [r[[ADDR]]]
   store i64 %old, i64* @var64
   ret void
}

define i8 @test_atomic_load_monotonic_i8() nounwind {
; CHECK-LABEL: test_atomic_load_monotonic_i8:
  %val = load atomic i8, i8* @var8 monotonic, align 1
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw r[[ADDR:[0-9]+]], :lower16:var8
; CHECK: movt r[[ADDR]], :upper16:var8
; CHECK: ldrb r0, [r[[ADDR]]]
; CHECK-NOT: dmb
; CHECK-NOT: mcr

  ret i8 %val
}

define i8 @test_atomic_load_monotonic_regoff_i8(i64 %base, i64 %off) nounwind {
; CHECK-LABEL: test_atomic_load_monotonic_regoff_i8:
  %addr_int = add i64 %base, %off
  %addr = inttoptr i64 %addr_int to i8*

  %val = load atomic i8, i8* %addr monotonic, align 1
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK-LE: ldrb r0, [r0, r2]
; CHECK-BE: ldrb r0, [r1, r3]
; CHECK-NOT: dmb
; CHECK-NOT: mcr

  ret i8 %val
}

define i8 @test_atomic_load_acquire_i8() nounwind {
; CHECK-LABEL: test_atomic_load_acquire_i8:
  %val = load atomic i8, i8* @var8 acquire, align 1
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw r[[ADDR:[0-9]+]], :lower16:var8
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movt r[[ADDR]], :upper16:var8
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: ldab r0, [r[[ADDR]]]
; CHECK-NOT: dmb
; CHECK-NOT: mcr
  ret i8 %val
}

define i8 @test_atomic_load_seq_cst_i8() nounwind {
; CHECK-LABEL: test_atomic_load_seq_cst_i8:
  %val = load atomic i8, i8* @var8 seq_cst, align 1
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw r[[ADDR:[0-9]+]], :lower16:var8
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movt r[[ADDR]], :upper16:var8
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: ldab r0, [r[[ADDR]]]
; CHECK-NOT: dmb
; CHECK-NOT: mcr
  ret i8 %val
}

define i16 @test_atomic_load_monotonic_i16() nounwind {
; CHECK-LABEL: test_atomic_load_monotonic_i16:
  %val = load atomic i16, i16* @var16 monotonic, align 2
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw r[[ADDR:[0-9]+]], :lower16:var16
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movt r[[ADDR]], :upper16:var16
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: ldrh r0, [r[[ADDR]]]
; CHECK-NOT: dmb
; CHECK-NOT: mcr

  ret i16 %val
}

define i32 @test_atomic_load_monotonic_regoff_i32(i64 %base, i64 %off) nounwind {
; CHECK-LABEL: test_atomic_load_monotonic_regoff_i32:
  %addr_int = add i64 %base, %off
  %addr = inttoptr i64 %addr_int to i32*

  %val = load atomic i32, i32* %addr monotonic, align 4
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK-LE: ldr r0, [r0, r2]
; CHECK-BE: ldr r0, [r1, r3]
; CHECK-NOT: dmb
; CHECK-NOT: mcr

  ret i32 %val
}

define i64 @test_atomic_load_seq_cst_i64() nounwind {
; CHECK-LABEL: test_atomic_load_seq_cst_i64:
  %val = load atomic i64, i64* @var64 seq_cst, align 8
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw r[[ADDR:[0-9]+]], :lower16:var64
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movt r[[ADDR]], :upper16:var64
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: ldaexd r0, r1, [r[[ADDR]]]
; CHECK-NOT: dmb
; CHECK-NOT: mcr
  ret i64 %val
}

define void @test_atomic_store_monotonic_i8(i8 %val) nounwind {
; CHECK-LABEL: test_atomic_store_monotonic_i8:
  store atomic i8 %val, i8* @var8 monotonic, align 1
; CHECK: movw r[[ADDR:[0-9]+]], :lower16:var8
; CHECK: movt r[[ADDR]], :upper16:var8
; CHECK: strb r0, [r[[ADDR]]]

  ret void
}

define void @test_atomic_store_monotonic_regoff_i8(i64 %base, i64 %off, i8 %val) nounwind {
; CHECK-LABEL: test_atomic_store_monotonic_regoff_i8:

  %addr_int = add i64 %base, %off
  %addr = inttoptr i64 %addr_int to i8*

  store atomic i8 %val, i8* %addr monotonic, align 1
; CHECK-LE: ldr{{b?(\.w)?}} [[VAL:r[0-9]+]], [sp]
; CHECK-LE: strb [[VAL]], [r0, r2]
; CHECK-BE: ldrb{{(\.w)?}} [[VAL:r[0-9]+]], [sp, #3]
; CHECK-BE: strb [[VAL]], [r1, r3]

  ret void
}

define void @test_atomic_store_release_i8(i8 %val) nounwind {
; CHECK-LABEL: test_atomic_store_release_i8:
  store atomic i8 %val, i8* @var8 release, align 1
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw r[[ADDR:[0-9]+]], :lower16:var8
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movt r[[ADDR]], :upper16:var8
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: stlb r0, [r[[ADDR]]]
; CHECK-NOT: dmb
; CHECK-NOT: mcr
  ret void
}

define void @test_atomic_store_seq_cst_i8(i8 %val) nounwind {
; CHECK-LABEL: test_atomic_store_seq_cst_i8:
  store atomic i8 %val, i8* @var8 seq_cst, align 1
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw r[[ADDR:[0-9]+]], :lower16:var8
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movt r[[ADDR]], :upper16:var8
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: stlb r0, [r[[ADDR]]]
; CHECK-NOT: dmb
; CHECK-NOT: mcr
  ret void
}

define void @test_atomic_store_monotonic_i16(i16 %val) nounwind {
; CHECK-LABEL: test_atomic_store_monotonic_i16:
  store atomic i16 %val, i16* @var16 monotonic, align 2
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw r[[ADDR:[0-9]+]], :lower16:var16
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movt r[[ADDR]], :upper16:var16
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: strh r0, [r[[ADDR]]]
; CHECK-NOT: dmb
; CHECK-NOT: mcr
  ret void
}

define void @test_atomic_store_monotonic_regoff_i32(i64 %base, i64 %off, i32 %val) nounwind {
; CHECK-LABEL: test_atomic_store_monotonic_regoff_i32:

  %addr_int = add i64 %base, %off
  %addr = inttoptr i64 %addr_int to i32*

  store atomic i32 %val, i32* %addr monotonic, align 4
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: ldr [[VAL:r[0-9]+]], [sp]
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK-LE: str [[VAL]], [r0, r2]
; CHECK-BE: str [[VAL]], [r1, r3]
; CHECK-NOT: dmb
; CHECK-NOT: mcr

  ret void
}

define void @test_atomic_store_release_i64(i64 %val) nounwind {
; CHECK-LABEL: test_atomic_store_release_i64:
  store atomic i64 %val, i64* @var64 release, align 8
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw [[ADDR:r[0-9]+|lr]], :lower16:var64
; CHECK: movt [[ADDR]], :upper16:var64

; CHECK: .LBB{{[0-9]+}}_1:
  ; r0, r1 below is a reasonable guess but could change: it certainly comes into the
  ; function there.
; CHECK: stlexd [[STATUS:r[0-9]+]], r0, r1, {{.*}}[[ADDR]]
; CHECK-NEXT: cmp [[STATUS]], #0
; CHECK-NEXT: bne .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb
; CHECK-NOT: mcr

  ret void
}

define i32 @not.barriers(i32* %var, i1 %cond) {
; CHECK-LABEL: not.barriers:
  br i1 %cond, label %atomic_ver, label %simple_ver
simple_ver:
  %oldval = load i32, i32* %var
  %newval = add nsw i32 %oldval, -1
  store i32 %newval, i32* %var
  br label %somewhere
atomic_ver:
  fence seq_cst
  %val = atomicrmw add i32* %var, i32 -1 monotonic
  fence seq_cst
  br label %somewhere
; CHECK: dmb
; CHECK: ldrex
; CHECK: dmb
  ; The key point here is that the second dmb isn't immediately followed by the
  ; simple_ver basic block, which LLVM attempted to do when DMB had been marked
  ; with isBarrier. For now, look for something that looks like "somewhere".
; CHECK-NEXT: {{mov|bx}}
somewhere:
  %combined = phi i32 [ %val, %atomic_ver ], [ %newval, %simple_ver]
  ret i32 %combined
}
