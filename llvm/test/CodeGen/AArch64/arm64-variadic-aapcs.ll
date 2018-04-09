; RUN: llc -verify-machineinstrs -mtriple=arm64-linux-gnu -pre-RA-sched=linearize -enable-misched=false -disable-post-ra < %s | FileCheck %s

%va_list = type {i8*, i8*, i8*, i32, i32}

@var = global %va_list zeroinitializer, align 8

declare void @llvm.va_start(i8*)

define void @test_simple(i32 %n, ...) {
; CHECK-LABEL: test_simple:
; CHECK: sub sp, sp, #[[STACKSIZE:[0-9]+]]
; CHECK: add [[STACK_TOP:x[0-9]+]], sp, #[[STACKSIZE]]

; CHECK: stp x1, x2, [sp, #[[GR_BASE:[0-9]+]]]
; ... omit middle ones ...
; CHECK: str x7, [sp, #

; CHECK: stp q0, q1, [sp]
; ... omit middle ones ...
; CHECK: stp q6, q7, [sp, #

; CHECK: str [[STACK_TOP]], [{{[x[0-9]+}}, :lo12:var]

; CHECK: add [[GR_TOPTMP:x[0-9]+]], sp, #[[GR_BASE]]
; CHECK: add [[GR_TOP:x[0-9]+]], [[GR_TOPTMP]], #56
; CHECK: str [[GR_TOP]], [{{[x[0-9]+}}, :lo12:var+8]

; CHECK: mov [[VR_TOPTMP:x[0-9]+]], sp
; CHECK: add [[VR_TOP:x[0-9]+]], [[VR_TOPTMP]], #128
; CHECK: str [[VR_TOP]], [{{[x[0-9]+}}, :lo12:var+16]

; CHECK: mov     [[GRVR1:w[0-9]+]], #-56
; CHECK: str     [[GRVR1]], [{{[x[0-9]+}}, :lo12:var+24]
; CHECK: orr     [[GRVR2:w[0-9]+]], wzr, #0xffffff80
; CHECK: str     [[GRVR2]], [{{[x[0-9]+}}, :lo12:var+28]

  %addr = bitcast %va_list* @var to i8*
  call void @llvm.va_start(i8* %addr)

  ret void
}

define void @test_fewargs(i32 %n, i32 %n1, i32 %n2, float %m, ...) {
; CHECK-LABEL: test_fewargs:
; CHECK: sub sp, sp, #[[STACKSIZE:[0-9]+]]
; CHECK: add [[STACK_TOP:x[0-9]+]], sp, #[[STACKSIZE]]

; CHECK: stp x3, x4, [sp, #[[GR_BASE:[0-9]+]]]
; ... omit middle ones ...
; CHECK: str x7, [sp, #

; CHECK: stp q1, q2, [sp]
; ... omit middle ones ...
; CHECK: str q7, [sp, #

; CHECK: str [[STACK_TOP]], [{{[x[0-9]+}}, :lo12:var]

; CHECK: add [[GR_TOPTMP:x[0-9]+]], sp, #[[GR_BASE]]
; CHECK: add [[GR_TOP:x[0-9]+]], [[GR_TOPTMP]], #40
; CHECK: str [[GR_TOP]], [{{[x[0-9]+}}, :lo12:var+8]

; CHECK: mov [[VR_TOPTMP:x[0-9]+]], sp
; CHECK: add [[VR_TOP:x[0-9]+]], [[VR_TOPTMP]], #112
; CHECK: str [[VR_TOP]], [{{[x[0-9]+}}, :lo12:var+16]

; CHECK: mov     [[GRVR1:w[0-9]+]], #-40
; CHECK: str     [[GRVR1]], [{{[x[0-9]+}}, :lo12:var+24]
; CHECK: mov     [[GRVR2:w[0-9]+]], #-112
; CHECK: str     [[GRVR2]], [{{[x[0-9]+}}, :lo12:var+28]

  %addr = bitcast %va_list* @var to i8*
  call void @llvm.va_start(i8* %addr)

  ret void
}

define void @test_nospare([8 x i64], [8 x float], ...) {
; CHECK-LABEL: test_nospare:

  %addr = bitcast %va_list* @var to i8*
  call void @llvm.va_start(i8* %addr)
; CHECK-NOT: sub sp, sp
; CHECK: mov [[STACK:x[0-9]+]], sp
; CHECK: str [[STACK]], [{{[x[0-9]+}}, :lo12:var]

  ret void
}

; If there are non-variadic arguments on the stack (here two i64s) then the
; __stack field should point just past them.
define void @test_offsetstack([8 x i64], [2 x i64], [3 x float], ...) {
; CHECK-LABEL: test_offsetstack:
; CHECK: stp {{q[0-9]+}}, {{q[0-9]+}}, [sp, #-80]!
; CHECK: add [[STACK_TOP:x[0-9]+]], sp, #96
; CHECK: str [[STACK_TOP]], [{{[x[0-9]+}}, :lo12:var]

  %addr = bitcast %va_list* @var to i8*
  call void @llvm.va_start(i8* %addr)
  ret void
}

declare void @llvm.va_end(i8*)

define void @test_va_end() nounwind {
; CHECK-LABEL: test_va_end:
; CHECK-NEXT: %bb.0

  %addr = bitcast %va_list* @var to i8*
  call void @llvm.va_end(i8* %addr)

  ret void
; CHECK-NEXT: ret
}

declare void @llvm.va_copy(i8* %dest, i8* %src)

@second_list = global %va_list zeroinitializer

define void @test_va_copy() {
; CHECK-LABEL: test_va_copy:
  %srcaddr = bitcast %va_list* @var to i8*
  %dstaddr = bitcast %va_list* @second_list to i8*
  call void @llvm.va_copy(i8* %dstaddr, i8* %srcaddr)

; CHECK: add x[[SRC:[0-9]+]], {{x[0-9]+}}, :lo12:var
; CHECK: ldr [[BLOCK:q[0-9]+]], [x[[SRC]]]
; CHECK: add x[[DST:[0-9]+]], {{x[0-9]+}}, :lo12:second_list
; CHECK: str [[BLOCK]], [x[[DST]]]

; CHECK: add x[[SRC:[0-9]+]], {{x[0-9]+}}, :lo12:var+16
; CHECK: ldr [[BLOCK:q[0-9]+]], [x[[SRC]]]
; CHECK: str [[BLOCK]], [x[[DST]]]
  ret void
; CHECK: ret
}
