; RUN: llc -emulated-tls -mtriple=arm-linux-android \
; RUN:     -relocation-model=pic < %s | FileCheck -check-prefix=ARM32 %s
; RUN: llc -mtriple=arm-linux-android \
; RUN:     -relocation-model=pic < %s | FileCheck -check-prefix=ARM32 %s

; Copied from X86/emutls.ll

; Use my_emutls_get_address like __emutls_get_address.
@my_emutls_v_xyz = external global ptr, align 4
declare ptr @my_emutls_get_address(ptr)

define i32 @my_get_xyz() {
; ARM32-LABEL: my_get_xyz:
; ARM32:        ldr r0,
; ARM32:        ldr r0, [pc, r0]
; ARM32-NEXT:   bl my_emutls_get_address
; ARM32-NEXT:   ldr r0, [r0]
; ARM32:        .long my_emutls_v_xyz(GOT_PREL)

entry:
  %call = call ptr @my_emutls_get_address(ptr @my_emutls_v_xyz)
  %0 = load i32, ptr %call, align 4
  ret i32 %0
}

@i1 = thread_local global i32 15
@i2 = external thread_local global i32
@i3 = internal thread_local global i32 15
@i4 = hidden thread_local global i32 15
@i5 = external hidden thread_local global i32
@s1 = thread_local global i16 15
@b1 = thread_local global i8 0

define i32 @f1() {
; ARM32-LABEL: f1:
; ARM32:        ldr r0,
; ARM32:        ldr r0, [pc, r0]
; ARM32-NEXT:   bl __emutls_get_address
; ARM32-NEXT:   ldr r0, [r0]
; ARM32:        .long __emutls_v.i1(GOT_PREL)

entry:
  %tmp1 = load i32, ptr @i1
  ret i32 %tmp1
}

define ptr @f2() {
; ARM32-LABEL: f2:
; ARM32:        ldr r0,
; ARM32:        ldr r0, [pc, r0]
; ARM32-NEXT:   bl __emutls_get_address
; ARM32-NEXT:   pop
; ARM32:        .long __emutls_v.i1(GOT_PREL)

entry:
  ret ptr @i1
}

define i32 @f3() nounwind {
; ARM32-LABEL: f3:
; ARM32:        ldr r0,
; ARM32:        ldr r0, [pc, r0]
; ARM32-NEXT:   bl __emutls_get_address
; ARM32-NEXT:   ldr r0, [r0]
; ARM32:        .long __emutls_v.i2(GOT_PREL)

entry:
  %tmp1 = load i32, ptr @i2
  ret i32 %tmp1
}

define ptr @f4() {
; ARM32-LABEL: f4:
; ARM32:        ldr r0,
; ARM32:        ldr r0, [pc, r0]
; ARM32-NEXT:   bl __emutls_get_address
; ARM32-NEXT:   pop
; ARM32:        .long __emutls_v.i2(GOT_PREL)

entry:
  ret ptr @i2
}

define i32 @f5() nounwind {
; ARM32-LABEL: f5:
; ARM32:        ldr r0,
; ARM32:        add	r0, pc, r0
; ARM32-NEXT:   bl __emutls_get_address
; ARM32-NEXT:   ldr r0, [r0]
; ARM32:        .long __emutls_v.i3-

entry:
  %tmp1 = load i32, ptr @i3
  ret i32 %tmp1
}

define ptr @f6() {
; ARM32-LABEL: f6:
; ARM32:        ldr r0,
; ARM32:        add	r0, pc, r0
; ARM32-NEXT:   bl __emutls_get_address
; ARM32-NEXT:   pop
; ARM32:        .long __emutls_v.i3-

entry:
  ret ptr @i3
}

define i32 @f7() {
; ARM32-LABEL: f7:
; ARM32:        ldr r0,
; ARM32:        add r0, pc, r0
; ARM32-NEXT:   bl __emutls_get_address
; ARM32-NEXT:   ldr r0, [r0]
; ARM32:        .long __emutls_v.i4-(.LPC

entry:
  %tmp1 = load i32, ptr @i4
  ret i32 %tmp1
}

define ptr @f8() {
; ARM32-LABEL: f8:
; ARM32:        ldr r0,
; ARM32:        add r0, pc, r0
; ARM32-NEXT:   bl __emutls_get_address
; ARM32-NEXT:   pop
; ARM32:        .long __emutls_v.i4-(.LPC

entry:
  ret ptr @i4
}

define i32 @f9() {
; ARM32-LABEL: f9:
; ARM32:        ldr r0,
; ARM32:        add r0, pc, r0
; ARM32-NEXT:   bl __emutls_get_address
; ARM32-NEXT:   ldr r0, [r0]

entry:
  %tmp1 = load i32, ptr @i5
  ret i32 %tmp1
}

define ptr @f10() {
; ARM32-LABEL: f10:
; ARM32:        ldr r0,
; ARM32:        add r0, pc, r0
; ARM32-NEXT:   bl __emutls_get_address
; ARM32-NEXT:   pop

entry:
  ret ptr @i5
}

define i16 @f11() {
; ARM32-LABEL: f11:
; ARM32:        ldr r0,
; ARM32:        ldr r0, [pc, r0]
; ARM32-NEXT:   bl __emutls_get_address
; ARM32-NEXT:   ldrh r0, [r0]

entry:
  %tmp1 = load i16, ptr @s1
  ret i16 %tmp1
}

define i32 @f12() {
; ARM32-LABEL: f12:
; ARM32:        ldr r0,
; ARM32:        ldr r0, [pc, r0]
; ARM32-NEXT:   bl __emutls_get_address
; ARM32-NEXT:   ldrsh r0, [r0]

entry:
  %tmp1 = load i16, ptr @s1
  %tmp2 = sext i16 %tmp1 to i32
  ret i32 %tmp2
}

define i8 @f13() {
; ARM32-LABEL: f13:
; ARM32:        ldr r0,
; ARM32:        ldr r0, [pc, r0]
; ARM32-NEXT:   bl __emutls_get_address
; ARM32-NEXT:   ldrb r0, [r0]
; ARM32-NEXT: pop

entry:
  %tmp1 = load i8, ptr @b1
  ret i8 %tmp1
}

define i32 @f14() {
; ARM32-LABEL: f14:
; ARM32:        ldr r0,
; ARM32:        ldr r0, [pc, r0]
; ARM32-NEXT:   bl __emutls_get_address
; ARM32-NEXT:   ldrsb r0, [r0]
; ARM32-NEXT: pop

entry:
  %tmp1 = load i8, ptr @b1
  %tmp2 = sext i8 %tmp1 to i32
  ret i32 %tmp2
}

;;;;;;;;;;;;;; 32-bit __emutls_v. and __emutls_t.

; ARM32:      .data{{$}}
; ARM32:      .globl __emutls_v.i1
; ARM32-LABEL: __emutls_v.i1:
; ARM32-NEXT: .long 4
; ARM32-NEXT: .long 4
; ARM32-NEXT: .long 0
; ARM32-NEXT: .long __emutls_t.i1

; ARM32:      .section .rodata,
; ARM32-LABEL: __emutls_t.i1:
; ARM32-NEXT: .long 15

; ARM32-NOT:   __emutls_v.i2

; ARM32:      .data{{$}}
; ARM32-NOT:  .globl
; ARM32-LABEL: __emutls_v.i3:
; ARM32-NEXT: .long 4
; ARM32-NEXT: .long 4
; ARM32-NEXT: .long 0
; ARM32-NEXT: .long __emutls_t.i3

; ARM32:      .section .rodata,
; ARM32-LABEL: __emutls_t.i3:
; ARM32-NEXT: .long 15

; ARM32:      .data{{$}}
; ARM32:      .globl __emutls_v.i4
; ARM32-LABEL: __emutls_v.i4:
; ARM32-NEXT: .long 4
; ARM32-NEXT: .long 4
; ARM32-NEXT: .long 0
; ARM32-NEXT: .long __emutls_t.i4

; ARM32:      .section .rodata,
; ARM32-LABEL: __emutls_t.i4:
; ARM32-NEXT: .long 15

; ARM32-NOT:   __emutls_v.i5:
; ARM32:      .hidden __emutls_v.i5
; ARM32-NOT:   __emutls_v.i5:

; ARM32:      .data{{$}}
; ARM32:      .globl __emutls_v.s1
; ARM32-LABEL: __emutls_v.s1:
; ARM32-NEXT: .long 2
; ARM32-NEXT: .long 2
; ARM32-NEXT: .long 0
; ARM32-NEXT: .long __emutls_t.s1

; ARM32: .section .rodata,
; ARM32-LABEL: __emutls_t.s1:
; ARM32-NEXT: .short 15

; ARM32:      .data{{$}}
; ARM32:      .globl __emutls_v.b1
; ARM32-LABEL: __emutls_v.b1:
; ARM32-NEXT: .long 1
; ARM32-NEXT: .long 1
; ARM32-NEXT: .long 0
; ARM32-NEXT: .long 0

; ARM32-NOT:   __emutls_t.b1
