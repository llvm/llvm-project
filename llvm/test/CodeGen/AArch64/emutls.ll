; RUN: llc -emulated-tls -mtriple=aarch64-linux-android \
; RUN:     -relocation-model=pic < %s | FileCheck -check-prefix=ARM64 %s

; Copied from X86/emutls.ll

; Use my_emutls_get_address like __emutls_get_address.
@my_emutls_v_xyz = external global i8*, align 4
declare i8* @my_emutls_get_address(i8*)

define i32 @my_get_xyz() {
; ARM64-LABEL: my_get_xyz:
; ARM64:        adrp x0, :got:my_emutls_v_xyz
; ARM64-NEXT:   ldr x0, [x0, :got_lo12:my_emutls_v_xyz]
; ARM64-NEXT:   bl my_emutls_get_address
; ARM64-NEXT:   ldr  w0, [x0]
; ARM64-NEXT:   ldp x29, x30, [sp]

entry:
  %call = call i8* @my_emutls_get_address(i8* bitcast (i8** @my_emutls_v_xyz to i8*))
  %0 = bitcast i8* %call to i32*
  %1 = load i32, i32* %0, align 4
  ret i32 %1
}

@i1 = thread_local global i32 15
@i2 = external thread_local global i32
@i3 = internal thread_local global i32 15
@i4 = hidden thread_local global i32 15
@i5 = external hidden thread_local global i32
@s1 = thread_local global i16 15
@b1 = thread_local global i8 0

define i32 @f1() {
; ARM64-LABEL: f1:
; ARM64:        adrp x0, :got:__emutls_v.i1
; ARM64-NEXT:   ldr x0, [x0, :got_lo12:__emutls_v.i1]
; ARM64-NEXT:   bl __emutls_get_address
; ARM64-NEXT:   ldr  w0, [x0]
; ARM64-NEXT:   ldp x29, x30, [sp]

entry:
  %tmp1 = load i32, i32* @i1
  ret i32 %tmp1
}

define i32* @f2() {
; ARM64-LABEL: f2:
; ARM64:        adrp x0, :got:__emutls_v.i1
; ARM64-NEXT:   ldr x0, [x0, :got_lo12:__emutls_v.i1]
; ARM64-NEXT:   bl __emutls_get_address
; ARM64-NEXT:   ldp x29, x30, [sp]

entry:
  ret i32* @i1
}

;;;;;;;;;;;;;; 64-bit __emutls_v. and __emutls_t.

; ARM64       .section .data.rel.local,
; ARM64-LABEL: __emutls_v.i1:
; ARM64-NEXT: .xword 4
; ARM64-NEXT: .xword 4
; ARM64-NEXT: .xword 0
; ARM64-NEXT: .xword __emutls_t.i1

; ARM64       .section .rodata,
; ARM64-LABEL: __emutls_t.i1:
; ARM64-NEXT: .word 15

; ARM64-NOT:   __emutls_v.i2

; ARM64       .section .data.rel.local,
; ARM64-LABEL: __emutls_v.i3:
; ARM64-NEXT: .xword 4
; ARM64-NEXT: .xword 4
; ARM64-NEXT: .xword 0
; ARM64-NEXT: .xword __emutls_t.i3

; ARM64       .section .rodata,
; ARM64-LABEL: __emutls_t.i3:
; ARM64-NEXT: .word 15

; ARM64       .section .data.rel.local,
; ARM64-LABEL: __emutls_v.i4:
; ARM64-NEXT: .xword 4
; ARM64-NEXT: .xword 4
; ARM64-NEXT: .xword 0
; ARM64-NEXT: .xword __emutls_t.i4

; ARM64       .section .rodata,
; ARM64-LABEL: __emutls_t.i4:
; ARM64-NEXT: .word 15

; ARM64-NOT:   __emutls_v.i5:
; ARM64       .hidden __emutls_v.i5
; ARM64-NOT:   __emutls_v.i5:

; ARM64       .section .data.rel.local,
; ARM64-LABEL: __emutls_v.s1:
; ARM64-NEXT: .xword 2
; ARM64-NEXT: .xword 2
; ARM64-NEXT: .xword 0
; ARM64-NEXT: .xword __emutls_t.s1

; ARM64       .section .rodata,
; ARM64-LABEL: __emutls_t.s1:
; ARM64-NEXT: .hword 15

; ARM64       .section .data.rel.local,
; ARM64-LABEL: __emutls_v.b1:
; ARM64-NEXT: .xword 1
; ARM64-NEXT: .xword 1
; ARM64-NEXT: .xword 0
; ARM64-NEXT: .xword 0

; ARM64-NOT:  __emutls_t.b1
