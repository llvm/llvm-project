; RUN: llc -emulated-tls -mtriple=aarch64-linux-android \
; RUN:     -relocation-model=pic -frame-pointer=all < %s | FileCheck -check-prefix=ARM64 %s
; RUN: llc -mtriple=aarch64-linux-android \
; RUN:     -relocation-model=pic -frame-pointer=all < %s | FileCheck -check-prefix=ARM64 %s

; Copied from X86/emutls.ll

; Use my_emutls_get_address like __emutls_get_address.
@my_emutls_v_xyz = external global ptr, align 4
declare ptr @my_emutls_get_address(ptr)

define i32 @my_get_xyz() uwtable {
; ARM64-LABEL: my_get_xyz:
; ARM64:        adrp x0, :got:my_emutls_v_xyz
; ARM64-NEXT:   ldr x0, [x0, :got_lo12:my_emutls_v_xyz]
; ARM64-NEXT:   bl my_emutls_get_address
; ARM64-NEXT:   ldr  w0, [x0]
; ARM64-NEXT: 	.cfi_def_cfa wsp, 16
; ARM64-NEXT:   ldp x29, x30, [sp]

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

define i32 @f1() uwtable {
; ARM64-LABEL: f1:
; ARM64:        adrp x0, :got:__emutls_v.i1
; ARM64-NEXT:   ldr x0, [x0, :got_lo12:__emutls_v.i1]
; ARM64-NEXT:   bl __emutls_get_address
; ARM64-NEXT:   ldr  w0, [x0]
; ARM64-NEXT: 	.cfi_def_cfa wsp, 16
; ARM64-NEXT:   ldp x29, x30, [sp]

entry:
  %tmp1 = load i32, ptr @i1
  ret i32 %tmp1
}

define ptr @f2() uwtable {
; ARM64-LABEL: f2:
; ARM64:        adrp x0, :got:__emutls_v.i1
; ARM64-NEXT:   ldr x0, [x0, :got_lo12:__emutls_v.i1]
; ARM64-NEXT:   bl __emutls_get_address
; ARM64-NEXT: 	.cfi_def_cfa wsp, 16
; ARM64-NEXT:   ldp x29, x30, [sp]

entry:
  ret ptr @i1
}

define i32 @f5() nounwind {
; ARM64-LABEL: f5:
; ARM64:        adrp x0, __emutls_v.i3
; ARM64:        add x0, x0, :lo12:__emutls_v.i3
; ARM64:        bl __emutls_get_address
; ARM64-NEXT:   ldr w0, [x0]

entry:
  %tmp1 = load i32, ptr @i3
  ret i32 %tmp1
}

define ptr @f6() uwtable {
; ARM64-LABEL: f6:
; ARM64:        adrp x0, __emutls_v.i3
; ARM64:        add x0, x0, :lo12:__emutls_v.i3
; ARM64-NEXT:   bl __emutls_get_address
; ARM64-NEXT: 	.cfi_def_cfa wsp, 16
; ARM64-NEXT:   ldp x29, x30, [sp]

entry:
  ret ptr @i3
}

; Simple test of comdat __thread variables.
; template <class T> struct A { static __thread T x; };
; template <class T> T __thread A<T>::x;
; int getIntX() { return A<int>::x++; }
; float getFloatX() { return A<float>::x++; }

$_ZN1AIiE1xE = comdat any
$_ZN1AIfE1xE = comdat any
@_ZN1AIiE1xE = linkonce_odr thread_local global i32 0, comdat, align 4
@_ZN1AIfE1xE = linkonce_odr thread_local global float 0.000000e+00, comdat, align 4

define i32 @_Z7getIntXv() {
; ARM64-LABEL: _Z7getIntXv:
; ARM64:        adrp x0, :got:__emutls_v._ZN1AIiE1xE
; ARM64:        ldr x0, [x0, :got_lo12:__emutls_v._ZN1AIiE1xE]
; ARM64-NEXT:   bl __emutls_get_address
; ARM64:        ldr {{.*}}, [x0]
; ARM64:        add
; ARM64:        str {{.*}}, [x8]

entry:
  %0 = load i32, ptr @_ZN1AIiE1xE, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, ptr @_ZN1AIiE1xE, align 4
  ret i32 %0
}

define float @_Z9getFloatXv() {
; ARM64-LABEL: _Z9getFloatXv:
; ARM64:        adrp x0, :got:__emutls_v._ZN1AIfE1xE
; ARM64:        ldr x0, [x0, :got_lo12:__emutls_v._ZN1AIfE1xE]
; ARM64-NEXT:   bl __emutls_get_address
; ARM64:        ldr {{.*}}, [x0]
; ARM64:        fadd s{{.*}}, s
; ARM64:        str s{{.*}}, [x0]

entry:
  %0 = load float, ptr @_ZN1AIfE1xE, align 4
  %inc = fadd float %0, 1.000000e+00
  store float %inc, ptr @_ZN1AIfE1xE, align 4
  ret float %0
}


;;;;;;;;;;;;;; 64-bit __emutls_v. and __emutls_t.

; ARM64:      .data{{$}}
; ARM64:      .globl __emutls_v.i1
; ARM64-LABEL: __emutls_v.i1:
; ARM64-NEXT: .xword 4
; ARM64-NEXT: .xword 4
; ARM64-NEXT: .xword 0
; ARM64-NEXT: .xword __emutls_t.i1

; ARM64:      .section .rodata,
; ARM64-LABEL: __emutls_t.i1:
; ARM64-NEXT: .word 15

; ARM64-NOT:   __emutls_v.i2

; ARM64:      .data{{$}}
; ARM64-NOT:  .globl
; ARM64-LABEL: __emutls_v.i3:
; ARM64-NEXT: .xword 4
; ARM64-NEXT: .xword 4
; ARM64-NEXT: .xword 0
; ARM64-NEXT: .xword __emutls_t.i3

; ARM64:      .section .rodata,
; ARM64-LABEL: __emutls_t.i3:
; ARM64-NEXT: .word 15

; ARM64:      .hidden __emutls_v.i4
; ARM64:      .data{{$}}
; ARM64:      .globl __emutls_v.i4
; ARM64-LABEL: __emutls_v.i4:
; ARM64-NEXT: .xword 4
; ARM64-NEXT: .xword 4
; ARM64-NEXT: .xword 0
; ARM64-NEXT: .xword __emutls_t.i4

; ARM64:      .section .rodata,
; ARM64-LABEL: __emutls_t.i4:
; ARM64-NEXT: .word 15

; ARM64-NOT:   __emutls_v.i5:
; ARM64:      .hidden __emutls_v.i5
; ARM64-NOT:   __emutls_v.i5:

; ARM64:      .data{{$}}
; ARM64:      .globl __emutls_v.s1
; ARM64-LABEL: __emutls_v.s1:
; ARM64-NEXT: .xword 2
; ARM64-NEXT: .xword 2
; ARM64-NEXT: .xword 0
; ARM64-NEXT: .xword __emutls_t.s1

; ARM64:      .section .rodata,
; ARM64-LABEL: __emutls_t.s1:
; ARM64-NEXT: .hword 15

; ARM64:      .data{{$}}
; ARM64-LABEL: __emutls_v.b1:
; ARM64-NEXT: .xword 1
; ARM64-NEXT: .xword 1
; ARM64-NEXT: .xword 0
; ARM64-NEXT: .xword 0

; ARM64-NOT:   __emutls_t.b1

; ARM64:      .section .data.__emutls_v._ZN1AIiE1xE,{{.*}},__emutls_v._ZN1AIiE1xE,comdat
; ARM64:      .weak __emutls_v._ZN1AIiE1xE
; ARM64:      .p2align 3
; ARM64-LABEL: __emutls_v._ZN1AIiE1xE:
; ARM64-NEXT: .xword 4
; ARM64-NEXT: .xword 4
; ARM64-NEXT: .xword 0
; ARM64-NEXT: .xword 0

; ARM64:      .section .data.__emutls_v._ZN1AIfE1xE,{{.*}},__emutls_v._ZN1AIfE1xE,comdat
; ARM64:      .weak __emutls_v._ZN1AIfE1xE
; ARM64:      .p2align 3
; ARM64-LABEL: __emutls_v._ZN1AIfE1xE:
; ARM64-NEXT: .xword 4
; ARM64-NEXT: .xword 4
; ARM64-NEXT: .xword 0
; ARM64-NEXT: .xword __emutls_t._ZN1AIfE1xE

; ARM64:      .section .rodata.__emutls_t._ZN1AIfE1xE,{{.*}},__emutls_t._ZN1AIfE1xE,comdat
; ARM64:      .weak __emutls_t._ZN1AIfE1xE
; ARM64:      .p2align 2
; ARM64-LABEL: __emutls_t._ZN1AIfE1xE:
; ARM64-NEXT: .word 0
; ARM64-NEXT: .size
