; RUN: llc < %s -emulated-tls -mtriple=i386-linux-gnu | FileCheck -check-prefix=X32 %s
; RUN: llc < %s -emulated-tls -mtriple=x86_64-linux-gnu | FileCheck -check-prefix=X64 %s
; RUN: llc < %s -emulated-tls -mtriple=i386-linux-android | FileCheck -check-prefix=X32 %s
; RUN: llc < %s -emulated-tls -mtriple=x86_64-linux-android | FileCheck -check-prefix=X64 %s

; Copied from tls.ll; emulated TLS model is not implemented
; for *-pc-win32 and *-pc-winows targets yet.

; Use my_emutls_get_address like __emutls_get_address.
@my_emutls_v_xyz = external global i8*, align 4
declare i8* @my_emutls_get_address(i8*)

define i32 @my_get_xyz() {
; X32-LABEL: my_get_xyz:
; X32:         movl $my_emutls_v_xyz, (%esp)
; X32-NEXT:    calll my_emutls_get_address
; X32-NEXT:    movl (%eax), %eax
; X32-NEXT:    addl $12, %esp
; X32-NEXT:    retl
; X64-LABEL: my_get_xyz:
; X64:         movl $my_emutls_v_xyz, %edi
; X64-NEXT:    callq my_emutls_get_address
; X64-NEXT:    movl (%rax), %eax
; X64-NEXT:    popq %rcx
; X64-NEXT:    retq

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
; X32-LABEL: f1:
; X32:         movl $__emutls_v.i1, (%esp)
; X32-NEXT:    calll __emutls_get_address
; X32-NEXT:    movl (%eax), %eax
; X32-NEXT:    addl $12, %esp
; X32-NEXT:    retl
; X64-LABEL: f1:
; X64:         movl $__emutls_v.i1, %edi
; X64-NEXT:    callq __emutls_get_address
; X64-NEXT:    movl (%rax), %eax
; X64-NEXT:    popq %rcx
; X64-NEXT:    retq

entry:
  %tmp1 = load i32, i32* @i1
  ret i32 %tmp1
}

define i32* @f2() {
; X32-LABEL: f2:
; X32:         movl $__emutls_v.i1, (%esp)
; X32-NEXT:    calll __emutls_get_address
; X32-NEXT:    addl $12, %esp
; X32-NEXT:    retl
; X64-LABEL: f2:
; X64:         movl $__emutls_v.i1, %edi
; X64-NEXT:    callq __emutls_get_address
; X64-NEXT:    popq %rcx
; X64-NEXT:    retq

entry:
  ret i32* @i1
}

define i32 @f3() nounwind {
; X32-LABEL: f3:
; X32:         movl $__emutls_v.i2, (%esp)
; X32-NEXT:    calll __emutls_get_address
; X32-NEXT:    movl (%eax), %eax
; X32-NEXT:    addl $12, %esp
; X32-NEXT:    retl

entry:
  %tmp1 = load i32, i32* @i2
  ret i32 %tmp1
}

define i32* @f4() {
; X32-LABEL: f4:
; X32:         movl $__emutls_v.i2, (%esp)
; X32-NEXT:    calll __emutls_get_address
; X32-NEXT:    addl $12, %esp
; X32-NEXT:    retl

entry:
  ret i32* @i2
}

define i32 @f5() nounwind {
; X32-LABEL: f5:
; X32:         movl $__emutls_v.i3, (%esp)
; X32-NEXT:    calll __emutls_get_address
; X32-NEXT:    movl (%eax), %eax
; X32-NEXT:    addl $12, %esp
; X32-NEXT:    retl

entry:
  %tmp1 = load i32, i32* @i3
  ret i32 %tmp1
}

define i32* @f6() {
; X32-LABEL: f6:
; X32:         movl $__emutls_v.i3, (%esp)
; X32-NEXT:    calll __emutls_get_address
; X32-NEXT:    addl $12, %esp
; X32-NEXT:    retl

entry:
  ret i32* @i3
}

define i32 @f7() {
; X32-LABEL: f7:
; X32:         movl $__emutls_v.i4, (%esp)
; X32-NEXT:    calll __emutls_get_address
; X32-NEXT:    movl (%eax), %eax
; X32-NEXT:    addl $12, %esp
; X32-NEXT:    retl

entry:
  %tmp1 = load i32, i32* @i4
  ret i32 %tmp1
}

define i32* @f8() {
; X32-LABEL: f8:
; X32:         movl $__emutls_v.i4, (%esp)
; X32-NEXT:    calll __emutls_get_address
; X32-NEXT:    addl $12, %esp
; X32-NEXT:    retl

entry:
  ret i32* @i4
}

define i32 @f9() {
; X32-LABEL: f9:
; X32:         movl $__emutls_v.i5, (%esp)
; X32-NEXT:    calll __emutls_get_address
; X32-NEXT:    movl (%eax), %eax
; X32-NEXT:    addl $12, %esp
; X32-NEXT:    retl

entry:
  %tmp1 = load i32, i32* @i5
  ret i32 %tmp1
}

define i32* @f10() {
; X32-LABEL: f10:
; X32:         movl $__emutls_v.i5, (%esp)
; X32-NEXT:    calll __emutls_get_address
; X32-NEXT:    addl $12, %esp
; X32-NEXT:    retl

entry:
  ret i32* @i5
}

define i16 @f11() {
; X32-LABEL: f11:
; X32:         movl $__emutls_v.s1, (%esp)
; X32-NEXT:    calll __emutls_get_address
; X32-NEXT:    movzwl (%eax), %eax
; X32-NEXT:    addl $12, %esp
; X32-NEXT:    retl

entry:
  %tmp1 = load i16, i16* @s1
  ret i16 %tmp1
}

define i32 @f12() {
; X32-LABEL: f12:
; X32:         movl $__emutls_v.s1, (%esp)
; X32-NEXT:    calll __emutls_get_address
; X32-NEXT:    movswl (%eax), %eax
; X32-NEXT:    addl $12, %esp
; X32-NEXT:    retl

entry:
  %tmp1 = load i16, i16* @s1
  %tmp2 = sext i16 %tmp1 to i32
  ret i32 %tmp2
}

define i8 @f13() {
; X32-LABEL: f13:
; X32:         movl $__emutls_v.b1, (%esp)
; X32-NEXT:    calll __emutls_get_address
; X32-NEXT:    movb (%eax), %al
; X32-NEXT:    addl $12, %esp
; X32-NEXT:    retl

entry:
  %tmp1 = load i8, i8* @b1
  ret i8 %tmp1
}

define i32 @f14() {
; X32-LABEL: f14:
; X32:         movl $__emutls_v.b1, (%esp)
; X32-NEXT:    calll __emutls_get_address
; X32-NEXT:    movsbl (%eax), %eax
; X32-NEXT:    addl $12, %esp
; X32-NEXT:    retl

entry:
  %tmp1 = load i8, i8* @b1
  %tmp2 = sext i8 %tmp1 to i32
  ret i32 %tmp2
}

;;;;;;;;;;;;;; 32-bit __emutls_v. and __emutls_t.

; X32       .section .data.rel.local,
; X32-LABEL: __emutls_v.i1:
; X32-NEXT: .long 4
; X32-NEXT: .long 4
; X32-NEXT: .long 0
; X32-NEXT: .long __emutls_t.i1

; X32       .section .rodata,
; X32-LABEL: __emutls_t.i1:
; X32-NEXT: .long 15

; X32-NOT:   __emutls_v.i2

; X32       .section .data.rel.local,
; X32-LABEL: __emutls_v.i3:
; X32-NEXT: .long 4
; X32-NEXT: .long 4
; X32-NEXT: .long 0
; X32-NEXT: .long __emutls_t.i3

; X32       .section .rodata,
; X32-LABEL: __emutls_t.i3:
; X32-NEXT: .long 15

; X32       .section .data.rel.local,
; X32-LABEL: __emutls_v.i4:
; X32-NEXT: .long 4
; X32-NEXT: .long 4
; X32-NEXT: .long 0
; X32-NEXT: .long __emutls_t.i4

; X32       .section .rodata,
; X32-LABEL: __emutls_t.i4:
; X32-NEXT: .long 15

; X32-NOT:   __emutls_v.i5:
; X32       .hidden __emutls_v.i5
; X32-NOT:   __emutls_v.i5:

; X32 .section .data.rel.local,
; X32-LABEL: __emutls_v.s1:
; X32-NEXT: .long 2
; X32-NEXT: .long 2
; X32-NEXT: .long 0
; X32-NEXT: .long __emutls_t.s1

; X32 .section .rodata,
; X32-LABEL: __emutls_t.s1:
; X32-NEXT: .short 15

; X32 .section .data.rel.local,
; X32-LABEL: __emutls_v.b1:
; X32-NEXT: .long 1
; X32-NEXT: .long 1
; X32-NEXT: .long 0
; X32-NEXT: .long 0

; X32-NOT:   __emutls_t.b1

;;;;;;;;;;;;;; 64-bit __emutls_v. and __emutls_t.

; X64       .section .data.rel.local,
; X64-LABEL: __emutls_v.i1:
; X64-NEXT: .quad 4
; X64-NEXT: .quad 4
; X64-NEXT: .quad 0
; X64-NEXT: .quad __emutls_t.i1

; X64       .section .rodata,
; X64-LABEL: __emutls_t.i1:
; X64-NEXT: .long 15

; X64-NOT:   __emutls_v.i2

; X64       .section .data.rel.local,
; X64-LABEL: __emutls_v.i3:
; X64-NEXT: .quad 4
; X64-NEXT: .quad 4
; X64-NEXT: .quad 0
; X64-NEXT: .quad __emutls_t.i3

; X64       .section .rodata,
; X64-LABEL: __emutls_t.i3:
; X64-NEXT: .long 15

; X64       .section .data.rel.local,
; X64-LABEL: __emutls_v.i4:
; X64-NEXT: .quad 4
; X64-NEXT: .quad 4
; X64-NEXT: .quad 0
; X64-NEXT: .quad __emutls_t.i4

; X64       .section .rodata,
; X64-LABEL: __emutls_t.i4:
; X64-NEXT: .long 15

; X64-NOT:   __emutls_v.i5:
; X64       .hidden __emutls_v.i5
; X64-NOT:   __emutls_v.i5:

; X64       .section .data.rel.local,
; X64-LABEL: __emutls_v.s1:
; X64-NEXT: .quad 2
; X64-NEXT: .quad 2
; X64-NEXT: .quad 0
; X64-NEXT: .quad __emutls_t.s1

; X64       .section .rodata,
; X64-LABEL: __emutls_t.s1:
; X64-NEXT: .short 15

; X64       .section .data.rel.local,
; X64-LABEL: __emutls_v.b1:
; X64-NEXT: .quad 1
; X64-NEXT: .quad 1
; X64-NEXT: .quad 0
; X64-NEXT: .quad 0

; X64-NOT:  __emutls_t.b1
