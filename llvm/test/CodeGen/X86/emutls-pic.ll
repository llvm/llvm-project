; RUN: llc < %s -emulated-tls -march=x86 -mtriple=i386-linux-gnu -relocation-model=pic | FileCheck -check-prefix=X32 %s
; RUN: llc < %s -emulated-tls -march=x86-64 -mtriple=x86_64-linux-gnu -relocation-model=pic | FileCheck -check-prefix=X64 %s
; RUN: llc < %s -emulated-tls -march=x86 -mtriple=i386-linux-android -relocation-model=pic | FileCheck -check-prefix=X32 %s
; RUN: llc < %s -emulated-tls -march=x86-64 -mtriple=x86_64-linux-android -relocation-model=pic | FileCheck -check-prefix=X64 %s

; Use my_emutls_get_address like __emutls_get_address.
@my_emutls_v_xyz = external global i8*, align 4
declare i8* @my_emutls_get_address(i8*)

define i32 @my_get_xyz() {
; X32-LABEL: my_get_xyz:
; X32:      movl my_emutls_v_xyz@GOT(%ebx), %eax
; X32-NEXT: movl %eax, (%esp)
; X32-NEXT: calll my_emutls_get_address@PLT
; X64-LABEL: my_get_xyz:
; X64:      movq my_emutls_v_xyz@GOTPCREL(%rip), %rdi
; X64-NEXT: callq my_emutls_get_address@PLT
; X64-NEXT: movl (%rax), %eax

entry:
  %call = call i8* @my_emutls_get_address(i8* bitcast (i8** @my_emutls_v_xyz to i8*))
  %0 = bitcast i8* %call to i32*
  %1 = load i32, i32* %0, align 4
  ret i32 %1
}

@i = thread_local global i32 15
@j = internal thread_local global i32 42
@k = internal thread_local global i32 0, align 8

define i32 @f1() {
entry:
  %tmp1 = load i32, i32* @i
  ret i32 %tmp1
}

; X32-LABEL: f1:
; X32:      movl __emutls_v.i@GOT(%ebx), %eax
; X32-NEXT: movl %eax, (%esp)
; X32-NEXT: calll __emutls_get_address@PLT
; X64-LABEL: f1:
; X64:      movq __emutls_v.i@GOTPCREL(%rip), %rdi
; X64-NEXT: callq __emutls_get_address@PLT
; X64-NEXT: movl (%rax), %eax

@i2 = external thread_local global i32

define i32* @f2() {
entry:
  ret i32* @i
}

; X32-LABEL: f2:
; X64-LABEL: f2:


define i32 @f3() {
entry:
  %tmp1 = load i32, i32* @i  ; <i32> [#uses=1]
  ret i32 %tmp1
}

; X32-LABEL: f3:
; X64-LABEL: f3:


define i32* @f4() nounwind {
entry:
  ret i32* @i
}

; X32-LABEL: f4:
; X64-LABEL: f4:


define i32 @f5() nounwind {
entry:
  %0 = load i32, i32* @j, align 4
  %1 = load i32, i32* @k, align 4
  %add = add nsw i32 %0, %1
  ret i32 %add
}

; X32-LABEL: f5:
; X32:      movl __emutls_v.j@GOT(%ebx), %eax
; X32-NEXT: movl %eax, (%esp)
; X32-NEXT: calll __emutls_get_address@PLT
; X32-NEXT: movl (%eax), %esi
; X32-NEXT: movl __emutls_v.k@GOT(%ebx), %eax
; X32-NEXT: movl %eax, (%esp)
; X32-NEXT: calll __emutls_get_address@PLT
; X32-NEXT: addl (%eax), %esi
; X32-NEXT: movl %esi, %eax

; X64-LABEL: f5:
; X64:      movq __emutls_v.j@GOTPCREL(%rip), %rdi
; X64-NEXT: callq __emutls_get_address@PLT
; X64-NEXT: movl (%rax), %ebx
; X64-NEXT: movq __emutls_v.k@GOTPCREL(%rip), %rdi
; X64-NEXT: callq __emutls_get_address@PLT
; X64-NEXT: addl (%rax), %ebx
; X64-NEXT: movl %ebx, %eax

;;;;; 32-bit targets

; X32:      .section .data.rel.local,
; X32-LABEL: __emutls_v.i:
; X32-NEXT: .long 4
; X32-NEXT: .long 4
; X32-NEXT: .long 0
; X32-NEXT: .long __emutls_t.i

; X32:      .section .rodata,
; X32-LABEL: __emutls_t.i:
; X32-NEXT: .long 15

; X32:      .section .data.rel.local,
; X32-LABEL: __emutls_v.j:
; X32-NEXT: .long 4
; X32-NEXT: .long 4
; X32-NEXT: .long 0
; X32-NEXT: .long __emutls_t.j

; X32:      .section .rodata,
; X32-LABEL: __emutls_t.j:
; X32-NEXT: .long 42

; X32:      .data
; X32-LABEL: __emutls_v.k:
; X32-NEXT: .long 4
; X32-NEXT: .long 8
; X32-NEXT: .long 0
; X32-NEXT: .long 0

; X32-NOT:   __emutls_t.k:

;;;;; 64-bit targets

; X64:      .section .data.rel.local,
; X64-LABEL: __emutls_v.i:
; X64-NEXT: .quad 4
; X64-NEXT: .quad 4
; X64-NEXT: .quad 0
; X64-NEXT: .quad __emutls_t.i

; X64:      .section .rodata,
; X64-LABEL: __emutls_t.i:
; X64-NEXT: .long 15

; X64:      .section .data.rel.local,
; X64-LABEL: __emutls_v.j:
; X64-NEXT: .quad 4
; X64-NEXT: .quad 4
; X64-NEXT: .quad 0
; X64-NEXT: .quad __emutls_t.j

; X64:      .section .rodata,
; X64-LABEL: __emutls_t.j:
; X64-NEXT: .long 42

; X64:      .data
; X64-LABEL: __emutls_v.k:
; X64-NEXT: .quad 4
; X64-NEXT: .quad 8
; X64-NEXT: .quad 0
; X64-NEXT: .quad 0

; X64-NOT:   __emutls_t.k:
