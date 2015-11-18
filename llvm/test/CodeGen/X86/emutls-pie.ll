; RUN: llc < %s -emulated-tls -march=x86 -mcpu=generic -mtriple=i386-linux-gnu -relocation-model=pic -enable-pie \
; RUN:   | FileCheck -check-prefix=X32 %s
; RUN: llc < %s -emulated-tls -march=x86-64 -mcpu=generic -mtriple=x86_64-linux-gnu -relocation-model=pic -enable-pie \
; RUN:   | FileCheck -check-prefix=X64 %s
; RUN: llc < %s -emulated-tls -march=x86 -mcpu=generic -mtriple=i386-linux-android -relocation-model=pic -enable-pie \
; RUN:   | FileCheck -check-prefix=X32 %s
; RUN: llc < %s -emulated-tls -march=x86-64 -mcpu=generic -mtriple=x86_64-linux-android -relocation-model=pic -enable-pie \
; RUN:   | FileCheck -check-prefix=X64 %s

; Use my_emutls_get_address like __emutls_get_address.
@my_emutls_v_xyz = external global i8*, align 4
declare i8* @my_emutls_get_address(i8*)

define i32 @my_get_xyz() {
; X32-LABEL: my_get_xyz:
; X32:      movl my_emutls_v_xyz@GOT(%ebx), %eax
; X32-NEXT: movl %eax, (%esp)
; X32-NEXT: calll my_emutls_get_address@PLT
; X32-NEXT: movl (%eax), %eax
; X32-NEXT: addl $8, %esp
; X32-NEXT: popl %ebx
; X32-NEXT: retl
; X64-LABEL: my_get_xyz:
; X64:      movq my_emutls_v_xyz@GOTPCREL(%rip), %rdi
; X64-NEXT: callq my_emutls_get_address@PLT
; X64-NEXT: movl (%rax), %eax
; X64-NEXT: popq %rdx
; X64-NEXT: retq

entry:
  %call = call i8* @my_emutls_get_address(i8* bitcast (i8** @my_emutls_v_xyz to i8*))
  %0 = bitcast i8* %call to i32*
  %1 = load i32, i32* %0, align 4
  ret i32 %1
}

@i = thread_local global i32 15
@i2 = external thread_local global i32

define i32 @f1() {
; X32-LABEL: f1:
; X32:      movl __emutls_v.i@GOT(%ebx), %eax
; X32-NEXT: movl %eax, (%esp)
; X32-NEXT: calll __emutls_get_address@PLT
; X32-NEXT: movl (%eax), %eax
; X32-NEXT: addl $8, %esp
; X32-NEXT: popl %ebx
; X32-NEXT: retl
; X64-LABEL: f1:
; X64:      movq __emutls_v.i@GOTPCREL(%rip), %rdi
; X64-NEXT: callq __emutls_get_address@PLT
; X64-NEXT: movl (%rax), %eax
; X64-NEXT: popq %rdx
; X64-NEXT: retq

entry:
  %tmp1 = load i32, i32* @i
  ret i32 %tmp1
}

define i32* @f2() {
; X32-LABEL: f2:
; X32:      movl __emutls_v.i@GOT(%ebx), %eax
; X32-NEXT: movl %eax, (%esp)
; X32-NEXT: calll __emutls_get_address@PLT
; X64-LABEL: f2:
; X64:      movq __emutls_v.i@GOTPCREL(%rip), %rdi
; X64-NEXT: callq __emutls_get_address@PLT

entry:
  ret i32* @i
}

define i32 @f3() {
; X32-LABEL: f3:
; X32:      movl __emutls_v.i2@GOT(%ebx), %eax
; X32-NEXT: movl %eax, (%esp)
; X32-NEXT: calll __emutls_get_address@PLT
; X64-LABEL: f3:
; X64:      movq __emutls_v.i2@GOTPCREL(%rip), %rdi
; X64-NEXT: callq __emutls_get_address@PLT

entry:
  %tmp1 = load i32, i32* @i2
  ret i32 %tmp1
}

define i32* @f4() {
; X32-LABEL: f4:
; X32:      movl __emutls_v.i2@GOT(%ebx), %eax
; X32-NEXT: movl %eax, (%esp)
; X32-NEXT: calll __emutls_get_address@PLT
; X64-LABEL: f4:
; X64:      movq __emutls_v.i2@GOTPCREL(%rip), %rdi
; X64-NEXT: callq __emutls_get_address@PLT

entry:
  ret i32* @i2
}

;;;;; 32-bit targets

; X32:      .data
; X32-LABEL: __emutls_v.i:
; X32-NEXT: .long 4
; X32-NEXT: .long 4
; X32-NEXT: .long 0
; X32-NEXT: .long __emutls_t.i

; X32:      .section .rodata,
; X32-LABEL: __emutls_t.i:
; X32-NEXT: .long 15

; X32-NOT:   __emutls_v.i2
; X32-NOT:   __emutls_t.i2

;;;;; 64-bit targets

; X64:      .data
; X64-LABEL: __emutls_v.i:
; X64-NEXT: .quad 4
; X64-NEXT: .quad 4
; X64-NEXT: .quad 0
; X64-NEXT: .quad __emutls_t.i

; X64:      .section .rodata,
; X64-LABEL: __emutls_t.i:
; X64-NEXT: .long 15

; X64-NOT:   __emutls_v.i2
; X64-NOT:   __emutls_t.i2
