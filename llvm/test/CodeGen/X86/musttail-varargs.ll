; RUN: llc < %s -enable-tail-merge=0 -mtriple=x86_64-linux | FileCheck %s --check-prefix=LINUX
; RUN: llc < %s -enable-tail-merge=0 -mtriple=x86_64-linux-gnux32 | FileCheck %s --check-prefix=LINUX-X32
; RUN: llc < %s -enable-tail-merge=0 -mtriple=x86_64-windows | FileCheck %s --check-prefix=WINDOWS
; RUN: llc < %s -enable-tail-merge=0 -mtriple=i686-windows | FileCheck %s --check-prefix=X86
; RUN: llc < %s -enable-tail-merge=0 -mtriple=i686-windows -mattr=+sse2 | FileCheck %s --check-prefix=X86

; Test that we actually spill and reload all arguments in the variadic argument
; pack. Doing a normal call will clobber all argument registers, and we will
; spill around it. A simple adjustment should not require any XMM spills.

declare void @llvm.va_start(i8*) nounwind

declare void(i8*, ...)* @get_f(i8* %this)

define void @f_thunk(i8* %this, ...) {
  ; Use va_start so that we exercise the combination.
  %ap = alloca [4 x i8*], align 16
  %ap_i8 = bitcast [4 x i8*]* %ap to i8*
  call void @llvm.va_start(i8* %ap_i8)

  %fptr = call void(i8*, ...)*(i8*) @get_f(i8* %this)
  musttail call void (i8*, ...) %fptr(i8* %this, ...)
  ret void
}

; Save and restore 6 GPRs, 8 XMMs, and AL around the call.

; LINUX-LABEL: f_thunk:
; LINUX-DAG: movq %rdi, {{.*}}
; LINUX-DAG: movq %rsi, {{.*}}
; LINUX-DAG: movq %rdx, {{.*}}
; LINUX-DAG: movq %rcx, {{.*}}
; LINUX-DAG: movq %r8, {{.*}}
; LINUX-DAG: movq %r9, {{.*}}
; LINUX-DAG: movb %al, {{.*}}
; LINUX-DAG: movaps %xmm0, {{[0-9]*}}(%rsp)
; LINUX-DAG: movaps %xmm1, {{[0-9]*}}(%rsp)
; LINUX-DAG: movaps %xmm2, {{[0-9]*}}(%rsp)
; LINUX-DAG: movaps %xmm3, {{[0-9]*}}(%rsp)
; LINUX-DAG: movaps %xmm4, {{[0-9]*}}(%rsp)
; LINUX-DAG: movaps %xmm5, {{[0-9]*}}(%rsp)
; LINUX-DAG: movaps %xmm6, {{[0-9]*}}(%rsp)
; LINUX-DAG: movaps %xmm7, {{[0-9]*}}(%rsp)
; LINUX: callq get_f
; LINUX-DAG: movaps {{[0-9]*}}(%rsp), %xmm0
; LINUX-DAG: movaps {{[0-9]*}}(%rsp), %xmm1
; LINUX-DAG: movaps {{[0-9]*}}(%rsp), %xmm2
; LINUX-DAG: movaps {{[0-9]*}}(%rsp), %xmm3
; LINUX-DAG: movaps {{[0-9]*}}(%rsp), %xmm4
; LINUX-DAG: movaps {{[0-9]*}}(%rsp), %xmm5
; LINUX-DAG: movaps {{[0-9]*}}(%rsp), %xmm6
; LINUX-DAG: movaps {{[0-9]*}}(%rsp), %xmm7
; LINUX-DAG: movq {{.*}}, %rdi
; LINUX-DAG: movq {{.*}}, %rsi
; LINUX-DAG: movq {{.*}}, %rdx
; LINUX-DAG: movq {{.*}}, %rcx
; LINUX-DAG: movq {{.*}}, %r8
; LINUX-DAG: movq {{.*}}, %r9
; LINUX-DAG: movb {{.*}}, %al
; LINUX: jmpq *{{.*}}  # TAILCALL

; LINUX-X32-LABEL: f_thunk:
; LINUX-X32-DAG: movl %edi, {{.*}}
; LINUX-X32-DAG: movq %rsi, {{.*}}
; LINUX-X32-DAG: movq %rdx, {{.*}}
; LINUX-X32-DAG: movq %rcx, {{.*}}
; LINUX-X32-DAG: movq %r8, {{.*}}
; LINUX-X32-DAG: movq %r9, {{.*}}
; LINUX-X32-DAG: movb %al, {{.*}}
; LINUX-X32-DAG: movaps %xmm0, {{[0-9]*}}(%esp)
; LINUX-X32-DAG: movaps %xmm1, {{[0-9]*}}(%esp)
; LINUX-X32-DAG: movaps %xmm2, {{[0-9]*}}(%esp)
; LINUX-X32-DAG: movaps %xmm3, {{[0-9]*}}(%esp)
; LINUX-X32-DAG: movaps %xmm4, {{[0-9]*}}(%esp)
; LINUX-X32-DAG: movaps %xmm5, {{[0-9]*}}(%esp)
; LINUX-X32-DAG: movaps %xmm6, {{[0-9]*}}(%esp)
; LINUX-X32-DAG: movaps %xmm7, {{[0-9]*}}(%esp)
; LINUX-X32: callq get_f
; LINUX-X32-DAG: movaps {{[0-9]*}}(%esp), %xmm0
; LINUX-X32-DAG: movaps {{[0-9]*}}(%esp), %xmm1
; LINUX-X32-DAG: movaps {{[0-9]*}}(%esp), %xmm2
; LINUX-X32-DAG: movaps {{[0-9]*}}(%esp), %xmm3
; LINUX-X32-DAG: movaps {{[0-9]*}}(%esp), %xmm4
; LINUX-X32-DAG: movaps {{[0-9]*}}(%esp), %xmm5
; LINUX-X32-DAG: movaps {{[0-9]*}}(%esp), %xmm6
; LINUX-X32-DAG: movaps {{[0-9]*}}(%esp), %xmm7
; LINUX-X32-DAG: movl {{.*}}, %edi
; LINUX-X32-DAG: movq {{.*}}, %rsi
; LINUX-X32-DAG: movq {{.*}}, %rdx
; LINUX-X32-DAG: movq {{.*}}, %rcx
; LINUX-X32-DAG: movq {{.*}}, %r8
; LINUX-X32-DAG: movq {{.*}}, %r9
; LINUX-X32-DAG: movb {{.*}}, %al
; LINUX-X32: jmpq *{{.*}}  # TAILCALL

; WINDOWS-LABEL: f_thunk:
; WINDOWS-NOT: mov{{.}}ps
; WINDOWS-DAG: movq %rdx, {{.*}}
; WINDOWS-DAG: movq %rcx, {{.*}}
; WINDOWS-DAG: movq %r8, {{.*}}
; WINDOWS-DAG: movq %r9, {{.*}}
; WINDOWS-NOT: mov{{.}}ps
; WINDOWS: callq get_f
; WINDOWS-NOT: mov{{.}}ps
; WINDOWS-DAG: movq {{.*}}, %rdx
; WINDOWS-DAG: movq {{.*}}, %rcx
; WINDOWS-DAG: movq {{.*}}, %r8
; WINDOWS-DAG: movq {{.*}}, %r9
; WINDOWS-NOT: mov{{.}}ps
; WINDOWS: jmpq *{{.*}} # TAILCALL

; No regparms on normal x86 conventions.

; X86-LABEL: _f_thunk:
; X86: calll _get_f
; X86: jmpl *{{.*}} # TAILCALL

; This thunk shouldn't require any spills and reloads, assuming the register
; allocator knows what it's doing.

define void @g_thunk(i8* %fptr_i8, ...) {
  %fptr = bitcast i8* %fptr_i8 to void (i8*, ...)*
  musttail call void (i8*, ...) %fptr(i8* %fptr_i8, ...)
  ret void
}

; LINUX-LABEL: g_thunk:
; LINUX-NOT: movq
; LINUX: jmpq *%rdi  # TAILCALL

; LINUX-X32-LABEL: g_thunk:
; LINUX-X32-DAG: movl %edi, %[[REG:e[abcd]x|ebp|esi|edi|r8|r9|r1[0-5]]]
; LINUX-X32-DAG: jmpq *%[[REG]]  # TAILCALL

; WINDOWS-LABEL: g_thunk:
; WINDOWS-NOT: movq
; WINDOWS: jmpq *%rcx # TAILCALL

; X86-LABEL: _g_thunk:
; X86-NOT: push %ebp
; X86-NOT: andl {{.*}}, %esp
; X86: jmpl *%eax # TAILCALL

; Do a simple multi-exit multi-bb test.

%struct.Foo = type { i1, i8*, i8* }

@g = external global i32

define void @h_thunk(%struct.Foo* %this, ...) {
  %cond_p = getelementptr %struct.Foo, %struct.Foo* %this, i32 0, i32 0
  %cond = load i1, i1* %cond_p
  br i1 %cond, label %then, label %else

then:
  %a_p = getelementptr %struct.Foo, %struct.Foo* %this, i32 0, i32 1
  %a_i8 = load i8*, i8** %a_p
  %a = bitcast i8* %a_i8 to void (%struct.Foo*, ...)*
  musttail call void (%struct.Foo*, ...) %a(%struct.Foo* %this, ...)
  ret void

else:
  %b_p = getelementptr %struct.Foo, %struct.Foo* %this, i32 0, i32 2
  %b_i8 = load i8*, i8** %b_p
  %b = bitcast i8* %b_i8 to void (%struct.Foo*, ...)*
  store i32 42, i32* @g
  musttail call void (%struct.Foo*, ...) %b(%struct.Foo* %this, ...)
  ret void
}

; LINUX-LABEL: h_thunk:
; LINUX: jne
; LINUX: jmpq *{{.*}} # TAILCALL
; LINUX: jmpq *{{.*}} # TAILCALL
; LINUX-X32-LABEL: h_thunk:
; LINUX-X32: jne
; LINUX-X32: jmpq *{{.*}} # TAILCALL
; LINUX-X32: jmpq *{{.*}} # TAILCALL
; WINDOWS-LABEL: h_thunk:
; WINDOWS: jne
; WINDOWS: jmpq *{{.*}} # TAILCALL
; WINDOWS: jmpq *{{.*}} # TAILCALL
; X86-LABEL: _h_thunk:
; X86: jne
; X86: jmpl *{{.*}} # TAILCALL
; X86: jmpl *{{.*}} # TAILCALL
