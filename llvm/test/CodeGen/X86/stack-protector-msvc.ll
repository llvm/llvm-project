; RUN: llc -mtriple=i386-pc-windows-msvc < %s -o - | FileCheck -check-prefix=MSVC-X86 %s
; RUN: llc -mtriple=x86_64-pc-windows-msvc < %s -o - | FileCheck -check-prefix=MSVC-X64 %s

; Make sure fastisel falls back and does something secure.
; RUN: llc -mtriple=i686-pc-windows-msvc -O0 < %s -o - | FileCheck -check-prefix=MSVC-X86-O0 %s
; RUN: llc -mtriple=x86_64-pc-windows-msvc -O0 < %s -o - | FileCheck -check-prefix=MSVC-X64-O0 %s

@"\01LC" = internal constant [11 x i8] c"buf == %s\0A\00"    ; <ptr> [#uses=1]

define void @test(ptr %a) nounwind ssp {
entry:
 %a_addr = alloca ptr    ; <ptr> [#uses=2]
 %buf = alloca [8 x i8]    ; <ptr> [#uses=2]
 store ptr %a, ptr %a_addr
 %0 = load ptr, ptr %a_addr, align 4    ; <ptr> [#uses=1]
 %1 = call ptr @strcpy(ptr %buf, ptr %0) nounwind   ; <ptr> [#uses=0]
 %2 = call i32 (ptr, ...) @printf(ptr @"\01LC", ptr %buf) nounwind    ; <i32> [#uses=0]
 br label %return

return:    ; preds = %entry
 ret void
}

; MSVC-X86-LABEL: _test:
; MSVC-X86: movl ___security_cookie, %[[REG1:[^ ]*]]
; MSVC-X86: xorl %esp, %[[REG1]]
; MSVC-X86: movl %[[REG1]], [[SLOT:[0-9]*]](%esp)
; MSVC-X86: calll _strcpy
; MSVC-X86: movl [[SLOT]](%esp), %ecx
; MSVC-X86: xorl %esp, %ecx
; MSVC-X86: calll @__security_check_cookie@4
; MSVC-X86: retl

; MSVC-X64-LABEL: test:
; MSVC-X64: movq __security_cookie(%rip), %[[REG1:[^ ]*]]
; MSVC-X64: xorq %rsp, %[[REG1]]
; MSVC-X64: movq %[[REG1]], [[SLOT:[0-9]*]](%rsp)
; MSVC-X64: callq strcpy
; MSVC-X64: movq [[SLOT]](%rsp), %rcx
; MSVC-X64: xorq %rsp, %rcx
; MSVC-X64: callq __security_check_cookie
; MSVC-X64: retq

; MSVC-X86-O0-LABEL: _test:
; MSVC-X86-O0: movl ___security_cookie, %[[REG1:[^ ]*]]
; MSVC-X86-O0: xorl %esp, %[[REG1]]
; MSVC-X86-O0: movl %[[REG1]], [[SLOT:[0-9]*]](%esp)
; MSVC-X86-O0: calll _strcpy
; MSVC-X86-O0: movl [[SLOT]](%esp), %ecx
; MSVC-X86-O0: xorl %esp, %ecx
; MSVC-X86-O0: calll @__security_check_cookie@4
; MSVC-X86-O0: retl

; MSVC-X64-O0-LABEL: test:
; MSVC-X64-O0: movq __security_cookie(%rip), %[[REG1:[^ ]*]]
; MSVC-X64-O0: xorq %rsp, %[[REG1]]
; MSVC-X64-O0: movq %[[REG1]], [[SLOT:[0-9]*]](%rsp)
; MSVC-X64-O0: callq strcpy
; MSVC-X64-O0: movq [[SLOT]](%rsp), %rcx
; MSVC-X64-O0: xorq %rsp, %rcx
; MSVC-X64-O0: callq __security_check_cookie
; MSVC-X64-O0: retq


declare void @escape(ptr)

define void @test_vla(i32 %n) nounwind ssp {
  %vla = alloca i32, i32 %n
  call void @escape(ptr %vla)
  ret void
}

; MSVC-X86-LABEL: _test_vla:
; MSVC-X86: pushl %ebp
; MSVC-X86: movl %esp, %ebp
; MSVC-X86: movl ___security_cookie, %[[REG1:[^ ]*]]
; MSVC-X86: xorl %ebp, %[[REG1]]
; MSVC-X86: movl %[[REG1]], [[SLOT:-[0-9]*]](%ebp)
; MSVC-X86: calll __chkstk
; MSVC-X86: pushl
; MSVC-X86: calll _escape
; MSVC-X86: movl [[SLOT]](%ebp), %ecx
; MSVC-X86: xorl %ebp, %ecx
; MSVC-X86: calll @__security_check_cookie@4
; MSVC-X86: movl %ebp, %esp
; MSVC-X86: popl %ebp
; MSVC-X86: retl

; MSVC-X64-LABEL: test_vla:
; MSVC-X64: pushq %rbp
; MSVC-X64: subq $16, %rsp
; MSVC-X64: leaq 16(%rsp), %rbp
; MSVC-X64: movq __security_cookie(%rip), %[[REG1:[^ ]*]]
; MSVC-X64: xorq %rbp, %[[REG1]]
; MSVC-X64: movq %[[REG1]], [[SLOT:-[0-9]*]](%rbp)
; MSVC-X64: callq __chkstk
; MSVC-X64: callq escape
; MSVC-X64: movq [[SLOT]](%rbp), %rcx
; MSVC-X64: xorq %rbp, %rcx
; MSVC-X64: callq __security_check_cookie
; MSVC-X64: retq


; This case is interesting because we address local variables with RBX but XOR
; the guard value with RBP. That's fine, either value will do, as long as they
; are the same across the life of the frame.

define void @test_vla_realign(i32 %n) nounwind ssp {
  %realign = alloca i32, align 32
  %vla = alloca i32, i32 %n
  call void @escape(ptr %realign)
  call void @escape(ptr %vla)
  ret void
}

; MSVC-X86-LABEL: _test_vla_realign:
; MSVC-X86: pushl %ebp
; MSVC-X86: movl %esp, %ebp
; MSVC-X86: pushl %esi
; MSVC-X86: andl $-32, %esp
; MSVC-X86: subl $32, %esp
; MSVC-X86: movl %esp, %esi
; MSVC-X86: movl ___security_cookie, %[[REG1:[^ ]*]]
; MSVC-X86: xorl %ebp, %[[REG1]]
; MSVC-X86: movl %[[REG1]], [[SLOT:[0-9]*]](%esi)
; MSVC-X86: calll __chkstk
; MSVC-X86: pushl
; MSVC-X86: calll _escape
; MSVC-X86: movl [[SLOT]](%esi), %ecx
; MSVC-X86: xorl %ebp, %ecx
; MSVC-X86: calll @__security_check_cookie@4
; MSVC-X86: leal -8(%ebp), %esp
; MSVC-X86: popl %esi
; MSVC-X86: popl %ebp
; MSVC-X86: retl

; MSVC-X64-LABEL: test_vla_realign:
; MSVC-X64: pushq %rbp
; MSVC-X64: pushq %rbx
; MSVC-X64: subq $32, %rsp
; MSVC-X64: leaq 32(%rsp), %rbp
; MSVC-X64: andq $-32, %rsp
; MSVC-X64: movq %rsp, %rbx
; MSVC-X64: movq __security_cookie(%rip), %[[REG1:[^ ]*]]
; MSVC-X64: xorq %rbp, %[[REG1]]
; MSVC-X64: movq %[[REG1]], [[SLOT:[0-9]*]](%rbx)
; MSVC-X64: callq __chkstk
; MSVC-X64: callq escape
; MSVC-X64: movq [[SLOT]](%rbx), %rcx
; MSVC-X64: xorq %rbp, %rcx
; MSVC-X64: callq __security_check_cookie
; MSVC-X64: retq


declare ptr @strcpy(ptr, ptr) nounwind

declare i32 @printf(ptr, ...) nounwind

