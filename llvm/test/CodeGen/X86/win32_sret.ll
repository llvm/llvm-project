; We specify -mcpu explicitly to avoid instruction reordering that happens on
; some setups (e.g., Atom) from affecting the output.
; RUN: llc < %s -mcpu=core2 -mtriple=i686-pc-win32 | FileCheck %s -check-prefix=WIN32
; RUN: llc < %s -mcpu=core2 -mtriple=i686-pc-mingw32 | FileCheck %s -check-prefix=MINGW_X86
; RUN: llc < %s -mcpu=core2 -mtriple=i686-pc-cygwin | FileCheck %s -check-prefix=CYGWIN
; RUN: llc < %s -mcpu=core2 -mtriple=i386-pc-linux | FileCheck %s -check-prefix=LINUX
; RUN: llc < %s -mcpu=core2 -O0 -mtriple=i686-pc-win32 | FileCheck %s -check-prefix=WIN32
; RUN: llc < %s -mcpu=core2 -O0 -mtriple=i686-pc-mingw32 | FileCheck %s -check-prefix=MINGW_X86
; RUN: llc < %s -mcpu=core2 -O0 -mtriple=i686-pc-cygwin | FileCheck %s -check-prefix=CYGWIN
; RUN: llc < %s -mcpu=core2 -O0 -mtriple=i386-pc-linux | FileCheck %s -check-prefix=LINUX

; The SysV ABI used by most Unixes and Mingw on x86 specifies that an sret pointer
; is callee-cleanup. However, in MSVC's cdecl calling convention, sret pointer
; arguments are caller-cleanup like normal arguments.

define void @sret1(ptr sret(i8) %x) nounwind {
entry:
; WIN32-LABEL:      _sret1:
; WIN32:      movb $42, ({{%e[abcd]x}})
; WIN32-NOT:  popl %eax
; WIN32:    {{retl$}}

; MINGW_X86-LABEL:  _sret1:
; MINGW_X86:  {{retl$}}

; CYGWIN-LABEL:     _sret1:
; CYGWIN:     retl $4

; LINUX-LABEL:      sret1:
; LINUX:      retl $4

  store i8 42, ptr %x, align 4
  ret void
}

define void @sret2(ptr sret(i8) %x, i8 %y) nounwind {
entry:
; WIN32-LABEL:      _sret2:
; WIN32:      movb {{.*}}, ({{%e[abcd]x}})
; WIN32-NOT:  popl %eax
; WIN32:    {{retl$}}

; MINGW_X86-LABEL:  _sret2:
; MINGW_X86:  {{retl$}}

; CYGWIN-LABEL:     _sret2:
; CYGWIN:     retl $4

; LINUX-LABEL:      sret2:
; LINUX:      retl $4

  store i8 %y, ptr %x
  ret void
}

define void @sret3(ptr sret(i8) %x, ptr %y) nounwind {
entry:
; WIN32-LABEL:      _sret3:
; WIN32:      movb $42, ([[REG1:%e[abcd]x]])
; WIN32-NOT:  movb $13, ([[REG1]])
; WIN32-NOT:  popl %eax
; WIN32:    {{retl$}}

; MINGW_X86-LABEL:  _sret3:
; MINGW_X86:  {{retl$}}

; CYGWIN-LABEL:     _sret3:
; CYGWIN:     retl $4

; LINUX-LABEL:      sret3:
; LINUX:      retl $4

  store i8 42, ptr %x
  store i8 13, ptr %y
  ret void
}

; PR15556
%struct.S4 = type { i32, i32, i32 }

define void @sret4(ptr noalias sret(%struct.S4) %agg.result) {
entry:
; WIN32-LABEL:     _sret4:
; WIN32:     movl $42, ({{%e[abcd]x}})
; WIN32-NOT: popl %eax
; WIN32:   {{retl$}}

; MINGW_X86-LABEL: _sret4:
; MINGW_X86: {{retl$}}

; CYGWIN-LABEL:    _sret4:
; CYGWIN:    retl $4

; LINUX-LABEL:     sret4:
; LINUX:     retl $4

  store i32 42, ptr %agg.result, align 4
  ret void
}

%struct.S5 = type { i32 }
%class.C5 = type { i8 }

define x86_thiscallcc void @"\01?foo@C5@@QAE?AUS5@@XZ"(ptr noalias sret(%struct.S5) %agg.result, ptr %this) {
entry:
  %this.addr = alloca ptr, align 4
  store ptr %this, ptr %this.addr, align 4
  %this1 = load ptr, ptr %this.addr
  store i32 42, ptr %agg.result, align 4
  ret void
; WIN32-LABEL:     {{^}}"?foo@C5@@QAE?AUS5@@XZ":
; MINGW_X86-LABEL: {{^}}"?foo@C5@@QAE?AUS5@@XZ":
; CYGWIN-LABEL:    {{^}}"?foo@C5@@QAE?AUS5@@XZ":
; LINUX-LABEL:     {{^}}"?foo@C5@@QAE?AUS5@@XZ":

; The address of the return structure is passed as an implicit parameter.
; In the -O0 build, %eax is spilled at the beginning of the function, hence we
; should match both 4(%esp) and 8(%esp).
; WIN32:     {{[48]}}(%esp), [[REG:%e[abcd]x]]
; WIN32:     movl $42, ([[REG]])
; WIN32:     retl $4
}

define void @call_foo5() {
entry:
  %c = alloca %class.C5, align 1
  %s = alloca %struct.S5, align 4
  call x86_thiscallcc void @"\01?foo@C5@@QAE?AUS5@@XZ"(ptr sret(%struct.S5) %s, ptr %c)
; WIN32-LABEL:      {{^}}_call_foo5:
; MINGW_X86-LABEL:  {{^}}_call_foo5:
; CYGWIN-LABEL:     {{^}}_call_foo5:
; LINUX-LABEL:      {{^}}call_foo5:


; Load the address of the result and put it onto stack
; The this pointer goes to ECX.
; (through %ecx in the -O0 build).
; WIN32-DAG:  leal {{[0-9]*}}(%esp), %e{{[a-d]}}x
; WIN32-DAG:  {{leal [1-9]+\(%esp\)|movl %esp}}, %ecx
; WIN32-DAG:  {{pushl %e[a-d]x|movl %e[a-d]x, \(%esp\)}}
; WIN32-NEXT: calll "?foo@C5@@QAE?AUS5@@XZ"
; WIN32:      retl
  ret void
}


%struct.test6 = type { i32, i32, i32 }
define void @test6_f(ptr %x) nounwind {
; WIN32-LABEL: _test6_f:
; MINGW_X86-LABEL: _test6_f:
; CYGWIN-LABEL: _test6_f:
; LINUX-LABEL: test6_f:

; The %x argument is moved to %ecx. It will be the this pointer.
; WIN32-DAG: movl    {{16|20}}(%esp), %ecx


; The sret pointer is (%esp)
; WIN32-DAG:      {{leal 4\(%esp\)|movl %esp}}, %eax
; WIN32-DAG:      {{pushl   %eax|movl %eax, \(%esp\)}}

; The sret pointer is %ecx
; The %x argument is moved to (%esp). It will be the this pointer.
; MINGW_X86-DAG:  {{leal 4\(%esp\)|movl %esp}}, %ecx
; MINGW_X86-DAG: {{pushl   16\(%esp\)|movl %eax, \(%esp\)}}
; MINGW_X86-NEXT: calll   _test6_g

; CYGWIN-DAG:  {{leal 4\(%esp\)|movl %esp}}, %ecx
; CYGWIN-DAG:  {{pushl   16\(%esp\)|movl %eax, \(%esp\)}}
; CYGWIN-NEXT: calll   _test6_g

  %tmp = alloca %struct.test6, align 4
  call x86_thiscallcc void @test6_g(ptr sret(%struct.test6) %tmp, ptr %x)
  ret void
}
declare x86_thiscallcc void @test6_g(ptr sret(%struct.test6), ptr)

; Flipping the parameters at the IR level generates the same code.
%struct.test7 = type { i32, i32, i32 }
define void @test7_f(ptr %x) nounwind {
; WIN32-LABEL: _test7_f:
; MINGW_X86-LABEL: _test7_f:
; CYGWIN-LABEL: _test7_f:
; LINUX-LABEL: test7_f:

; The %x argument is moved to %ecx on all OSs. It will be the this pointer.
; WIN32:      movl    {{16|20}}(%esp), %ecx
; MINGW_X86:  movl    {{16|20}}(%esp), %ecx
; CYGWIN:     movl    {{16|20}}(%esp), %ecx

; The sret pointer is (%esp)
; WIN32:      {{leal 4\(%esp\)|movl %esp}}, %eax
; WIN32-NEXT:     {{pushl   %eax|movl %eax, \(%esp\)}}
; MINGW_X86:      {{leal 4\(%esp\)|movl %esp}}, %eax
; MINGW_X86-NEXT: {{pushl   %eax|movl %eax, \(%esp\)}}
; CYGWIN:      {{leal 4\(%esp\)|movl %esp}}, %eax
; CYGWIN-NEXT: {{pushl   %eax|movl %eax, \(%esp\)}}

  %tmp = alloca %struct.test7, align 4
  call x86_thiscallcc void @test7_g(ptr %x, ptr sret(%struct.test7) %tmp)
  ret void
}

define x86_thiscallcc void @test7_g(ptr %in, ptr sret(%struct.test7) %out) {
  %v = load i32, ptr %in
  store i32 %v, ptr %out
  call void @clobber_eax()
  ret void

; Make sure we return the second parameter in %eax.
; WIN32-LABEL: _test7_g:
; WIN32: calll _clobber_eax
; WIN32: movl {{.*}}, %eax
; WIN32: retl
}

declare void @clobber_eax()

; Test what happens if the first parameter has to be split by codegen.
; Realistically, no frontend will generate code like this, but here it is for
; completeness.
define void @test8_f(i64 inreg %a, ptr sret(i64) %out) {
  store i64 %a, ptr %out
  call void @clobber_eax()
  ret void

; WIN32-LABEL: _test8_f:
; WIN32: movl {{[0-9]+}}(%esp), %[[out:[a-z]+]]
; WIN32-DAG: movl {{%e[abcd]x}}, 4(%[[out]])
; WIN32-DAG: movl {{%e[abcd]x}}, (%[[out]])
; WIN32: calll _clobber_eax
; WIN32: movl {{.*}}, %eax
; WIN32: retl
}
