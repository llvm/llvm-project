; RUN: llc < %s -mtriple=i386-pc-win32       -mattr=+avx512bw  | FileCheck --check-prefix=X32 %s
; RUN: llc < %s -mtriple=x86_64-win32        -mattr=+avx512bw  | FileCheck --check-prefix=WIN64 %s
; RUN: llc < %s -mtriple=x86_64-linux-gnu    -mattr=+avx512bw  | FileCheck --check-prefix=LINUXOSX64 %s

; X32-LABEL:  test_argv64i1:
; X32:        kmovd   %edx, %k0
; X32:        kmovd   %edi, %k1
; X32:        kmovd   %eax, %k1
; X32:        kmovd   %ecx, %k2
; X32:        ad{{d|c}}l  {{([0-9])*}}(%ebp), %e{{a|c}}x
; X32:        ad{{d|c}}l  {{([0-9])*}}(%ebp), %e{{a|c}}x
; X32:        ad{{d|c}}l  {{([0-9])*}}(%ebp), %e{{a|c}}x
; X32:        ad{{d|c}}l  {{([0-9])*}}(%ebp), %e{{a|c}}x
; X32:        ad{{d|c}}l  {{([0-9])*}}(%ebp), %e{{a|c}}x
; X32:        ad{{d|c}}l  {{([0-9])*}}(%ebp), %e{{a|c}}x
; X32:        ad{{d|c}}l  {{([0-9])*}}(%ebp), %e{{a|c}}x
; X32:        ad{{d|c}}l  {{([0-9])*}}(%ebp), %e{{a|c}}x
; X32:        ad{{d|c}}l  {{([0-9])*}}(%ebp), %e{{a|c}}x
; X32:        ad{{d|c}}l  {{([0-9])*}}(%ebp), %e{{a|c}}x
; X32:        ad{{d|c}}l  {{([0-9])*}}(%ebp), %e{{a|c}}x
; X32:        ad{{d|c}}l  {{([0-9])*}}(%ebp), %e{{a|c}}x
; X32:        ad{{d|c}}l  {{([0-9])*}}(%ebp), %e{{a|c}}x
; X32:        ad{{d|c}}l  {{([0-9])*}}(%ebp), %e{{a|c}}x
; X32:        ad{{d|c}}l  {{([0-9])*}}(%ebp), %e{{a|c}}x
; X32:        ad{{d|c}}l  {{([0-9])*}}(%ebp), %e{{a|c}}x
; X32:        ad{{d|c}}l  {{([0-9])*}}(%ebp), %e{{a|c}}x
; X32:        ad{{d|c}}l  {{([0-9])*}}(%ebp), %e{{a|c}}x
; X32:        ad{{d|c}}l  {{([0-9])*}}(%ebp), %e{{a|c}}x
; X32:        ad{{d|c}}l  {{([0-9])*}}(%ebp), %e{{a|c}}x
; X32:        ad{{d|c}}l  {{([0-9])*}}(%ebp), %e{{a|c}}x
; X32:        ad{{d|c}}l  {{([0-9])*}}(%ebp), %e{{a|c}}x
; X32:        retl

; WIN64-LABEL: test_argv64i1:
; WIN64:       addq    %rcx, %rax
; WIN64:       addq    %rdx, %rax
; WIN64:       addq    %rdi, %rax
; WIN64:       addq    %rsi, %rax
; WIN64:       addq    %r8, %rax
; WIN64:       addq    %r9, %rax
; WIN64:       addq    %r10, %rax
; WIN64:       addq    %r11, %rax
; WIN64:       addq    %r12, %rax
; WIN64:       addq    %r14, %rax
; WIN64:       addq    %r15, %rax
; WIN64:       addq  {{([0-9])*}}(%rsp), %rax
; WIN64:       retq

; LINUXOSX64-LABEL: test_argv64i1:
; LINUXOSX64:       addq    %rcx, %rax
; LINUXOSX64:       addq    %rdx, %rax
; LINUXOSX64:       addq    %rdi, %rax
; LINUXOSX64:       addq    %rsi, %rax
; LINUXOSX64:       addq    %r8, %rax
; LINUXOSX64:       addq    %r9, %rax
; LINUXOSX64:       addq    %r12, %rax
; LINUXOSX64:       addq    %r13, %rax
; LINUXOSX64:       addq    %r14, %rax
; LINUXOSX64:       addq    %r15, %rax
; LINUXOSX64:       addq    {{([0-9])*}}(%rsp), %rax
; LINUXOSX64:       addq    {{([0-9])*}}(%rsp), %rax
; LINUXOSX64:       retq

; Test regcall when receiving arguments of v64i1 type
define x86_regcallcc i64 @test_argv64i1(<64 x i1> %x0, <64 x i1> %x1, <64 x i1> %x2,
                                        <64 x i1> %x3, <64 x i1> %x4, <64 x i1> %x5,
                                        <64 x i1> %x6, <64 x i1> %x7, <64 x i1> %x8,
                                        <64 x i1> %x9, <64 x i1> %x10, <64 x i1> %x11,
                                        <64 x i1> %x12)  {
  %y0 = bitcast <64 x i1> %x0 to i64
  %y1 = bitcast <64 x i1> %x1 to i64
  %y2 = bitcast <64 x i1> %x2 to i64
  %y3 = bitcast <64 x i1> %x3 to i64
  %y4 = bitcast <64 x i1> %x4 to i64
  %y5 = bitcast <64 x i1> %x5 to i64
  %y6 = bitcast <64 x i1> %x6 to i64
  %y7 = bitcast <64 x i1> %x7 to i64
  %y8 = bitcast <64 x i1> %x8 to i64
  %y9 = bitcast <64 x i1> %x9 to i64
  %y10 = bitcast <64 x i1> %x10 to i64
  %y11 = bitcast <64 x i1> %x11 to i64
  %y12 = bitcast <64 x i1> %x12 to i64
  %add1 = add i64 %y0, %y1
  %add2 = add i64 %add1, %y2
  %add3 = add i64 %add2, %y3
  %add4 = add i64 %add3, %y4
  %add5 = add i64 %add4, %y5
  %add6 = add i64 %add5, %y6
  %add7 = add i64 %add6, %y7
  %add8 = add i64 %add7, %y8
  %add9 = add i64 %add8, %y9
  %add10 = add i64 %add9, %y10
  %add11 = add i64 %add10, %y11
  %add12 = add i64 %add11, %y12
  ret i64 %add12
}

; X32-LABEL:  caller_argv64i1:
; X32:        movl    $2, %eax
; X32:        movl    $1, %ecx
; X32:        movl    $2, %edx
; X32:        movl    $1, %edi
; X32:        pushl    ${{1|2}}
; X32:        pushl    ${{1|2}}
; X32:        pushl    ${{1|2}}
; X32:        pushl    ${{1|2}}
; X32:        pushl    ${{1|2}}
; X32:        pushl    ${{1|2}}
; X32:        pushl    ${{1|2}}
; X32:        pushl    ${{1|2}}
; X32:        pushl    ${{1|2}}
; X32:        pushl    ${{1|2}}
; X32:        pushl    ${{1|2}}
; X32:        pushl    ${{1|2}}
; X32:        pushl    ${{1|2}}
; X32:        pushl    ${{1|2}}
; X32:        pushl    ${{1|2}}
; X32:        pushl    ${{1|2}}
; X32:        pushl    ${{1|2}}
; X32:        pushl    ${{1|2}}
; X32:        pushl    ${{1|2}}
; X32:        pushl    ${{1|2}}
; X32:        pushl    ${{1|2}}
; X32:        pushl    ${{1|2}}
; X32:        call{{.*}}   _test_argv64i1
        
; WIN64-LABEL: caller_argv64i1:
; WIN64:       movabsq    $4294967298, %rax
; WIN64:       movq   %rax, (%rsp)
; WIN64:       movq   %rax, %rcx
; WIN64:       movq   %rax, %rdx
; WIN64:       movq   %rax, %rdi
; WIN64:       movq   %rax, %rsi
; WIN64:       movq   %rax, %r8
; WIN64:       movq   %rax, %r9
; WIN64:       movq   %rax, %r10
; WIN64:       movq   %rax, %r11
; WIN64:       movq   %rax, %r12
; WIN64:       movq   %rax, %r14
; WIN64:       movq   %rax, %r15
; WIN64:       callq   test_argv64i1

; LINUXOSX64-LABEL: caller_argv64i1:
; LINUXOSX64:       movabsq    $4294967298, %rax
; LINUXOSX64:       movq   %rax, %rcx
; LINUXOSX64:       movq   %rax, %rdx
; LINUXOSX64:       movq   %rax, %rdi
; LINUXOSX64:       movq   %rax, %rsi
; LINUXOSX64:       movq   %rax, %r8
; LINUXOSX64:       movq   %rax, %r9
; LINUXOSX64:       movq   %rax, %r12
; LINUXOSX64:       movq   %rax, %r13
; LINUXOSX64:       movq   %rax, %r14
; LINUXOSX64:       movq   %rax, %r15
; LINUXOSX64:       call{{.*}}   test_argv64i1

; Test regcall when passing arguments of v64i1 type
define x86_regcallcc i64 @caller_argv64i1() #0 {
entry:
  %v0 = bitcast i64 4294967298 to <64 x i1>
  %call = call x86_regcallcc i64 @test_argv64i1(<64 x i1> %v0, <64 x i1> %v0, <64 x i1> %v0,
                                                <64 x i1> %v0, <64 x i1> %v0, <64 x i1> %v0,
                                                <64 x i1> %v0, <64 x i1> %v0, <64 x i1> %v0,
                                                <64 x i1> %v0, <64 x i1> %v0, <64 x i1> %v0,
                                                <64 x i1> %v0)
  ret i64 %call
}

; X32-LABEL: test_retv64i1:
; X32:       mov{{.*}}    $2, %eax
; X32:       mov{{.*}}    $1, %ecx
; X32:       ret{{.*}}

; WIN64-LABEL: test_retv64i1:
; WIN64:       mov{{.*}} $4294967298, %rax
; WIN64:       ret{{.*}}

; Test regcall when returning v64i1 type
define x86_regcallcc <64 x i1> @test_retv64i1()  {
  %a = bitcast i64 4294967298 to <64 x i1>
 ret <64 x i1> %a
}

; X32-LABEL: caller_retv64i1:
; X32:       call{{.*}}   _test_retv64i1
; X32:       kmov{{.*}}   %eax, %k0
; X32:       kmov{{.*}}   %ecx, %k1
; X32:       kunpckdq     %k0, %k1, %k0

; Test regcall when processing result of v64i1 type
define x86_regcallcc <64 x i1> @caller_retv64i1() #0 {
entry:
  %call = call x86_regcallcc <64 x i1> @test_retv64i1()
  ret <64 x i1> %call
}
