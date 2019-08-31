; RUN: llc < %s -O0 -mattr=sse2 -mtriple=x86_64-pc-windows-itanium | FileCheck %s -check-prefix=WIN64 -check-prefix=NORM
; RUN: llc < %s -O0 -mattr=sse2 -mtriple=x86_64-pc-mingw32 | FileCheck %s -check-prefix=WIN64 -check-prefix=NORM
; RUN: llc < %s -O0 -mattr=sse2 -mtriple=x86_64-pc-mingw32 -mcpu=atom | FileCheck %s -check-prefix=WIN64 -check-prefix=ATOM

; Check function without prolog
define void @foo0() uwtable {
entry:
  ret void
}
; WIN64-LABEL: foo0:
; WIN64-NOT: .seh_proc foo0
; WIN64-NOT: .seh_endprologue
; WIN64: ret
; WIN64-NOT: .seh_endproc

; Checks a small stack allocation
define void @foo1() uwtable {
entry:
  %baz = alloca [2000 x i16], align 2
  ret void
}
; WIN64-LABEL: foo1:
; WIN64: .seh_proc foo1
; NORM:  subq $4000, %rsp
; ATOM:  leaq -4000(%rsp), %rsp
; WIN64: .seh_stackalloc 4000
; WIN64: .seh_endprologue
; WIN64: addq $4000, %rsp
; WIN64: ret
; WIN64: .seh_endproc

; Checks a stack allocation requiring call to __chkstk/___chkstk_ms
define void @foo2() uwtable {
entry:
  %baz = alloca [4000 x i16], align 2
  ret void
}
; WIN64-LABEL: foo2:
; WIN64: .seh_proc foo2
; WIN64: movl $8000, %eax
; WIN64: callq {{__chkstk|___chkstk_ms}}
; WIN64: subq %rax, %rsp
; WIN64: .seh_stackalloc 8000
; WIN64: .seh_endprologue
; WIN64: addq $8000, %rsp
; WIN64: ret
; WIN64: .seh_endproc


define i32 @foo3(i32 %f_arg, i32 %e_arg, i32 %d_arg, i32 %c_arg, i32 %b_arg, i32 %a_arg) uwtable {
entry:
  %a = alloca i32
  %b = alloca i32
  %c = alloca i32
  %d = alloca i32
  %e = alloca i32
  %f = alloca i32
  store i32 %a_arg, i32* %a
  store i32 %b_arg, i32* %b
  store i32 %c_arg, i32* %c
  store i32 %d_arg, i32* %d
  store i32 %e_arg, i32* %e
  store i32 %f_arg, i32* %f
  %tmp = load i32, i32* %a
  %tmp1 = mul i32 %tmp, 2
  %tmp2 = load i32, i32* %b
  %tmp3 = mul i32 %tmp2, 3
  %tmp4 = add i32 %tmp1, %tmp3
  %tmp5 = load i32, i32* %c
  %tmp6 = mul i32 %tmp5, 5
  %tmp7 = add i32 %tmp4, %tmp6
  %tmp8 = load i32, i32* %d
  %tmp9 = mul i32 %tmp8, 7
  %tmp10 = add i32 %tmp7, %tmp9
  %tmp11 = load i32, i32* %e
  %tmp12 = mul i32 %tmp11, 11
  %tmp13 = add i32 %tmp10, %tmp12
  %tmp14 = load i32, i32* %f
  %tmp15 = mul i32 %tmp14, 13
  %tmp16 = add i32 %tmp13, %tmp15
  ret i32 %tmp16
}
; WIN64-LABEL: foo3:
; WIN64: .seh_proc foo3
; NORM:  subq $24, %rsp
; ATOM:  leaq -24(%rsp), %rsp
; WIN64: .seh_stackalloc 24
; WIN64: .seh_endprologue
; WIN64: addq $24, %rsp
; WIN64: ret
; WIN64: .seh_endproc


; Check emission of eh handler and handler data
declare i32 @_d_eh_personality(i32, i32, i64, i8*, i8*)
declare void @_d_eh_resume_unwind(i8*)

declare i32 @bar()

define i32 @foo4() #0 personality i32 (i32, i32, i64, i8*, i8*)* @_d_eh_personality {
entry:
  %step = alloca i32, align 4
  store i32 0, i32* %step
  %tmp = load i32, i32* %step

  %tmp1 = invoke i32 @bar()
          to label %finally unwind label %landingpad

finally:
  store i32 1, i32* %step
  br label %endtryfinally

landingpad:
  %landing_pad = landingpad { i8*, i32 }
          cleanup
  %tmp3 = extractvalue { i8*, i32 } %landing_pad, 0
  store i32 2, i32* %step
  call void @_d_eh_resume_unwind(i8* %tmp3)
  unreachable

endtryfinally:
  %tmp10 = load i32, i32* %step
  ret i32 %tmp10
}
; WIN64-LABEL: foo4:
; WIN64: .seh_proc foo4
; WIN64: .seh_handler _d_eh_personality, @unwind, @except
; NORM:  subq $40, %rsp
; ATOM:  leaq -40(%rsp), %rsp
; WIN64: .seh_stackalloc 40
; WIN64: .seh_endprologue
; WIN64: addq $40, %rsp
; WIN64: ret
; WIN64: .seh_handlerdata
; WIN64: .seh_endproc


; Check stack re-alignment and xmm spilling
define void @foo5() uwtable {
entry:
  %s = alloca i32, align 64
  call void asm sideeffect "", "~{rbx},~{rdi},~{xmm6},~{xmm7}"()
  ret void
}
; WIN64-LABEL: foo5:
; WIN64: .seh_proc foo5
; WIN64: pushq %rbp
; WIN64: .seh_pushreg %rbp
; WIN64: pushq %rdi
; WIN64: .seh_pushreg %rdi
; WIN64: pushq %rbx
; WIN64: .seh_pushreg %rbx
; NORM:  subq  $96, %rsp
; ATOM:  leaq -96(%rsp), %rsp
; WIN64: .seh_stackalloc 96
; WIN64: leaq  96(%rsp), %rbp
; WIN64: .seh_setframe %rbp, 96
; WIN64: movaps  %xmm7, -16(%rbp)        # 16-byte Spill
; WIN64: .seh_savexmm %xmm7, 80
; WIN64: movaps  %xmm6, -32(%rbp)        # 16-byte Spill
; WIN64: .seh_savexmm %xmm6, 64
; WIN64: .seh_endprologue
; WIN64: andq  $-64, %rsp
; WIN64: movaps  -32(%rbp), %xmm6        # 16-byte Reload
; WIN64: movaps  -16(%rbp), %xmm7        # 16-byte Reload
; WIN64: movq  %rbp, %rsp
; WIN64: popq  %rbx
; WIN64: popq  %rdi
; WIN64: popq  %rbp
; WIN64: retq
; WIN64: .seh_endproc
