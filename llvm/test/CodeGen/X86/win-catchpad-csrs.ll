; RUN: llc -verify-machineinstrs -mtriple=i686-pc-windows-msvc < %s | FileCheck --check-prefix=X86 %s
; RUN: llc -verify-machineinstrs -mtriple=x86_64-pc-windows-msvc < %s | FileCheck --check-prefix=X64 %s

%rtti.TypeDescriptor2 = type { i8**, i8*, [3 x i8] }
%eh.CatchableType = type { i32, i8*, i32, i32, i32, i32, i8* }
%eh.CatchableTypeArray.1 = type { i32, [1 x %eh.CatchableType*] }
%eh.ThrowInfo = type { i32, i8*, i8*, i8* }

$"\01??_R0H@8" = comdat any

@"\01??_7type_info@@6B@" = external constant i8*
@"\01??_R0H@8" = linkonce_odr global %rtti.TypeDescriptor2 { i8** @"\01??_7type_info@@6B@", i8* null, [3 x i8] c".H\00" }, comdat

declare i32 @getint()
declare void @useints(...)
declare void @f(i32 %p)
declare i32 @__CxxFrameHandler3(...)

define i32 @try_catch_catch() personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  %a = call i32 @getint()
  %b = call i32 @getint()
  %c = call i32 @getint()
  %d = call i32 @getint()
  call void (...) @useints(i32 %a, i32 %b, i32 %c, i32 %d)
  invoke void @f(i32 1)
          to label %try.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %0 = catchpad [%rtti.TypeDescriptor2* @"\01??_R0H@8", i32 0, i8* null]
          to label %catch unwind label %catchendblock

catch:
  invoke void @f(i32 2)
          to label %invoke.cont.2 unwind label %catchendblock

invoke.cont.2:                                    ; preds = %catch
  catchret %0 to label %try.cont

try.cont:                                         ; preds = %entry, %invoke.cont.2, %invoke.cont.3
  ret i32 0

catchendblock:                                    ; preds = %catch,
  catchendpad unwind to caller
}

; X86-LABEL: _try_catch_catch:
; X86: pushl %ebp
; X86: movl %esp, %ebp
; X86: pushl %ebx
; X86: pushl %edi
; X86: pushl %esi
; X86: subl ${{[0-9]+}}, %esp
; X86: calll _getint
; X86: calll _getint
; X86: calll _getint
; X86: calll _getint
; X86: calll _useints
; X86: movl $0, -{{[0-9]+}}(%ebp)
; X86: movl $1, (%esp)
; X86: calll _f
; X86: [[contbb:LBB0_[0-9]+]]: # %try.cont
; X86: popl %esi
; X86: popl %edi
; X86: popl %ebx
; X86: popl %ebp
; X86: retl

; X86: [[restorebb:LBB0_[0-9]+]]:
; X86: addl $12, %ebp
; X86: jmp [[contbb]]

; X86: "?catch$[[catch1bb:[0-9]+]]@?0?try_catch_catch@4HA":
; X86: LBB0_[[catch1bb]]: # %catch.dispatch{{$}}
; X86: pushl %ebp
; X86-NOT: pushl
; X86: subl $16, %esp
; X86: addl $12, %ebp
; X86: movl $1, -{{[0-9]+}}(%ebp)
; X86: movl $2, (%esp)
; X86: calll _f
; X86: movl $[[restorebb]], %eax
; X86-NEXT: addl $16, %esp
; X86-NEXT: popl %ebp
; X86-NEXT: retl

; X86: L__ehtable$try_catch_catch:
; X86: $handlerMap$0$try_catch_catch:
; X86:   .long   0
; X86:   .long   "??_R0H@8"
; X86:   .long   0
; X86:   .long   "?catch$[[catch1bb]]@?0?try_catch_catch@4HA"

; X64-LABEL: try_catch_catch:
; X64: pushq %rbp
; X64: .seh_pushreg 5
; X64: pushq %rsi
; X64: .seh_pushreg 6
; X64: pushq %rdi
; X64: .seh_pushreg 7
; X64: pushq %rbx
; X64: .seh_pushreg 3
; X64: subq $40, %rsp
; X64: .seh_stackalloc 40
; X64: leaq 32(%rsp), %rbp
; X64: .seh_setframe 5, 32
; X64: .seh_endprologue
; X64: movq $-2, (%rbp)
; X64: callq getint
; X64: callq getint
; X64: callq getint
; X64: callq getint
; X64: callq useints
; X64: movl $1, %ecx
; X64: callq f
; X64: [[contbb:\.LBB0_[0-9]+]]: # Block address taken
; X64-NEXT:                      # %try.cont
; X64: addq $40, %rsp
; X64: popq %rbp
; X64: retq

; X64: "?catch$[[catch1bb:[0-9]+]]@?0?try_catch_catch@4HA":
; X64: LBB0_[[catch1bb]]: # %catch.dispatch{{$}}
; X64: movq %rdx, 16(%rsp)
; X64: pushq %rbp
; X64: .seh_pushreg 5
; X64: pushq %rsi
; X64: .seh_pushreg 6
; X64: pushq %rdi
; X64: .seh_pushreg 7
; X64: pushq %rbx
; X64: .seh_pushreg 3
; X64: subq $40, %rsp
; X64: .seh_stackalloc 40
; X64: leaq 32(%rdx), %rbp
; X64: .seh_endprologue
; X64: movl $2, %ecx
; X64: callq f
; X64: leaq [[contbb]](%rip), %rax
; X64: addq $40, %rsp
; X64: popq %rbx
; X64: popq %rdi
; X64: popq %rsi
; X64: popq %rbp
; X64: retq

; X64: $handlerMap$0$try_catch_catch:
; X64:   .long   0
; X64:   .long   "??_R0H@8"@IMGREL
; X64:   .long   0
; X64:   .long   "?catch$[[catch1bb]]@?0?try_catch_catch@4HA"@IMGREL
; X64:   .long   88

define i32 @try_one_csr() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  %a = call i32 @getint()
  %b = call i32 @getint()
  call void (...) @useints(i32 %a)
  invoke void @f(i32 1)
          to label %try.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %0 = catchpad [%rtti.TypeDescriptor2* @"\01??_R0H@8", i32 0, i8* null]
          to label %catch unwind label %catchendblock

catch:
  catchret %0 to label %try.cont

try.cont:                                         ; preds = %entry, %invoke.cont.2, %invoke.cont.3
  ret i32 0

catchendblock:                                    ; preds = %catch,
  catchendpad unwind to caller
}

; X64-LABEL: try_one_csr:
; X64: pushq %rbp
; X64: .seh_pushreg 5
; X64: pushq %rsi
; X64: .seh_pushreg 6
; X64-NOT: pushq
; X64: subq $40, %rsp
; X64: .seh_stackalloc 40
; X64: leaq 32(%rsp), %rbp
; X64: .seh_setframe 5, 32
; X64: .seh_endprologue
; X64: callq getint
; X64: callq getint
; X64: callq useints
; X64: movl $1, %ecx
; X64: callq f
; X64: [[contbb:\.LBB1_[0-9]+]]: # Block address taken
; X64-NEXT:                      # %try.cont
; X64: addq $40, %rsp
; X64-NOT: popq
; X64: popq %rsi
; X64: popq %rbp
; X64: retq

; X64: "?catch$[[catch1bb:[0-9]+]]@?0?try_one_csr@4HA":
; X64: LBB1_[[catch1bb]]: # %catch.dispatch{{$}}
; X64: movq %rdx, 16(%rsp)
; X64: pushq %rbp
; X64: .seh_pushreg 5
; X64: pushq %rsi
; X64: .seh_pushreg 6
; X64: subq $40, %rsp
; X64: .seh_stackalloc 40
; X64: leaq 32(%rdx), %rbp
; X64: .seh_endprologue
; X64: leaq [[contbb]](%rip), %rax
; X64: addq $40, %rsp
; X64: popq %rsi
; X64: popq %rbp
; X64: retq

; X64: $handlerMap$0$try_one_csr:
; X64:   .long   0
; X64:   .long   "??_R0H@8"@IMGREL
; X64:   .long   0
; X64:   .long   "?catch$[[catch1bb]]@?0?try_one_csr@4HA"@IMGREL
; X64:   .long   72

define i32 @try_no_csr() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  invoke void @f(i32 1)
          to label %try.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %0 = catchpad [%rtti.TypeDescriptor2* @"\01??_R0H@8", i32 0, i8* null]
          to label %catch unwind label %catchendblock

catch:
  catchret %0 to label %try.cont

try.cont:                                         ; preds = %entry, %invoke.cont.2, %invoke.cont.3
  ret i32 0

catchendblock:                                    ; preds = %catch,
  catchendpad unwind to caller
}

; X64-LABEL: try_no_csr:
; X64: pushq %rbp
; X64: .seh_pushreg 5
; X64-NOT: pushq
; X64: subq $48, %rsp
; X64: .seh_stackalloc 48
; X64: leaq 48(%rsp), %rbp
; X64: .seh_setframe 5, 48
; X64: .seh_endprologue
; X64: movl $1, %ecx
; X64: callq f
; X64: [[contbb:\.LBB2_[0-9]+]]: # Block address taken
; X64-NEXT:                      # %try.cont
; X64: addq $48, %rsp
; X64-NOT: popq
; X64: popq %rbp
; X64: retq

; X64: "?catch$[[catch1bb:[0-9]+]]@?0?try_no_csr@4HA":
; X64: LBB2_[[catch1bb]]: # %catch.dispatch{{$}}
; X64: movq %rdx, 16(%rsp)
; X64: pushq %rbp
; X64: .seh_pushreg 5
; X64: subq $32, %rsp
; X64: .seh_stackalloc 32
; X64: leaq 48(%rdx), %rbp
; X64: .seh_endprologue
; X64: leaq [[contbb]](%rip), %rax
; X64: addq $32, %rsp
; X64: popq %rbp
; X64: retq

; X64: $handlerMap$0$try_no_csr:
; X64:   .long   0
; X64:   .long   "??_R0H@8"@IMGREL
; X64:   .long   0
; X64:   .long   "?catch$[[catch1bb]]@?0?try_no_csr@4HA"@IMGREL
; X64:   .long   56
