; RUN: llc -verify-machineinstrs -mtriple=i686-pc-windows-msvc < %s | FileCheck --check-prefix=X86 %s
; RUN: llc -verify-machineinstrs -mtriple=x86_64-pc-windows-msvc < %s | FileCheck --check-prefix=X64 %s

; Loosely based on IR for this C++ source code:
;   void f(int p);
;   int main() {
;     try {
;       f(1);
;     } catch (int e) {
;       f(e);
;     } catch (...) {
;       f(3);
;     }
;   }

%rtti.TypeDescriptor2 = type { i8**, i8*, [3 x i8] }
%eh.CatchableType = type { i32, i8*, i32, i32, i32, i32, i8* }
%eh.CatchableTypeArray.1 = type { i32, [1 x %eh.CatchableType*] }
%eh.ThrowInfo = type { i32, i8*, i8*, i8* }

$"\01??_R0H@8" = comdat any

@"\01??_7type_info@@6B@" = external constant i8*
@"\01??_R0H@8" = linkonce_odr global %rtti.TypeDescriptor2 { i8** @"\01??_7type_info@@6B@", i8* null, [3 x i8] c".H\00" }, comdat


declare void @f(i32 %p, i32* %l)
declare i1 @getbool()
declare i32 @__CxxFrameHandler3(...)

define i32 @try_catch_catch() personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  %e.addr = alloca i32
  %local = alloca i32
  invoke void @f(i32 1, i32* %local)
          to label %try.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %0 = catchpad [%rtti.TypeDescriptor2* @"\01??_R0H@8", i32 0, i32* %e.addr]
          to label %catch unwind label %catch.dispatch.2

catch:                                            ; preds = %catch.dispatch
  %e = load i32, i32* %e.addr
  invoke void @f(i32 %e, i32* %local)
          to label %invoke.cont.2 unwind label %catchendblock

invoke.cont.2:                                    ; preds = %catch
  catchret %0 to label %try.cont

catch.dispatch.2:                                   ; preds = %catch.dispatch
  %1 = catchpad [i8* null, i32 u0x40, i8* null]
          to label %catch.2 unwind label %catchendblock

catch.2:                                            ; preds = %catch.dispatch.2
  invoke void @f(i32 3, i32* %local)
          to label %invoke.cont.3 unwind label %catchendblock

invoke.cont.3:                                    ; preds = %catch.2
  catchret %1 to label %try.cont

try.cont:                                         ; preds = %entry, %invoke.cont.2, %invoke.cont.3
  ret i32 0

catchendblock:                                    ; preds = %catch, %catch.2, %catch.dispatch.2
  catchendpad unwind to caller
}

; X86-LABEL: _try_catch_catch:
; X86: movl $0, -{{[0-9]+}}(%ebp)
; X86: leal -[[local_offs:[0-9]+]](%ebp), %[[addr_reg:[a-z]+]]
; X86-DAG: movl %[[addr_reg]], 4(%esp)
; X86-DAG: movl $1, (%esp)
; X86: calll _f
; X86: [[contbb:LBB0_[0-9]+]]: # %try.cont
; X86: retl

; X86: [[restorebb1:LBB0_[0-9]+]]: # Block address taken
; X86-NEXT:                        # %invoke.cont.2
; X86: movl -16(%ebp), %esp
; X86: addl $12, %ebp
; X86: jmp [[contbb]]

; FIXME: These should be de-duplicated.
; X86: [[restorebb2:LBB0_[0-9]+]]: # Block address taken
; X86-NEXT:                        # %invoke.cont.3
; X86: movl -16(%ebp), %esp
; X86: addl $12, %ebp
; X86: jmp [[contbb]]

; X86: "?catch$[[catch1bb:[0-9]+]]@?0?try_catch_catch@4HA":
; X86: LBB0_[[catch1bb]]: # %catch.dispatch{{$}}
; X86: pushl %ebp
; X86: subl $8, %esp
; X86: addl $12, %ebp
; X86: leal -[[local_offs]](%ebp), %[[addr_reg:[a-z]+]]
; X86: movl -32(%ebp), %[[e_reg:[a-z]+]]
; X86: movl $1, -{{[0-9]+}}(%ebp)
; X86-DAG: movl %[[addr_reg]], 4(%esp)
; X86-DAG: movl %[[e_reg]], (%esp)
; X86: calll _f
; X86-NEXT: movl $[[restorebb1]], %eax
; X86-NEXT: addl $8, %esp
; X86-NEXT: popl %ebp
; X86-NEXT: retl

; X86: "?catch$[[catch2bb:[0-9]+]]@?0?try_catch_catch@4HA":
; X86: LBB0_[[catch2bb]]: # %catch.dispatch.2{{$}}
; X86: pushl %ebp
; X86: subl $8, %esp
; X86: addl $12, %ebp
; X86: leal -[[local_offs]](%ebp), %[[addr_reg:[a-z]+]]
; X86: movl $1, -{{[0-9]+}}(%ebp)
; X86-DAG: movl %[[addr_reg]], 4(%esp)
; X86-DAG: movl $3, (%esp)
; X86: calll _f
; X86-NEXT: movl $[[restorebb2]], %eax
; X86-NEXT: addl $8, %esp
; X86-NEXT: popl %ebp
; X86-NEXT: retl

; X86: L__ehtable$try_catch_catch:
; X86: $handlerMap$0$try_catch_catch:
; X86-NEXT:   .long   0
; X86-NEXT:   .long   "??_R0H@8"
; X86-NEXT:   .long   -20
; X86-NEXT:   .long   "?catch$[[catch1bb]]@?0?try_catch_catch@4HA"
; X86-NEXT:   .long   64
; X86-NEXT:   .long   0
; X86-NEXT:   .long   0
; X86-NEXT:   .long   "?catch$[[catch2bb]]@?0?try_catch_catch@4HA"

; X64-LABEL: try_catch_catch:
; X64: Lfunc_begin0:
; X64: pushq %rbp
; X64: .seh_pushreg 5
; X64: subq $48, %rsp
; X64: .seh_stackalloc 48
; X64: leaq 48(%rsp), %rbp
; X64: .seh_setframe 5, 48
; X64: .seh_endprologue
; X64: .Ltmp0
; X64-DAG: leaq -[[local_offs:[0-9]+]](%rbp), %rdx
; X64-DAG: movl $1, %ecx
; X64: callq f
; X64: [[contbb:\.LBB0_[0-9]+]]: # Block address taken
; X64-NEXT:                      # %try.cont
; X64: addq $48, %rsp
; X64: popq %rbp
; X64: retq

; X64: "?catch$[[catch1bb:[0-9]+]]@?0?try_catch_catch@4HA":
; X64: LBB0_[[catch1bb]]: # %catch.dispatch{{$}}
; X64: movq %rdx, 16(%rsp)
; X64: pushq %rbp
; X64: .seh_pushreg 5
; X64: subq $32, %rsp
; X64: .seh_stackalloc 32
; X64: leaq 48(%rdx), %rbp
; X64: .seh_endprologue
; X64-DAG: .Ltmp4
; X64-DAG: leaq -[[local_offs]](%rbp), %rdx
; X64-DAG: movl -4(%rbp), %ecx
; X64: callq f
; X64: leaq [[contbb]](%rip), %rax
; X64-NEXT: addq $32, %rsp
; X64-NEXT: popq %rbp
; X64-NEXT: retq

; X64: "?catch$[[catch2bb:[0-9]+]]@?0?try_catch_catch@4HA":
; X64: LBB0_[[catch2bb]]: # %catch.dispatch.2{{$}}
; X64: movq %rdx, 16(%rsp)
; X64: pushq %rbp
; X64: .seh_pushreg 5
; X64: subq $32, %rsp
; X64: .seh_stackalloc 32
; X64: leaq 48(%rdx), %rbp
; X64: .seh_endprologue
; X64-DAG: leaq -[[local_offs]](%rbp), %rdx
; X64-DAG: movl $3, %ecx
; X64: callq f
; X64: .Ltmp3
; X64: leaq [[contbb]](%rip), %rax
; X64-NEXT: addq $32, %rsp
; X64-NEXT: popq %rbp
; X64-NEXT: retq

; X64: $cppxdata$try_catch_catch:
; X64-NEXT: .long   429065506
; X64-NEXT: .long   2
; X64-NEXT: .long   ($stateUnwindMap$try_catch_catch)@IMGREL
; X64-NEXT: .long   1
; X64-NEXT: .long   ($tryMap$try_catch_catch)@IMGREL
; X64-NEXT: .long   4
; X64-NEXT: .long   ($ip2state$try_catch_catch)@IMGREL
; X64-NEXT: .long   32
; X64-NEXT: .long   0
; X64-NEXT: .long   1

; X64: $tryMap$try_catch_catch:
; X64-NEXT: .long   0
; X64-NEXT: .long   0
; X64-NEXT: .long   1
; X64-NEXT: .long   2
; X64-NEXT: .long   ($handlerMap$0$try_catch_catch)@IMGREL

; X64: $handlerMap$0$try_catch_catch:
; X64-NEXT:   .long   0
; X64-NEXT:   .long   "??_R0H@8"@IMGREL
; FIXME: This should probably be offset from rsp, not rbp.
; X64-NEXT:   .long   44
; X64-NEXT:   .long   "?catch$[[catch1bb]]@?0?try_catch_catch@4HA"@IMGREL
; X64-NEXT:   .long   56
; X64-NEXT:   .long   64
; X64-NEXT:   .long   0
; X64-NEXT:   .long   0
; X64-NEXT:   .long   "?catch$[[catch2bb]]@?0?try_catch_catch@4HA"@IMGREL
; X64-NEXT:   .long   56

; X64: $ip2state$try_catch_catch:
; X64-NEXT: .long   .Lfunc_begin0@IMGREL
; X64-NEXT: .long   -1
; X64-NEXT: .long   .Ltmp0@IMGREL+1
; X64-NEXT: .long   0
; X64-NEXT: .long   .Ltmp4@IMGREL+1
; X64-NEXT: .long   1
; X64-NEXT: .long   .Ltmp3@IMGREL+1
; X64-NEXT: .long   -1


define i32 @branch_to_normal_dest() personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  invoke void @f(i32 1, i32* null)
          to label %try.cont unwind label %catch.dispatch

catch.dispatch:
  %0 = catchpad [i8* null, i32 64, i8* null]
          to label %catch unwind label %catchendblock

catch:
  %V = call i1 @getbool()
  br i1 %V, label %catch, label %catch.done

catch.done:
  catchret %0 to label %try.cont

try.cont:
  ret i32 0

catchendblock:
  catchendpad unwind to caller
}

; X86-LABEL: _branch_to_normal_dest:
; X86: calll _f

; X86: [[contbb:LBB1_[0-9]+]]: # %try.cont
; X86: retl

; X86: [[restorebb:LBB1_[0-9]+]]: # Block address taken
; X86-NEXT:                       # %catch.done
; X86: movl -16(%ebp), %esp
; X86: addl $12, %ebp
; X86: jmp [[contbb]]

; X86: "?catch$[[catchdispbb:[0-9]+]]@?0?branch_to_normal_dest@4HA":
; X86: LBB1_[[catchdispbb]]: # %catch.dispatch{{$}}
; X86: pushl %ebp
; X86: subl $8, %esp
; X86: addl $12, %ebp

; X86: LBB1_[[catchbb:[0-9]+]]: # %catch
; X86: movl    $-1, -16(%ebp)
; X86: calll   _getbool
; X86: testb   $1, %al
; X86: jne LBB1_[[catchbb]]
; X86: # %catch.done
; X86-NEXT: movl $[[restorebb]], %eax
; X86-NEXT: addl $8, %esp
; X86-NEXT: popl %ebp
; X86-NEXT: retl

; X86: L__ehtable$branch_to_normal_dest:
; X86: $handlerMap$0$branch_to_normal_dest:
; X86-NEXT:   .long   64
; X86-NEXT:   .long   0
; X86-NEXT:   .long   0
; X86-NEXT:   .long   "?catch$[[catchdispbb]]@?0?branch_to_normal_dest@4HA"

; X64-LABEL: branch_to_normal_dest:
; X64: # %entry
; X64: pushq %rbp
; X64: .seh_pushreg 5
; X64: subq $48, %rsp
; X64: .seh_stackalloc 48
; X64: leaq 48(%rsp), %rbp
; X64: .seh_setframe 5, 48
; X64: .seh_endprologue
; X64: .Ltmp[[before_call:[0-9]+]]:
; X64: callq f
; X64: .Ltmp[[after_call:[0-9]+]]:
; X64: [[contbb:\.LBB1_[0-9]+]]: # Block address taken
; X64-NEXT:                      # %try.cont
; X64: addq $48, %rsp
; X64: popq %rbp
; X64: retq

; X64: "?catch$[[catchbb:[0-9]+]]@?0?branch_to_normal_dest@4HA":
; X64: LBB1_[[catchbb]]: # %catch.dispatch{{$}}
; X64: movq %rdx, 16(%rsp)
; X64: pushq %rbp
; X64: .seh_pushreg 5
; X64: subq $32, %rsp
; X64: .seh_stackalloc 32
; X64: leaq 48(%rdx), %rbp
; X64: .seh_endprologue
; X64: .LBB1_[[normal_dest_bb:[0-9]+]]: # %catch
; X64: callq   getbool
; X64: testb   $1, %al
; X64: jne     .LBB1_[[normal_dest_bb]]
; X64: # %catch.done
; X64: leaq [[contbb]](%rip), %rax
; X64-NEXT: addq $32, %rsp
; X64-NEXT: popq %rbp
; X64-NEXT: retq

; X64-LABEL: $cppxdata$branch_to_normal_dest:
; X64-NEXT: .long   429065506
; X64-NEXT: .long   2
; X64-NEXT: .long   ($stateUnwindMap$branch_to_normal_dest)@IMGREL
; X64-NEXT: .long   1
; X64-NEXT: .long   ($tryMap$branch_to_normal_dest)@IMGREL
; X64-NEXT: .long   3
; X64-NEXT: .long   ($ip2state$branch_to_normal_dest)@IMGREL
; X64-NEXT: .long   40
; X64-NEXT: .long   0
; X64-NEXT: .long   1

; X64-LABEL: $stateUnwindMap$branch_to_normal_dest:
; X64-NEXT: .long   -1
; X64-NEXT: .long   0
; X64-NEXT: .long   -1
; X64-NEXT: .long   0

; X64-LABEL: $tryMap$branch_to_normal_dest:
; X64-NEXT: .long   0
; X64-NEXT: .long   0
; X64-NEXT: .long   1
; X64-NEXT: .long   1
; X64-NEXT: .long   ($handlerMap$0$branch_to_normal_dest)@IMGREL

; X64-LABEL: $handlerMap$0$branch_to_normal_dest:
; X64-NEXT: .long   64
; X64-NEXT: .long   0
; X64-NEXT: .long   0
; X64-NEXT: .long   "?catch$[[catchbb]]@?0?branch_to_normal_dest@4HA"@IMGREL
; X64-NEXT: .long   56

; X64-LABEL: $ip2state$branch_to_normal_dest:
; X64-NEXT: .long   .Lfunc_begin1@IMGREL
; X64-NEXT: .long   -1
; X64-NEXT: .long   .Ltmp[[before_call]]@IMGREL+1
; X64-NEXT: .long   0
; X64-NEXT: .long   .Ltmp[[after_call]]@IMGREL+1
; X64-NEXT: .long   -1
