; FIXME: Fix machine verifier issues and remove -verify-machineinstrs=0. PR39439.
; RUN: llc -mtriple i386-windows-gnu -regalloc-csr-cost-scale=0 -exception-model sjlj -filetype asm -o - %s -verify-machineinstrs=0 | FileCheck %s
; RUN: llc -mtriple x86_64-windows-gnu -exception-model sjlj -filetype asm -o - %s -verify-machineinstrs=0 | FileCheck %s -check-prefix CHECK-X64
; RUN: llc -mtriple x86_64-linux -exception-model sjlj -filetype asm -o - %s -verify-machineinstrs=0 | FileCheck %s -check-prefix CHECK-X64-LINUX
; RUN: llc -mtriple x86_64-linux-gnux32 -exception-model sjlj -filetype asm -o - %s -verify-machineinstrs=0 | FileCheck %s -check-prefix CHECK-X32

declare void @_Z20function_that_throwsv()
declare i32 @__gxx_personality_sj0(...)
declare ptr @__cxa_begin_catch(ptr)
declare void @__cxa_end_catch()

define void @_Z8functionv() personality ptr @__gxx_personality_sj0 {
entry:
  invoke void @_Z20function_that_throwsv()
          to label %try.cont unwind label %lpad

lpad:
  %0 = landingpad { ptr, i32 }
          catch ptr null
  %1 = extractvalue { ptr, i32 } %0, 0
  %2 = tail call ptr @__cxa_begin_catch(ptr %1)
  tail call void @__cxa_end_catch()
  br label %try.cont

try.cont:
  ret void
}

;     struct _Unwind_FunctionContext {
; +00   struct _Unwind_FunctionContext *prev;   -64(%ebp)
; +04   int __callsite;                         -60(%ebp)
; +08   _Unwind_Word __buffer[4];               -56(%ebp)
; +24   __personality_routine __personality;    -40(%ebp)
; +28   uintptr_t __lsda;                       -36(%ebp)
; +32   void *__jbuf[];                         -32(%ebp)
;     };


; CHECK-LABEL: __Z8functionv:
;     struct _Unwind_FunctionContext UFC;
;
;     UFC.__personality = __gxx_personality_sj0
; CHECK: movl $___gxx_personality_sj0, -40(%ebp)
;     UFC.__lsda = $LSDA
; CHECK: movl $[[LSDA:GCC_except_table[0-9]+]], -36(%ebp)
;     UFC.__jbuf[0] = $ebp
; CHECK: movl %ebp, -32(%ebp)
;     UFC.__jbuf[2] = $esp
; CHECK: movl %esp, -24(%ebp)
;     UFC.__jbuf[1] = $EIP
; CHECK: movl $[[RESUME:LBB[0-9]+_[0-9]+]], -28(%ebp)
;     _Unwind_SjLj_Register(&UFC);
; CHECK: leal -64(%ebp), %eax
; CHECK: pushl %eax
; CHECK: calll __Unwind_SjLj_Register
; CHECK: addl $4, %esp
;     function_that_throws();
; CHECK: movl $1, -60(%ebp)
; CHECK: calll __Z20function_that_throwsv
;     _Unwind_SjLj_Unregister(&UFC);
; CHECK: leal -64(%ebp), %eax
; CHECK: calll __Unwind_SjLj_Unregister
;
; CHECK: [[RESUME]]:
; CHECK: leal -64(%ebp), %esi
;     assert(UFC.__callsite < 1);
; CHECK: movl -60(%ebp), %eax
; CHECK: cmpl $1, %eax
; CHECK: jb [[CONT:LBB[0-9]+_[0-9]+]]
; CHECK: ud2
; CHECK: [[CONT]]:
;     *Handlers[UFC.__callsite]
; CHECK: jmpl *LJTI


;     struct _Unwind_FunctionContext {
; +00   struct _Unwind_FunctionContext *prev;   -328(%rbp)
; +08   int __callsite;                         -320(%rbp)
; +16   _Unwind_Word __buffer[4];               -312(%rbp)
; +48   __personality_routine __personality;    -280(%rbp)
; +56   uintptr_t __lsda;                       -272(%rbp)
; +64   void *__jbuf[];                         -264(%rbp)
;     };


; CHECK-X64-LABEL: _Z8functionv:
;     struct _Unwind_FunctionContext UFC;
;
;     UFC.__personality = __gxx_personality_sj0
; CHECK-X64: leaq __gxx_personality_sj0(%rip), %rax
; CHECK-X64: movq %rax, -280(%rbp)
;     UFC.__lsda = $LSDA
; CHECK-X64: leaq [[LSDA:GCC_except_table[0-9]+]](%rip), %rax
; CHECK-X64: movq %rax, -272(%rbp)
;     UFC.__jbuf[0] = $rbp
; CHECK-X64: movq %rbp, -264(%rbp)
;     UFC.__jbuf[2] = $rsp
; CHECK-X64: movq %rsp, -248(%rbp)
;     UFC.__jbuf[1] = $RIP
; CHECK-X64: leaq .[[RESUME:LBB[0-9]+_[0-9]+]](%rip), %rax
; CHECK-X64: movq %rax, -256(%rbp)
;     _Unwind_SjLj_Register(&UFC);
; CHECK-X64: leaq -328(%rbp), %rcx
; CHECK-X64: callq _Unwind_SjLj_Register
;     function_that_throws();
; CHECK-X64: movl $1, -320(%rbp)
; CHECK-X64: callq _Z20function_that_throwsv
;     _Unwind_SjLj_Unregister(&UFC);
; CHECK-X64: leaq -328(%rbp), %rcx
; CHECK-X64: callq _Unwind_SjLj_Unregister
;
; CHECK-X64: [[RESUME]]:
;     assert(UFC.__callsite < 1);
; CHECK-X64: movl -320(%rbp), %eax
; CHECK-X64: cmpl $1, %eax
; CHECK-X64: jb .[[CONT:LBB[0-9]+_[0-9]+]]
; CHECK-X64: ud2
; CHECK-X64: [[CONT]]:
;     *Handlers[UFC.__callsite]
; CHECK-X64: leaq .[[TABLE:LJTI[0-9]+_[0-9]+]](%rip), %rcx
; CHECK-X64: movl (%rcx,%rax,4), %rax
; CHECK-X64: leaq __ImageBase@IMGREL(%rip), %rcx
; CHECK-X64: addq %rax, %rcx
; CHECK-X64: jmpq *%rcx

; CHECK-X64-LINUX: .[[RESUME:LBB[0-9]+_[0-9]+]]:
;     assert(UFC.__callsite < 1);
; CHECK-X64-LINUX: movl -136(%rbp), %eax
; CHECK-X64-LINUX: cmpl $1, %eax
; CHECK-X64-LINUX: jb .[[CONT:LBB[0-9]+_[0-9]+]]
; CHECK-X64-LINUX: ud2
; CHECK-X64-LINUX: [[CONT]]:
;     *Handlers[UFC.__callsite]
; CHECK-X64-LINUX: leaq .[[TABLE:LJTI[0-9]+_[0-9]+]](%rip), %rcx
; CHECK-X64-LINUX: jmpq *(%rcx,%rax,8)

; x32 has 32-bit pointers and 64-bit __data words.
;     struct _Unwind_FunctionContext {
; +00   struct _Unwind_FunctionContext *prev;   -112(%ebp)
; +04   int __callsite;                         -108(%ebp)
; +08   _Unwind_Word __buffer[4];               -104(%ebp)
; +40   __personality_routine __personality;    -72(%ebp)
; +44   uintptr_t __lsda;                       -68(%ebp)
; +48   void *__jbuf[];                         -64(%ebp)
;     };

; CHECK-X32: movl %eax, -72(%ebp)
; CHECK-X32: movl $GCC_except_table0, -68(%ebp)
; CHECK-X32: movl %ebp, -64(%ebp)
; CHECK-X32: movl %esp, -56(%ebp)
; CHECK-X32: movl $.[[RESUME:LBB[0-9]+_[0-9]+]], -60(%ebp)
; CHECK-X32: leal -112(%rbp), %edi
; CHECK-X32: callq _Unwind_SjLj_Register@PLT
; CHECK-X32: movl $1, -108(%ebp)
; CHECK-X32: .[[RESUME]]:
; CHECK-X32: movl -108(%ebp), %eax
; CHECK-X32: movq -104(%ebp), %rax
; CHECK-X32: movq -96(%ebp), %rcx
