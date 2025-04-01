; FIXME: Fix machine verifier issues and remove -verify-machineinstrs=0. PR39439.
; RUN: llc -mtriple i386-windows-gnu -exception-model sjlj -filetype asm -o - %s -verify-machineinstrs=0 | FileCheck %s
; RUN: llc -mtriple x86_64-windows-gnu -exception-model sjlj -filetype asm -o - %s -verify-machineinstrs=0 | FileCheck %s -check-prefix CHECK-X64
; RUN: llc -mtriple x86_64-linux -exception-model sjlj -filetype asm -o - %s -verify-machineinstrs=0 | FileCheck %s -check-prefix CHECK-X64-LINUX

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
; +04   uint32_t __callsite;                    -60(%ebp)
; +08   uint32_t __buffer[4];                   -56(%ebp)
; +24   __personality_routine __personality;    -40(%ebp)
; +28   uintptr_t __lsda;                       -36(%ebp)
; +32   ptr__jbuf[];                         -32(%ebp)
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
; +00   struct _Unwind_FunctionContext *prev;   -312(%rbp)
; +08   uint32_t __callsite;                    -304(%rbp)
; +12   uint32_t __buffer[4];                   -300(%rbp)
; +32   __personality_routine __personality;    -280(%rbp)
; +40   uintptr_t __lsda;                       -272(%rbp)
; +48   ptr__jbuf[];                         -264(%rbp)
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
; CHECK-X64: leaq -312(%rbp), %rcx
; CHECK-X64: callq _Unwind_SjLj_Register
;     function_that_throws();
; CHECK-X64: movl $1, -304(%rbp)
; CHECK-X64: callq _Z20function_that_throwsv
;     _Unwind_SjLj_Unregister(&UFC);
; CHECK-X64: leaq -312(%rbp), %rcx
; CHECK-X64: callq _Unwind_SjLj_Unregister
;
; CHECK-X64: [[RESUME]]:
;     assert(UFC.__callsite < 1);
; CHECK-X64: movl -304(%rbp), %eax
; CHECK-X64: cmpl $1, %eax
; CHECK-X64: jb .[[CONT:LBB[0-9]+_[0-9]+]]
; CHECK-X64: ud2
; CHECK-X64: [[CONT]]:
;     *Handlers[UFC.__callsite]
; CHECK-X64: leaq .[[TABLE:LJTI[0-9]+_[0-9]+]](%rip), %rcx
; CHECK-X64: movl (%rcx,%rax,4), %eax
; CHECK-X64: cltq
; CHECK-X64: addq %rcx, %rax
; CHECK-X64: jmpq *%rax

; CHECK-X64-LINUX: .[[RESUME:LBB[0-9]+_[0-9]+]]:
;     assert(UFC.__callsite < 1);
; CHECK-X64-LINUX: movl -120(%rbp), %eax
; CHECK-X64-LINUX: cmpl $1, %eax
; CHECK-X64-LINUX: jb .[[CONT:LBB[0-9]+_[0-9]+]]
; CHECK-X64-LINUX: ud2
; CHECK-X64-LINUX: [[CONT]]:
;     *Handlers[UFC.__callsite]
; CHECK-X64-LINUX: leaq .[[TABLE:LJTI[0-9]+_[0-9]+]](%rip), %rcx
; CHECK-X64-LINUX: jmpq *(%rcx,%rax,8)
