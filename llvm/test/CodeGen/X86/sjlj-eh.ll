; RUN: llc -mtriple i386-windows-gnu -exception-model sjlj -filetype asm -o - %s | FileCheck %s

declare void @_Z20function_that_throwsv()
declare i32 @__gxx_personality_sj0(...)
declare i8* @__cxa_begin_catch(i8*)
declare void @__cxa_end_catch()

define void @_Z8functionv() personality i8* bitcast (i32 (...)* @__gxx_personality_sj0 to i8*) {
entry:
  invoke void @_Z20function_that_throwsv()
          to label %try.cont unwind label %lpad

lpad:
  %0 = landingpad { i8*, i32 }
          catch i8* null
  %1 = extractvalue { i8*, i32 } %0, 0
  %2 = tail call i8* @__cxa_begin_catch(i8* %1)
  tail call void @__cxa_end_catch()
  br label %try.cont

try.cont:
  ret void
}

;     struct _Unwind_FunctionContext {
; +00   struct _Unwind_FunctionContext *prev;   -64(%ebp)
; +04   uintptr_t __callsite;                   -60(%ebp)
; +08   uintptr_t __buffer[4];                  -44(%ebp)
; +28   __personality_routine __personality;    -40(%ebp)
; +32   uintptr_t __lsda;                       -36(%ebp)
; +36   void *__jbuf[];                         -32(%ebp)
;     };


; CHECK-LABEL: __Z8functionv:
;     struct _Unwind_FunctionContext UFC;
;
;     UFC.__personality = __gxx_personality_sj0
; CHECK: movl $___gxx_personality_sj0, -40(%ebp)
;     UFC.__lsda = $LSDA
; CHECK: movl $[[LSDA:GCC_except_table[0-9]+]], -36(%ebp)
;     UFC.__jbuf[0] = $EBP
; CHECK: movl %ebp, -32(%ebp)
;     UFC.__jbuf[2] = $ESP
; CHECK: movl %esp, -24(%ebp)
;     UFC.__jbuf[1] = $EIP
; CHECK: movl $[[RESUME:LBB[0-9]+_[0-9]+]], -28(%ebp)
;     UFC.__callsite = 1
; CHECK: movl $1, -60(%ebp)
;     _Unwind_SjLj_Register(&UFC);
; CHECK: leal -64(%ebp), %eax
; CHECK: pushl %eax
; CHECK: calll __Unwind_SjLj_Register
; CHECK: addl $4, %esp
;     function_that_throws();
; CHECK: calll __Z20function_that_throwsv
;     _Unwind_SjLj_Unregister(&UFC);
; CHECK: leal -64(%ebp), %eax
; CHECK: calll __Unwind_SjLj_Unregister
;
; CHECK: [[RESUME]]:
; CHECK: leal -64(%ebp), %esi
;     assert(UFC.__callsite <= 1);
; CHECK: movl -60(%ebp), %eax
; CHECK: cmpl $1, %eax
; CHECK: jbe [[CONT:LBB[0-9]+_[0-9]+]]
; CHECK: ud2
; CHECK: [[CONT]]:
;     *Handlers[--UFC.__callsite]
; CHECK: subl $1, %eax
; CHECK: jmpl *LJTI

