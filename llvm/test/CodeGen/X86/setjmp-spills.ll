; RUN: llc < %s -mtriple=i386-linux | FileCheck %s -check-prefix=X86-32
; RUN: llc < %s -mtriple=x86_64-linux | FileCheck %s -check-prefix=X86-64

declare i32 @get_val()
declare void @use_val(i32)
declare i1 @setjmp()
declare void @longjmp()
declare void @personality()


; Test that llc avoids reusing spill slots in functions that call
; setjmp(), whether they use "call" or "invoke" for calling setjmp()
; (PR18244).

define void @setjmp_caller() {
; X86-32-LABEL: setjmp_caller:
; X86-64-LABEL: setjmp_caller:
; This code keeps enough variables live across the setjmp() call that
; they don't all fit in registers and the compiler will allocate a
; spill slot.
  %a1 = call i32 @get_val()
  %a2 = call i32 @get_val()
  %a3 = call i32 @get_val()
  %a4 = call i32 @get_val()
  %a5 = call i32 @get_val()
  %a6 = call i32 @get_val()
  %a7 = call i32 @get_val()
  %a8 = call i32 @get_val()
; X86-32: movl %eax, [[SPILL_SLOT:[0-9]+]](%esp)
; X86-32: calll get_val
; X86-64: movl %eax, [[SPILL_SLOT:[0-9]+]](%rsp)
; X86-64: callq get_val

  %setjmp_result = call i1 @setjmp() returns_twice
  br i1 %setjmp_result, label %second, label %first
; X86-32: calll setjmp
; X86-64: callq setjmp

; Again, keep enough variables live that they need spill slots.  Since
; this function calls a returns_twice function (setjmp()), the
; compiler should not reuse the spill slots.  longjmp() can return to
; where the first spill slots were still live.
first:
  %b1 = call i32 @get_val()
  %b2 = call i32 @get_val()
  %b3 = call i32 @get_val()
  %b4 = call i32 @get_val()
  %b5 = call i32 @get_val()
  %b6 = call i32 @get_val()
  %b7 = call i32 @get_val()
  %b8 = call i32 @get_val()
  call void @use_val(i32 %b1)
  call void @use_val(i32 %b2)
  call void @use_val(i32 %b3)
  call void @use_val(i32 %b4)
  call void @use_val(i32 %b5)
  call void @use_val(i32 %b6)
  call void @use_val(i32 %b7)
  call void @use_val(i32 %b8)
  call void @longjmp()
  unreachable
; X86-32-NOT: movl {{.*}}, [[SPILL_SLOT]](%esp)
; X86-64-NOT: movl {{.*}}, [[SPILL_SLOT]](%rsp)

second:
  call void @use_val(i32 %a1)
  call void @use_val(i32 %a2)
  call void @use_val(i32 %a3)
  call void @use_val(i32 %a4)
  call void @use_val(i32 %a5)
  call void @use_val(i32 %a6)
  call void @use_val(i32 %a7)
  call void @use_val(i32 %a8)
  ret void
}


; This is the same as above, but using "invoke" rather than "call" to
; call setjmp().

define void @setjmp_invoker() personality ptr @personality {
; X86-32-LABEL: setjmp_invoker:
; X86-64-LABEL: setjmp_invoker:
  %a1 = call i32 @get_val()
  %a2 = call i32 @get_val()
  %a3 = call i32 @get_val()
  %a4 = call i32 @get_val()
  %a5 = call i32 @get_val()
  %a6 = call i32 @get_val()
  %a7 = call i32 @get_val()
  %a8 = call i32 @get_val()
; X86-32: movl %eax, [[SPILL_SLOT:[0-9]+]](%esp)
; X86-32: calll get_val
; X86-64: movl %eax, [[SPILL_SLOT:[0-9]+]](%rsp)
; X86-64: callq get_val

  %setjmp_result = invoke i1 @setjmp() returns_twice
      to label %cont unwind label %lpad
; X86-32: calll setjmp
; X86-64: callq setjmp

cont:
  br i1 %setjmp_result, label %second, label %first

lpad:
  %lp = landingpad { ptr, i32 } cleanup
  unreachable

first:
  %b1 = call i32 @get_val()
  %b2 = call i32 @get_val()
  %b3 = call i32 @get_val()
  %b4 = call i32 @get_val()
  %b5 = call i32 @get_val()
  %b6 = call i32 @get_val()
  %b7 = call i32 @get_val()
  %b8 = call i32 @get_val()
  call void @use_val(i32 %b1)
  call void @use_val(i32 %b2)
  call void @use_val(i32 %b3)
  call void @use_val(i32 %b4)
  call void @use_val(i32 %b5)
  call void @use_val(i32 %b6)
  call void @use_val(i32 %b7)
  call void @use_val(i32 %b8)
  call void @longjmp()
  unreachable
; X86-32-NOT: movl {{.*}}, [[SPILL_SLOT]](%esp)
; X86-64-NOT: movl {{.*}}, [[SPILL_SLOT]](%rsp)

second:
  call void @use_val(i32 %a1)
  call void @use_val(i32 %a2)
  call void @use_val(i32 %a3)
  call void @use_val(i32 %a4)
  call void @use_val(i32 %a5)
  call void @use_val(i32 %a6)
  call void @use_val(i32 %a7)
  call void @use_val(i32 %a8)
  ret void
}
