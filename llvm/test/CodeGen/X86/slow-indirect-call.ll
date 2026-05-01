; RUN: llc < %s -mtriple=x86_64-linux -mattr=+slow-indirect-call | FileCheck %s --check-prefix=SLOW
; RUN: llc < %s -mtriple=x86_64-linux -mattr=-slow-indirect-call | FileCheck %s --check-prefix=NON-SLOW
; RUN: llc < %s -mtriple=x86_64-linux -mcpu=znver5 | FileCheck %s --check-prefix=SLOW
; RUN: llc < %s -mtriple=x86_64-linux -mcpu=znver4 | FileCheck %s --check-prefix=NON-SLOW
; RUN: llc < %s -mtriple=i686-linux -mattr=+slow-indirect-call | FileCheck %s --check-prefix=SLOW32
; RUN: llc < %s -mtriple=i686-linux -mattr=-slow-indirect-call | FileCheck %s --check-prefix=NON-SLOW32

@vtable = external dso_local global ptr

; Indirect call through a global function pointer. With slow-indirect-call the
; load must not be folded into the call.
define i32 @test_call_global() nounwind {
; SLOW-LABEL: test_call_global:
; SLOW:       movq vtable(%rip), %rax
; SLOW:       callq *%rax
;
; NON-SLOW-LABEL: test_call_global:
; NON-SLOW:       callq *vtable(%rip)
;
; SLOW32-LABEL: test_call_global:
; SLOW32:       movl vtable, %eax
; SLOW32:       calll *%eax
;
; NON-SLOW32-LABEL: test_call_global:
; NON-SLOW32:       calll *vtable
entry:
  %fp = load ptr, ptr @vtable, align 8
  %ret = call i32 %fp(i32 42)
  ret i32 %ret
}

; Vtable dispatch: load vtable pointer, load function from vtable, call.
; The second load (vtable slot) must not be folded into the call.
define i32 @test_call_vtable(ptr %obj) nounwind {
; SLOW-LABEL: test_call_vtable:
; SLOW:       movq (%rdi), %rax
; SLOW:       movq (%rax), %rax
; SLOW:       callq *%rax
;
; NON-SLOW-LABEL: test_call_vtable:
; NON-SLOW:       movq (%rdi), %rax
; NON-SLOW:       callq *(%rax)
;
; SLOW32-LABEL: test_call_vtable:
; SLOW32-NOT:   calll *(%{{.*}})
; SLOW32:       calll *%{{.*}}
;
; NON-SLOW32-LABEL: test_call_vtable:
; NON-SLOW32:       calll *(%{{.*}})
entry:
  %vt = load ptr, ptr %obj, align 8
  %fp = load ptr, ptr %vt, align 8
  %ret = call i32 %fp(ptr %obj)
  ret i32 %ret
}

; With minsize, the load should be folded even with slow-indirect-call.
; The minsize attribute overrides the slow-indirect-call guard during
; register allocation folding.
define i32 @test_call_minsize() nounwind minsize {
; SLOW-LABEL: test_call_minsize:
; SLOW:       callq *vtable(%rip)
;
; NON-SLOW-LABEL: test_call_minsize:
; NON-SLOW:       callq *vtable(%rip)
entry:
  %fp = load ptr, ptr @vtable, align 8
  %ret = call i32 %fp(i32 42)
  ret i32 %ret
}

; Tail calls should not be affected by slow-indirect-call. The load into the
; jump target should still be folded.
define void @test_tail_call(ptr %obj) nounwind {
; SLOW-LABEL: test_tail_call:
; SLOW:       movq (%rdi), %rax
; SLOW:       jmpq *(%rax)
;
; NON-SLOW-LABEL: test_tail_call:
; NON-SLOW:       movq (%rdi), %rax
; NON-SLOW:       jmpq *(%rax)
entry:
  %vt = load ptr, ptr %obj, align 8
  %fp = load ptr, ptr %vt, align 8
  musttail call void %fp(ptr %obj)
  ret void
}
