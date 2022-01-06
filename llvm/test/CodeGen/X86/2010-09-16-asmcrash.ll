; RUN: llc < %s -mtriple=x86_64-unknown-freebsd8.1 -o /dev/null
; This formerly crashed, PR 8154.

module asm ".weak sem_close"
module asm ".equ sem_close, _sem_close"
module asm ".weak sem_destroy"
module asm ".equ sem_destroy, _sem_destroy"
module asm ".weak sem_getvalue"
module asm ".equ sem_getvalue, _sem_getvalue"
module asm ".weak sem_init"
module asm ".equ sem_init, _sem_init"
module asm ".weak sem_open"
module asm ".equ sem_open, _sem_open"
module asm ".weak sem_post"
module asm ".equ sem_post, _sem_post"
module asm ".weak sem_timedwait"
module asm ".equ sem_timedwait, _sem_timedwait"
module asm ".weak sem_trywait"
module asm ".equ sem_trywait, _sem_trywait"
module asm ".weak sem_unlink"
module asm ".equ sem_unlink, _sem_unlink"
module asm ".weak sem_wait"
module asm ".equ sem_wait, _sem_wait"

%struct._sem = type { i32, %struct._usem }
%struct._usem = type { i32, i32, i32 }

define void @_sem_timedwait(%struct._sem* noalias %sem) nounwind ssp {
entry:
  br i1 undef, label %while.cond.preheader, label %sem_check_validity.exit

while.cond.preheader:                             ; preds = %entry
  %tmp4 = getelementptr inbounds %struct._sem, %struct._sem* %sem, i64 0, i32 1, i32 1
  br label %while.cond

sem_check_validity.exit:                          ; preds = %entry
  ret void

while.cond:                                       ; preds = %while.body, %while.cond.preheader
  br i1 undef, label %while.body, label %while.end

while.body:                                       ; preds = %while.cond
  %0 = call i8 asm sideeffect "\09lock ; \09\09\09cmpxchgl $2,$1 ;\09       sete\09$0 ;\09\091:\09\09\09\09# atomic_cmpset_int", "={ax},=*m,r,{ax},*m,~{memory},~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) %tmp4, i32 undef, i32 undef, i32* elementtype(i32) %tmp4) nounwind, !srcloc !0
  br i1 undef, label %while.cond, label %return

while.end:                                        ; preds = %while.cond
  br i1 undef, label %if.end18, label %return

if.end18:                                         ; preds = %while.end
  unreachable

return:                                           ; preds = %while.end, %while.body
  ret void
}

!0 = !{i32 158484}
