; RUN: llc %s -o - -O0 -regalloc=fast | FileCheck %s
target datalayout = "e-m:o-p:32:32-f64:32:64-f80:128-n8:16:32-S128"
target triple = "i386-apple-macosx10.10"

@c = external global i8, align 1
@p = thread_local global i8* null, align 4

; Check that regalloc fast correctly preserves EAX that is set by the TLS call
; until the actual use.
; PR26485.
;
; CHECK-LABEL: f:
; Get p.
; CHECK: movl _p@{{[0-9a-zA-Z]+}}, [[P_ADDR:%[a-z]+]]
; CHECK-NEXT: calll *([[P_ADDR]])
; At this point eax contiains the address of p.
; Load c address.
; Make sure we do not clobber eax.
; CHECK-NEXT: movl L_c{{[^,]*}}, [[C_ADDR:%e[b-z]x+]]
; Store c address into p.
; CHECK-NEXT: movl [[C_ADDR]], (%eax)
define void @f() #0 {
entry:
  store i8* @c, i8** @p, align 4
  ret void
}
