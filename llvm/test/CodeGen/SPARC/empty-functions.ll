; RUN: llc < %s -mtriple=sparc-linux-gnu | FileCheck -check-prefix=LINUX-NO-FP %s
; RUN: llc < %s -mtriple=sparc-linux-gnu -disable-fp-elim | FileCheck -check-prefix=LINUX-FP %s

define void @func() {
entry:
  unreachable
}

; An empty function is perfectly fine on ELF.
; LINUX-NO-FP: func:
; LINUX-NO-FP-NEXT: .cfi_startproc
; LINUX-NO-FP-NEXT: {{^}}!
; LINUX-NO-FP-NEXT: {{^}}.L{{.*}}:{{$}}
; LINUX-NO-FP-NEXT: .size   func, .L{{.*}}-func
; LINUX-NO-FP-NEXT: .cfi_endproc

; A cfi directive cannot point to the end of a function.
; LINUX-FP: func:
; LINUX-FP-NEXT: .cfi_startproc
; LINUX-FP-NEXT: {{^}}!
; LINUX-FP-NEXT: save %sp, -96, %sp
; LINUX-FP-NEXT: {{^}}.L{{.*}}:{{$}}
; LINUX-FP-NEXT: .size   func, .Lfunc_end0-func
; LINUX-FP-NEXT: .cfi_endproc
