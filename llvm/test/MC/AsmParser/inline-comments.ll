; RUN: llc %s -o - | sed -n -e '/#APP/,/#NO_APP/p' > %t
; RUN: sed -n -e 's/^;CHECK://p' %s > %t2
; RUN: diff %t %t2

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define void @foo() #0 {
entry:
  call void asm sideeffect "", "~{dirflag},~{fpsr},~{flags}"() #0
;CHECK:	#APP
;CHECK:	#NO_APP
  call void asm sideeffect " ", "~{dirflag},~{fpsr},~{flags}"() #0
;CHECK:	#APP
;CHECK:
;CHECK:	#NO_APP
  call void asm sideeffect "\0A", "~{dirflag},~{fpsr},~{flags}"() #0
;CHECK:	#APP
;CHECK:
;CHECK:
;CHECK:	#NO_APP
  call void asm sideeffect "/*isolated c comment*/", "~{dirflag},~{fpsr},~{flags}"() #0
;CHECK:	#APP
;CHECK:	#isolated c comment
;CHECK:	#NO_APP
  call void asm sideeffect "/**/", "~{dirflag},~{fpsr},~{flags}"() #0
;CHECK:	#APP
;CHECK:	#
;CHECK:	#NO_APP
  call void asm sideeffect "/*comment with\0Anewline*/", "~{dirflag},~{fpsr},~{flags}"() #0
;CHECK:	#APP
;CHECK:	#comment with
;CHECK:	#newline
;CHECK:	#NO_APP
  call void asm sideeffect "//isolated line comment", "~{dirflag},~{fpsr},~{flags}"() #0
;CHECK:	#APP
;CHECK:	#isolated line comment
;CHECK:	#NO_APP
  call void asm sideeffect "#isolated line comment", "~{dirflag},~{fpsr},~{flags}"() #0
;CHECK:	#APP
;CHECK:	#isolated line comment
;CHECK:	#NO_APP
   call void asm sideeffect "nop /* after nop */", "~{dirflag},~{fpsr},~{flags}"() #0
;CHECK:	#APP
;CHECK:	nop	# after nop 
;CHECK:	#NO_APP
  call void asm sideeffect "nop // after nop", "~{dirflag},~{fpsr},~{flags}"() #0
;CHECK:	#APP
;CHECK:	nop	# after nop
;CHECK:	#NO_APP
  call void asm sideeffect "nop # after nop", "~{dirflag},~{fpsr},~{flags}"() #0
;CHECK:	#APP
;CHECK:	nop	# after nop
;CHECK:	#NO_APP
  call void asm sideeffect "nop /* after explicit ended nop */", "~{dirflag},~{fpsr},~{flags}"() #0
;CHECK:	#APP
;CHECK:	nop	# after explicit ended nop 
;CHECK:	#NO_APP
  call void asm sideeffect "nop # after explicit ended nop", "~{dirflag},~{fpsr},~{flags}"() #0
;CHECK:	#APP
;CHECK:	nop	# after explicit ended nop
;CHECK:	#NO_APP
  call void asm sideeffect "nop # after explicit end nop", "~{dirflag},~{fpsr},~{flags}"() #0
;CHECK:	#APP
;CHECK:	nop	# after explicit end nop
;CHECK:	#NO_APP
  call void asm sideeffect "/* before nop */ nop", "~{dirflag},~{fpsr},~{flags}"() #0
;CHECK:	#APP
;CHECK:	nop	# before nop 
;CHECK:	#NO_APP
  call void asm sideeffect "//comment with escaped newline\0A", "~{dirflag},~{fpsr},~{flags}"() #0
;CHECK:	#APP
;CHECK:	#comment with escaped newline
;CHECK:
;CHECK:	#NO_APP
  call void asm sideeffect "/*0*/xor/*1*/%eax,/*2*/%ecx/*3*///eol", "~{dirflag},~{fpsr},~{flags}"() #0
;CHECK:	#APP
;CHECK:	xorl	%eax, %ecx	#0	#1	#2	#3	#eol
;CHECK:	#NO_APP
  ret void
}

attributes #0 = { nounwind }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.9.0 (trunk 268625) (llvm/trunk 268631)"}
