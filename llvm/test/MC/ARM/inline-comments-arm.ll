; RUN: llc %s -o - | sed -n -e '/@APP/,/@NO_APP/p' > %t
; RUN: sed -n -e 's/^;CHECK://p' %s > %t2
; RUN: diff %t %t2

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "arm-eabi"

; Function Attrs: nounwind uwtable
define void @foo() #0 {
entry:
  call void asm sideeffect "#isolated preprocessor comment", "~{dirflag},~{fpsr},~{flags}"() #0
;CHECK:	@APP
;CHECK:	@isolated preprocessor comment
;CHECK:	@NO_APP
  ret void
}

attributes #0 = { nounwind }

!llvm.ident = !{!0}

!0 = !{!""}
