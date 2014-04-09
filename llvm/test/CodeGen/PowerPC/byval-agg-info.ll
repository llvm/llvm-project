; RUN: llc < %s -print-after=prologepilog >%t 2>&1 && FileCheck <%t %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

%struct.anon = type { i32, i32 }

declare void @foo(%struct.anon* %v)
define void @test(i32 %a, i32 %b, %struct.anon* byval nocapture %v) {
entry:
  call void @foo(%struct.anon* %v)
  ret void
}

; Make sure that the MMO on the store has no offset from the byval
; variable itself (we used to have mem:ST8[%v+64]).
; CHECK: STD %X5<kill>, 176, %X1; mem:ST8[%v](align=16)

