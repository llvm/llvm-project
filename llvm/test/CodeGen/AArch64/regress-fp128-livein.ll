; RUN: llc -mtriple=aarch64-linux-gnu -verify-machineinstrs -o - %s

; Regression test for NZCV reg live-in not being added to fp128csel IfTrue BB,
; causing a crash during live range calc.
define void @fp128_livein(i64 %a) {
  %tobool = icmp ne i64 %a, 0
  %conv = zext i1 %tobool to i32
  %conv2 = sitofp i32 %conv to fp128
  %conv6 = sitofp i32 %conv to double
  %call3 = tail call i32 @g(fp128 %conv2)
  %call8 = tail call i32 @h(double %conv6)
  ret void
}

declare i32 @f()
declare i32 @g(fp128)
declare i32 @h(double)
