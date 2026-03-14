; RUN: not llc -mcpu=pwr7 -o /dev/null %s 2>&1 | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define zeroext i1 @testi1(i1 zeroext %b1, i1 zeroext %b2) #0 {
entry:
  %0 = tail call i8 asm "crand $0, $1, $2", "=^wc,^wc,^wc"(i1 %b1, i1 %b2) #0
  %1 = and i8 %0, 1
  %tobool3 = icmp ne i8 %1, 0
  ret i1 %tobool3

; CHECK: error: couldn't allocate output register for constraint 'wc'
}

attributes #0 = { nounwind "target-features"="-crbits" }

