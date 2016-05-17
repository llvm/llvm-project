; RUN: not llc -mcpu=pwr7 -o /dev/null %s 2>&1 | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define <4 x i32> @testi1(<4 x i32> %b1, <4 x i32> %b2) #0 {
entry:
  %0 = call <4 x i32> asm "xxland $0, $1, $2", "=^wd,^wd,^wd"(<4 x i32> %b1, <4 x i32> %b2) #0
  ret <4 x i32> %0

; CHECK: error: couldn't allocate output register for constraint 'wd'
}

attributes #0 = { nounwind "target-features"="-vsx" }

