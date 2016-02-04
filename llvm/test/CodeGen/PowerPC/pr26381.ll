; RUN: llc -mcpu=pwr7 -mtriple=powerpc64le-unknown-unknown -O0 < %s | FileCheck %s

define internal signext i32 @foo() #0 {
  ret i32 -125452974
}

; CHECK: lis 3, -1915
; CHECK: ori 3, 3, 48466
