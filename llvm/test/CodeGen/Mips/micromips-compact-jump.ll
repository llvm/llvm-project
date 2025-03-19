; RUN: llc -mtriple=mipsel -mcpu=mips32r2 -mattr=+micromips \
; RUN:   -disable-mips-delay-filler -O3 < %s | FileCheck %s

define i32 @foo(i32 signext %a) #0 {
entry:
  ret i32 0
}

declare i32 @bar(i32 signext) #1

; CHECK:      jrc
