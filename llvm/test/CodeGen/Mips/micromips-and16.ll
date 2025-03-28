; RUN: llc -mtriple=mipsel -mcpu=mips32r2 -mattr=+micromips \
; RUN:   -relocation-model=pic -O3 < %s | FileCheck %s

define i32 @main() {
entry:
  %retval = alloca i32, align 4
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %c = alloca i32, align 4
  store i32 0, ptr %retval
  %0 = load i32, ptr %b, align 4
  %1 = load i32, ptr %c, align 4
  %and = and i32 %0, %1
  store i32 %and, ptr %a, align 4
  ret i32 0
}

; CHECK: and16
