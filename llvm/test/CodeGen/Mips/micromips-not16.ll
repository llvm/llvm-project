; RUN: llc -march=mipsel -mcpu=mips32r2 -mattr=+micromips \
; RUN:   -relocation-model=pic -O3 < %s | FileCheck %s

define i32 @main() {
entry:
  %retval = alloca i32, align 4
  %x = alloca i64, align 8
  store i32 0, ptr %retval
  %0 = load i64, ptr %x, align 8
  %cmp = icmp ne i64 %0, 9223372036854775807
  br i1 %cmp, label %if.then, label %if.end

if.then:
  store i32 1, ptr %retval
  br label %return

if.end:
  store i32 0, ptr %retval
  br label %return

return:
  %1 = load i32, ptr %retval
  ret i32 %1
}

; CHECK: not16
