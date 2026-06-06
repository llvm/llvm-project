; RUN: llc %s -mtriple=mipsel -mattr=micromips -filetype=asm \
; RUN: -relocation-model=pic -O3 -o - | FileCheck %s

define i32 @sum(ptr %x, ptr %y) nounwind uwtable {
entry:
  %x.addr = alloca ptr, align 8
  %y.addr = alloca ptr, align 8
  store ptr %x, ptr %x.addr, align 8
  store ptr %y, ptr %y.addr, align 8
  %0 = load ptr, ptr %x.addr, align 8
  %1 = load i32, ptr %0, align 4
  %2 = load ptr, ptr %y.addr, align 8
  %3 = load i32, ptr %2, align 4
  %add = add nsw i32 %1, %3
  ret i32 %add
}

define i32 @main() nounwind uwtable {
entry:
  %retval = alloca i32, align 4
  %x = alloca i32, align 4
  %y = alloca i32, align 4
  store i32 0, ptr %retval
  %call = call i32 @sum(ptr %x, ptr %y)
  ret i32 %call
}

; CHECK: addiu ${{[0-9]+}}, $sp, {{[0-9]+}}
; CHECK: addiu ${{[0-9]+}}, $sp, {{[0-9]+}}
