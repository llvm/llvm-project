; RUN: llc %s -march=mipsel -mattr=micromips -filetype=asm -O3 \
; RUN: -disable-mips-delay-filler -relocation-model=pic -o - | FileCheck %s

define void @main() nounwind uwtable {
entry:
  %x = alloca i32, align 4
  %0 = load i32, i32* %x, align 4
  %cmp = icmp eq i32 %0, 0
  br i1 %cmp, label %if.then, label %if.end, !prof !1

if.then:
  store i32 10, i32* %x, align 4
  br label %if.end

if.end:
  ret void
}

; CHECK: bnezc
!1 = !{!"branch_weights", i32 2, i32 1}
