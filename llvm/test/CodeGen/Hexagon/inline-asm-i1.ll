; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: r[[REG0:[0-9]+]] = usr
; CHECK: [[REG0]] = insert(r{{[0-9]+}},#1,#16)

target triple = "hexagon"

define hidden void @fred() #0 {
entry:
  %0 = call { i32, i32 } asm sideeffect " $0 = usr\0A $1 = $2\0A $0 = insert($1, #1, #16)\0Ausr = $0 \0A", "=&r,=&r,r"(i1 undef) #1
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv60" }
attributes #1 = { nounwind }
