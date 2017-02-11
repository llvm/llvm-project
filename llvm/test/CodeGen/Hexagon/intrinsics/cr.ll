; RUN: llc -march=hexagon < %s | FileCheck %s
; RUN: llc -march=hexagon -O0 < %s | FileCheck -check-prefix=CHECK-CALL %s
; Hexagon Programmer's Reference Manual 11.2 CR

; CHECK-CALL-NOT: call

; Corner detection acceleration
declare i32 @llvm.hexagon.C4.fastcorner9(i32, i32)
define i32 @C4_fastcorner9(i32 %a, i32 %b) {
  %z = call i32@llvm.hexagon.C4.fastcorner9(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: = fastcorner9({{.*}},{{.*}})

declare i32 @llvm.hexagon.C4.fastcorner9.not(i32, i32)
define i32 @C4_fastcorner9_not(i32 %a, i32 %b) {
  %z = call i32@llvm.hexagon.C4.fastcorner9.not(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: = !fastcorner9({{.*}},{{.*}})

; Logical reductions on predicates
declare i32 @llvm.hexagon.C2.any8(i32)
define i32 @C2_any8(i32 %a) {
  %z = call i32@llvm.hexagon.C2.any8(i32 %a)
  ret i32 %z
}
; CHECK: = any8({{.*}})

declare i32 @llvm.hexagon.C2.all8(i32)
define i32 @C2_all8(i32 %a) {
  %z = call i32@llvm.hexagon.C2.all8(i32 %a)
  ret i32 %z
}

; CHECK: = all8({{.*}})

; Logical operations on predicates
declare i32 @llvm.hexagon.C2.and(i32, i32)
define i32 @C2_and(i32 %a, i32 %b) {
  %z = call i32@llvm.hexagon.C2.and(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: = and({{.*}},{{.*}})

declare i32 @llvm.hexagon.C4.and.and(i32, i32, i32)
define i32 @C4_and_and(i32 %a, i32 %b, i32 %c) {
  %z = call i32@llvm.hexagon.C4.and.and(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: = and({{.*}},and({{.*}},{{.*}}))

declare i32 @llvm.hexagon.C2.or(i32, i32)
define i32 @C2_or(i32 %a, i32 %b) {
  %z = call i32@llvm.hexagon.C2.or(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: = or({{.*}},{{.*}})

declare i32 @llvm.hexagon.C4.and.or(i32, i32, i32)
define i32 @C4_and_or(i32 %a, i32 %b, i32 %c) {
  %z = call i32@llvm.hexagon.C4.and.or(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: = and({{.*}},or({{.*}},{{.*}}))

declare i32 @llvm.hexagon.C2.xor(i32, i32)
define i32 @C2_xor(i32 %a, i32 %b) {
  %z = call i32@llvm.hexagon.C2.xor(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: = xor({{.*}},{{.*}})

declare i32 @llvm.hexagon.C4.or.and(i32, i32, i32)
define i32 @C4_or_and(i32 %a, i32 %b, i32 %c) {
  %z = call i32@llvm.hexagon.C4.or.and(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: = or({{.*}},and({{.*}},{{.*}}))

declare i32 @llvm.hexagon.C2.andn(i32, i32)
define i32 @C2_andn(i32 %a, i32 %b) {
  %z = call i32@llvm.hexagon.C2.andn(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: = and({{.*}},!{{.*}})

declare i32 @llvm.hexagon.C4.or.or(i32, i32, i32)
define i32 @C4_or_or(i32 %a, i32 %b, i32 %c) {
  %z = call i32@llvm.hexagon.C4.or.or(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: = or({{.*}},or({{.*}},{{.*}}))

declare i32 @llvm.hexagon.C4.and.andn(i32, i32, i32)
define i32 @C4_and_andn(i32 %a, i32 %b, i32 %c) {
  %z = call i32@llvm.hexagon.C4.and.andn(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: = and({{.*}},and({{.*}},!{{.*}}))

declare i32 @llvm.hexagon.C4.and.orn(i32, i32, i32)
define i32 @C4_and_orn(i32 %a, i32 %b, i32 %c) {
  %z = call i32@llvm.hexagon.C4.and.orn(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: = and({{.*}},or({{.*}},!{{.*}}))

declare i32 @llvm.hexagon.C2.not(i32)
define i32 @C2_not(i32 %a) {
  %z = call i32@llvm.hexagon.C2.not(i32 %a)
  ret i32 %z
}
; CHECK: = not({{.*}})

declare i32 @llvm.hexagon.C4.or.andn(i32, i32, i32)
define i32 @C4_or_andn(i32 %a, i32 %b, i32 %c) {
  %z = call i32@llvm.hexagon.C4.or.andn(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: = or({{.*}},and({{.*}},!{{.*}}))

declare i32 @llvm.hexagon.C2.orn(i32, i32)
define i32 @C2_orn(i32 %a, i32 %b) {
  %z = call i32@llvm.hexagon.C2.orn(i32 %a, i32 %b)
  ret i32 %z
}
; CHECK: = or({{.*}},!{{.*}})

declare i32 @llvm.hexagon.C4.or.orn(i32, i32, i32)
define i32 @C4_or_orn(i32 %a, i32 %b, i32 %c) {
  %z = call i32@llvm.hexagon.C4.or.orn(i32 %a, i32 %b, i32 %c)
  ret i32 %z
}
; CHECK: = or({{.*}},or({{.*}},!{{.*}}))
