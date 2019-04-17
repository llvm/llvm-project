; RUN: opt < %s -internalize -internalize-public-api-list c1 -internalize-public-api-list c2 -internalize-public-api-list c3 -internalize-public-api-list c4 -S | FileCheck %s

$c1 = comdat any
$c2 = comdat any
$c3 = comdat any
$c4 = comdat any

; CHECK: @c1_c = global i32 0, comdat($c1)
@c1_c = global i32 0, comdat($c1)

; CHECK: @c2_b = internal global i32 0{{$}}
@c2_b = global i32 0, comdat($c2)

; CHECK: @c3 = global i32 0, comdat{{$}}
@c3 = global i32 0, comdat

; CHECK: @c4_a = internal global i32 0, comdat($c4)
@c4_a = internal global i32 0, comdat($c4)

; CHECK: @c1_d = alias i32, i32* @c1_c
@c1_d = alias i32, i32* @c1_c

; CHECK: @c2_c = internal alias i32, i32* @c2_b
@c2_c = alias i32, i32* @c2_b

; CHECK: @c4 = alias i32, i32* @c4_a
@c4 = alias i32, i32* @c4_a

; CHECK: define void @c1() comdat {
define void @c1() comdat {
  ret void
}

; CHECK: define void @c1_a() comdat($c1) {
define void @c1_a() comdat($c1) {
  ret void
}

; CHECK: define internal void @c2() {
define internal void @c2() comdat {
  ret void
}

; CHECK: define internal void @c2_a() {
define void @c2_a() comdat($c2) {
  ret void
}

; CHECK: define void @c3_a() comdat($c3) {
define void @c3_a() comdat($c3) {
  ret void
}
