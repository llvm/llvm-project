; RUN: opt < %s -mtriple=x86_64 -internalize -internalize-public-api-list main -internalize-public-api-list c1 -internalize-public-api-list c2 \
; RUN:   -internalize-public-api-list c3 -internalize-public-api-list c4 -S | FileCheck %s --check-prefixes=CHECK,ELF
; RUN: opt < %s -mtriple=x86_64-windows -internalize -internalize-public-api-list main -internalize-public-api-list c1 -internalize-public-api-list c2 \
; RUN:   -internalize-public-api-list c3 -internalize-public-api-list c4 -S | FileCheck %s --check-prefixes=CHECK,COFF

$c1 = comdat any
$c2 = comdat any
$c3 = comdat any
$c4 = comdat any
$c5 = comdat any

; ELF:   $c2.[[MODULEID:[0-9a-f]+]] = comdat any
; COFF:  $c2 = comdat any

; CHECK: @c1_c = global i32 0, comdat($c1)
@c1_c = global i32 0, comdat($c1)

;; $c2 has more than one member. Keep the comdat.
; ELF:   @c2_b = internal global i32 0, comdat($c2.[[MODULEID]])
; COFF:  @c2_b = internal global i32 0, comdat($c2)
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

; ELF:   define internal void @c2() comdat($c2.[[MODULEID]]) {
; COFF:  define internal void @c2() comdat {
define internal void @c2() comdat {
  ret void
}

; ELF:   define internal void @c2_a() comdat($c2.[[MODULEID]]) {
; COFF:  define internal void @c2_a() comdat($c2) {
define void @c2_a() comdat($c2) {
  ret void
}

; CHECK: define void @c3_a() comdat($c3) {
define void @c3_a() comdat($c3) {
  ret void
}

;; There is only one member in the comdat. Delete the comdat as a size optimization.
; CHECK: define internal void @c5() {
define void @c5() comdat($c5) {
  ret void
}

;; Add a non-comdat symbol to ensure the module ID is not empty so that we can
;; create unique comdat names.
define void @main() {
  ret void
}
