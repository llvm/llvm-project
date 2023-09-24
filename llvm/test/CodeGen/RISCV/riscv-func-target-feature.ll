; RUN: llc -mtriple=riscv64 -mcpu=sifive-u74 -verify-machineinstrs < %s | FileCheck %s

; CHECK: .option push
; CHECK-NEXT: .option arch, +v, +zve32f, +zve32x, +zve64d, +zve64f, +zve64x, +zvl128b, +zvl32b, +zvl64b
define void @test1() "target-features"="+a,+d,+f,+m,+c,+v,+zifencei,+zve32f,+zve32x,+zve64d,+zve64f,+zve64x,+zvl128b,+zvl32b,+zvl64b" {
; CHECK-LABEL: test1
; CHECK: .option pop
entry:
  ret void
}

; CHECK: .option push
; CHECK-NEXT: .option arch, +zihintntl
define void @test2() "target-features"="+a,+d,+f,+m,+zihintntl,+zifencei" {
; CHECK-LABEL: test2
; CHECK: .option pop
entry:
  ret void
}

; CHECK: .option push
; CHECK-NEXT: .option arch, -a, -d, -f, -m
define void @test3() "target-features"="-a,-d,-f,-m" {
; CHECK-LABEL: test3
; CHECK: .option pop
entry:
  ret void
}

; CHECK-NOT: .option push
define void @test4() {
; CHECK-LABEL: test4
; CHECK-NOT: .option pop
entry:
  ret void
}

; CHECK-NOT: .option push
define void @test5() "target-features"="+unaligned-scalar-mem" {
; CHECK-LABEL: test5
; CHECK-NOT: .option pop
entry:
  ret void
}
