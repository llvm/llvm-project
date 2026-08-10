; RUN: llc < %s -mtriple=thumbv7-apple-ios -O0 -verify-machineinstrs -fast-isel-abort=1 -relocation-model=static -mattr=+long-calls | FileCheck -check-prefix=CHECK-LONG %s
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -O0 -verify-machineinstrs -fast-isel-abort=1 -relocation-model=static -mattr=+long-calls | FileCheck -check-prefix=CHECK-LONG %s
; RUN: llc < %s -mtriple=thumbv7-apple-ios -O0 -verify-machineinstrs -fast-isel-abort=1 -relocation-model=static | FileCheck -check-prefix=CHECK-NORM %s
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -O0 -verify-machineinstrs -fast-isel-abort=1 -relocation-model=static | FileCheck -check-prefix=CHECK-NORM %s

define void @myadd(ptr %sum, ptr %addend) nounwind {
entry:
  %sum.addr = alloca ptr, align 4
  %addend.addr = alloca ptr, align 4
  store ptr %sum, ptr %sum.addr, align 4
  store ptr %addend, ptr %addend.addr, align 4
  %tmp = load ptr, ptr %sum.addr, align 4
  %tmp1 = load float, ptr %tmp
  %tmp2 = load ptr, ptr %addend.addr, align 4
  %tmp3 = load float, ptr %tmp2
  %add = fadd float %tmp1, %tmp3
  %tmp4 = load ptr, ptr %sum.addr, align 4
  store float %add, ptr %tmp4
  ret void
}

define i32 @main(i32 %argc, ptr %argv) nounwind {
entry:
  %ztot = alloca float, align 4
  %z = alloca float, align 4
  store float 0.000000e+00, ptr %ztot, align 4
  store float 1.000000e+00, ptr %z, align 4
; CHECK-LONG: blx     r
; CHECK-NORM: bl      {{_?}}myadd
  call void @myadd(ptr %ztot, ptr %z)
  ret i32 0
}
