; RUN: llc < %s -march=x86-64 -mtriple=x86_64-apple-darwin -mcpu=skx | FileCheck %s

define i32 @mask32(i32 %x) {
  %m0 = bitcast i32 %x to <32 x i1>
  %m1 = xor <32 x i1> %m0, <i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1,
                            i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1,
                            i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1,
                            i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1>
  %ret = bitcast <32 x i1> %m1 to i32
  ret i32 %ret
; CHECK-LABEL: mask32
; CHECK: kmovd
; CHECK-NEXT: knotd
; CHECK-NEXT: kmovd
; CHECK-NEXT: ret
}

define i64 @mask64(i64 %x) {
  %m0 = bitcast i64 %x to <64 x i1>
  %m1 = xor <64 x i1> %m0, <i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1,
                            i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1,
                            i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1,
                            i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1,
                            i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1,
                            i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1,
                            i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1,
                            i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1>
  %ret = bitcast <64 x i1> %m1 to i64
  ret i64 %ret
; CHECK-LABEL: mask64
; CHECK: kmovq
; CHECK-NEXT: knotq
; CHECK-NEXT: kmovq
; CHECK-NEXT: ret
}

define void @mask32_mem(i32* %ptr) {
  %x = load i32, i32* %ptr, align 4
  %m0 = bitcast i32 %x to <32 x i1>
  %m1 = xor <32 x i1> %m0, <i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1,
                            i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1,
                            i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1,
                            i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1>
  %ret = bitcast <32 x i1> %m1 to i32
  store i32 %ret, i32* %ptr, align 4
  ret void
; CHECK-LABEL: mask32_mem
; CHECK: kmovd ([[ARG1:%rdi|%rcx]]), %k{{[0-7]}}
; CHECK-NEXT: knotd
; CHECK-NEXT: kmovd %k{{[0-7]}}, ([[ARG1]])
; CHECK-NEXT: ret
}

define void @mask64_mem(i64* %ptr) {
  %x = load i64, i64* %ptr, align 4
  %m0 = bitcast i64 %x to <64 x i1>
  %m1 = xor <64 x i1> %m0, <i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1,
                            i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1,
                            i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1,
                            i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1,
                            i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1,
                            i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1,
                            i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1,
                            i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1>
  %ret = bitcast <64 x i1> %m1 to i64
  store i64 %ret, i64* %ptr, align 4
  ret void
; CHECK-LABEL: mask64_mem
; CHECK: kmovq ([[ARG1]]), %k{{[0-7]}}
; CHECK-NEXT: knotq
; CHECK-NEXT: kmovq %k{{[0-7]}}, ([[ARG1]])
; CHECK-NEXT: ret
}

define i32 @mand32(i32 %x, i32 %y) {
  %ma = bitcast i32 %x to <32 x i1>
  %mb = bitcast i32 %y to <32 x i1>
  %mc = and <32 x i1> %ma, %mb
  %md = xor <32 x i1> %ma, %mb
  %me = or <32 x i1> %mc, %md
  %ret = bitcast <32 x i1> %me to i32
; CHECK: kandd
; CHECK: kxord
; CHECK: kord
  ret i32 %ret
}

define i64 @mand64(i64 %x, i64 %y) {
  %ma = bitcast i64 %x to <64 x i1>
  %mb = bitcast i64 %y to <64 x i1>
  %mc = and <64 x i1> %ma, %mb
  %md = xor <64 x i1> %ma, %mb
  %me = or <64 x i1> %mc, %md
  %ret = bitcast <64 x i1> %me to i64
; CHECK: kandq
; CHECK: kxorq
; CHECK: korq
  ret i64 %ret
}
