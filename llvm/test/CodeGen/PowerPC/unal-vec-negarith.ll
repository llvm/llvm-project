; RUN: llc -verify-machineinstrs -debug-only=isel <%s >%t 2>&1 && FileCheck <%t %s
; REQUIRES: asserts

target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define <16 x i8> @test_l_v16i8(ptr %p) #0 {
entry:
  %r = load <16 x i8>, ptr %p, align 1
  ret <16 x i8> %r

; CHECK-NOT: v4i32,ch = llvm.ppc.altivec.lvx{{.*}}<(load (s248) from %ir.p + 4294967281, align 1)>
; CHECK:     v4i32,ch = llvm.ppc.altivec.lvx{{.*}}<(load (s248) from %ir.p - 15, align 1)>
}

attributes #0 = { nounwind "target-cpu"="pwr7" }

