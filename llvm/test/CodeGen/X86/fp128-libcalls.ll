; RUN: llc < %s -O2 -mtriple=x86_64-linux-android -mattr=+mmx \
; RUN:     -enable-legalize-types-checking | FileCheck %s
; RUN: llc < %s -O2 -mtriple=x86_64-linux-gnu -mattr=+mmx \
; RUN:     -enable-legalize-types-checking | FileCheck %s

; Check all soft floating point library function calls.

@vf64 = common global double 0.000000e+00, align 8
@vf128 = common global fp128 0xL00000000000000000000000000000000, align 16

define void @Test128Add(fp128 %d1, fp128 %d2) {
entry:
  %add = fadd fp128 %d1, %d2
  store fp128 %add, fp128* @vf128, align 16
  ret void
; CHECK-LABEL: Test128Add:
; CHECK:       callq __addtf3
; CHECK-NEXT:  movaps %xmm0, vf128(%rip)
; CHECK:       retq
}

define void @Test128_1Add(fp128 %d1){
entry:
  %0 = load fp128, fp128* @vf128, align 16
  %add = fadd fp128 %0, %d1
  store fp128 %add, fp128* @vf128, align 16
  ret void
; CHECK-LABEL: Test128_1Add:
; CHECK:       movaps  %xmm0, %xmm1
; CHECK-NEXT:  movaps  vf128(%rip), %xmm0
; CHECK-NEXT:  callq   __addtf3
; CHECK-NEXT:  movaps  %xmm0, vf128(%rip)
; CHECK:       retq
}

define void @Test128Sub(fp128 %d1, fp128 %d2){
entry:
  %sub = fsub fp128 %d1, %d2
  store fp128 %sub, fp128* @vf128, align 16
  ret void
; CHECK-LABEL: Test128Sub:
; CHECK:       callq __subtf3
; CHECK-NEXT:  movaps %xmm0, vf128(%rip)
; CHECK:       retq
}

define void @Test128_1Sub(fp128 %d1){
entry:
  %0 = load fp128, fp128* @vf128, align 16
  %sub = fsub fp128 %0, %d1
  store fp128 %sub, fp128* @vf128, align 16
  ret void
; CHECK-LABEL: Test128_1Sub:
; CHECK:       movaps  %xmm0, %xmm1
; CHECK-NEXT:  movaps  vf128(%rip), %xmm0
; CHECK-NEXT:  callq   __subtf3
; CHECK-NEXT:  movaps  %xmm0, vf128(%rip)
; CHECK:       retq
}

define void @Test128Mul(fp128 %d1, fp128 %d2){
entry:
  %mul = fmul fp128 %d1, %d2
  store fp128 %mul, fp128* @vf128, align 16
  ret void
; CHECK-LABEL: Test128Mul:
; CHECK:       callq __multf3
; CHECK-NEXT:  movaps %xmm0, vf128(%rip)
; CHECK:       retq
}

define void @Test128_1Mul(fp128 %d1){
entry:
  %0 = load fp128, fp128* @vf128, align 16
  %mul = fmul fp128 %0, %d1
  store fp128 %mul, fp128* @vf128, align 16
  ret void
; CHECK-LABEL: Test128_1Mul:
; CHECK:       movaps  %xmm0, %xmm1
; CHECK-NEXT:  movaps  vf128(%rip), %xmm0
; CHECK-NEXT:  callq   __multf3
; CHECK-NEXT:  movaps  %xmm0, vf128(%rip)
; CHECK:       retq
}

define void @Test128Div(fp128 %d1, fp128 %d2){
entry:
  %div = fdiv fp128 %d1, %d2
  store fp128 %div, fp128* @vf128, align 16
  ret void
; CHECK-LABEL: Test128Div:
; CHECK:       callq __divtf3
; CHECK-NEXT:  movaps %xmm0, vf128(%rip)
; CHECK:       retq
}

define void @Test128_1Div(fp128 %d1){
entry:
  %0 = load fp128, fp128* @vf128, align 16
  %div = fdiv fp128 %0, %d1
  store fp128 %div, fp128* @vf128, align 16
  ret void
; CHECK-LABEL: Test128_1Div:
; CHECK:       movaps  %xmm0, %xmm1
; CHECK-NEXT:  movaps  vf128(%rip), %xmm0
; CHECK-NEXT:  callq   __divtf3
; CHECK-NEXT:  movaps  %xmm0, vf128(%rip)
; CHECK:       retq
}
