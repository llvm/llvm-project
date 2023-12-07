; RUN: llc < %s -mcpu=cortex-a8 | FileCheck %s
target triple = "thumbv7-apple-ios"

; CHECK: local_split
;
; The load must go into d0-15 which are all clobbered by the asm.
; RAGreedy should split the range and use d16-d31 to avoid a spill.
;
; CHECK: vldr s
; CHECK-NOT: vstr
; CHECK: vadd.f32
; CHECK-NOT: vstr
; CHECK: vorr
; CHECK: vstr s
define void @local_split(ptr nocapture %p) nounwind ssp {
entry:
  %x = load float, ptr %p, align 4
  %a = fadd float %x, 1.0
  tail call void asm sideeffect "", "~{d0},~{d1},~{d2},~{d3},~{d4},~{d5},~{d6},~{d7},~{d8},~{d9},~{d10},~{d11},~{d12},~{d13},~{d14},~{d15}"() nounwind
  store float %a, ptr %p, align 4
  ret void
}

; CHECK: global_split
;
; Same thing, but across basic blocks.
;
; CHECK: vldr s
; CHECK-NOT: vstr
; CHECK: vadd.f32
; CHECK-NOT: vstr
; CHECK: vorr
; CHECK: vstr s
define void @global_split(ptr nocapture %p1, ptr nocapture %p2) nounwind ssp {
entry:
  %0 = load float, ptr %p1, align 4
  %add = fadd float %0, 1.000000e+00
  tail call void asm sideeffect "", "~{d0},~{d1},~{d2},~{d3},~{d4},~{d5},~{d6},~{d7},~{d8},~{d9},~{d10},~{d11},~{d12},~{d13},~{d14},~{d15}"() nounwind
  %cmp = fcmp ogt float %add, 0.000000e+00
  br i1 %cmp, label %if.then, label %if.end

if.then:
  store float %add, ptr %p2, align 4
  br label %if.end

if.end:
  store float %add, ptr %p1, align 4
  ret void
}
