; RUN: opt < %s -passes=adce -S | FileCheck %s

define void @test() {
entry:
; CHECK: tail call i32 asm sideeffect "add.u32 $0, $0, $0;", "=r"()
; CHECK: tail call void asm "st.u32 [$0], $1;", "l,r,~{memory}"

  %0 = tail call i16 asm "{  cvt.rn.f16.f64 $0, $1;}\0A", "=h,d"(double 0.000000e+00)
  %1 = tail call float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %0)
  %2 = tail call float asm "{  cvt.f32.f16 $0, $1;}\0A", "=f,h"(i16 %0)
  %add.i.i = fadd float %1, %2
  %3 = tail call i16 asm "{  cvt.rn.f16.f32 $0, $1;}\0A", "=h,f"(float %add.i.i)
  %4 = tail call i32 asm "add.u32 $0, $0, $0;", "=r"()
  %5 = tail call i32 asm sideeffect "add.u32 $0, $0, $0;", "=r"()
  tail call void asm "st.u32 [$0], $1;", "l,r,~{memory}"(i32* undef, i32 undef)
  ret void
}


;; An inline asm with an indirect ("=*m") output writes through the pointer it
;; is handed, so ADCE must keep the call even though it is not volatile and has
;; no direct uses.

@gi = external global i32

define void @indirect_output(i32 %i) {
; CHECK-LABEL: define void @indirect_output(
; CHECK: call void asm "st $1,$0", "=*m,r"
entry:
  call void asm "st $1,$0", "=*m,r"(ptr elementtype(i32) @gi, i32 %i)
  ret void
}

;; A pure-register asm with no side effects and an unused result stays dead.
define void @unused_register_output(i32 %i) {
; CHECK-LABEL: define void @unused_register_output(
; CHECK-NOT: call i32 asm
entry:
  %0 = call i32 asm "lr $0,$1", "=r,r"(i32 %i)
  ret void
}
