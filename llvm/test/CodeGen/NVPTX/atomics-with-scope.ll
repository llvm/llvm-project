; RUN: llc < %s -mtriple=nvptx -mcpu=sm_60 | FileCheck %s -check-prefixes=CHECK,CHECK32
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_60 | FileCheck %s
; RUN: %if ptxas-sm_60 && ptxas-ptr32 %{ llc < %s -mtriple=nvptx -mcpu=sm_60 | %ptxas-verify -arch=sm_60 %}
; RUN: %if ptxas-sm_60 %{ llc < %s -mtriple=nvptx64 -mcpu=sm_60 | %ptxas-verify -arch=sm_60 %}

; CHECK-LABEL: .func test_atomics_scope(
define void @test_atomics_scope(ptr %fp, float %f,
                                ptr %dfp, double %df,
                                ptr %ip, i32 %i,
                                ptr %uip, i32 %ui,
                                ptr %llp, i64 %ll) #0 {
entry:
; CHECK: atom.cta.add.s32
  %tmp36 = tail call i32 @llvm.nvvm.atomic.add.gen.i.cta.i32.p0(ptr %ip, i32 %i)
; CHECK: atom.cta.add.u64
  %tmp38 = tail call i64 @llvm.nvvm.atomic.add.gen.i.cta.i64.p0(ptr %llp, i64 %ll)
; CHECK: atom.sys.add.s32
  %tmp39 = tail call i32 @llvm.nvvm.atomic.add.gen.i.sys.i32.p0(ptr %ip, i32 %i)
; CHECK: atom.sys.add.u64
  %tmp41 = tail call i64 @llvm.nvvm.atomic.add.gen.i.sys.i64.p0(ptr %llp, i64 %ll)
; CHECK: atom.cta.add.f32
  %tmp42 = tail call float @llvm.nvvm.atomic.add.gen.f.cta.f32.p0(ptr %fp, float %f)
; CHECK: atom.cta.add.f64
  %tmp43 = tail call double @llvm.nvvm.atomic.add.gen.f.cta.f64.p0(ptr %dfp, double %df)
; CHECK: atom.sys.add.f32
  %tmp44 = tail call float @llvm.nvvm.atomic.add.gen.f.sys.f32.p0(ptr %fp, float %f)
; CHECK: atom.sys.add.f64
  %tmp45 = tail call double @llvm.nvvm.atomic.add.gen.f.sys.f64.p0(ptr %dfp, double %df)

; CHECK: atom.cta.exch.b32
  %tmp46 = tail call i32 @llvm.nvvm.atomic.exch.gen.i.cta.i32.p0(ptr %ip, i32 %i)
; CHECK: atom.cta.exch.b64
  %tmp48 = tail call i64 @llvm.nvvm.atomic.exch.gen.i.cta.i64.p0(ptr %llp, i64 %ll)
; CHECK: atom.sys.exch.b32
  %tmp49 = tail call i32 @llvm.nvvm.atomic.exch.gen.i.sys.i32.p0(ptr %ip, i32 %i)
; CHECK: atom.sys.exch.b64
  %tmp51 = tail call i64 @llvm.nvvm.atomic.exch.gen.i.sys.i64.p0(ptr %llp, i64 %ll)

; CHECK: atom.cta.max.s32
  %tmp52 = tail call i32 @llvm.nvvm.atomic.max.gen.i.cta.i32.p0(ptr %ip, i32 %i)
; CHECK: atom.cta.max.s64
  %tmp56 = tail call i64 @llvm.nvvm.atomic.max.gen.i.cta.i64.p0(ptr %llp, i64 %ll)
; CHECK: atom.sys.max.s32
  %tmp58 = tail call i32 @llvm.nvvm.atomic.max.gen.i.sys.i32.p0(ptr %ip, i32 %i)
; CHECK: atom.sys.max.s64
  %tmp62 = tail call i64 @llvm.nvvm.atomic.max.gen.i.sys.i64.p0(ptr %llp, i64 %ll)

; CHECK: atom.cta.min.s32
  %tmp64 = tail call i32 @llvm.nvvm.atomic.min.gen.i.cta.i32.p0(ptr %ip, i32 %i)
; CHECK: atom.cta.min.s64
  %tmp68 = tail call i64 @llvm.nvvm.atomic.min.gen.i.cta.i64.p0(ptr %llp, i64 %ll)
; CHECK: atom.sys.min.s32
  %tmp70 = tail call i32 @llvm.nvvm.atomic.min.gen.i.sys.i32.p0(ptr %ip, i32 %i)
; CHECK: atom.sys.min.s64
  %tmp74 = tail call i64 @llvm.nvvm.atomic.min.gen.i.sys.i64.p0(ptr %llp, i64 %ll)

; CHECK: atom.cta.inc.u32
  %tmp76 = tail call i32 @llvm.nvvm.atomic.inc.gen.i.cta.i32.p0(ptr %ip, i32 %i)
; CHECK: atom.sys.inc.u32
  %tmp77 = tail call i32 @llvm.nvvm.atomic.inc.gen.i.sys.i32.p0(ptr %ip, i32 %i)

; CHECK: atom.cta.dec.u32
  %tmp78 = tail call i32 @llvm.nvvm.atomic.dec.gen.i.cta.i32.p0(ptr %ip, i32 %i)
; CHECK: atom.sys.dec.u32
  %tmp79 = tail call i32 @llvm.nvvm.atomic.dec.gen.i.sys.i32.p0(ptr %ip, i32 %i)

; CHECK: atom.cta.and.b32
  %tmp80 = tail call i32 @llvm.nvvm.atomic.and.gen.i.cta.i32.p0(ptr %ip, i32 %i)
; CHECK: atom.cta.and.b64
  %tmp82 = tail call i64 @llvm.nvvm.atomic.and.gen.i.cta.i64.p0(ptr %llp, i64 %ll)
; CHECK: atom.sys.and.b32
  %tmp83 = tail call i32 @llvm.nvvm.atomic.and.gen.i.sys.i32.p0(ptr %ip, i32 %i)
; CHECK: atom.sys.and.b64
  %tmp85 = tail call i64 @llvm.nvvm.atomic.and.gen.i.sys.i64.p0(ptr %llp, i64 %ll)

; CHECK: atom.cta.or.b32
  %tmp86 = tail call i32 @llvm.nvvm.atomic.or.gen.i.cta.i32.p0(ptr %ip, i32 %i)
; CHECK: atom.cta.or.b64
  %tmp88 = tail call i64 @llvm.nvvm.atomic.or.gen.i.cta.i64.p0(ptr %llp, i64 %ll)
; CHECK: atom.sys.or.b32
  %tmp89 = tail call i32 @llvm.nvvm.atomic.or.gen.i.sys.i32.p0(ptr %ip, i32 %i)
; CHECK: atom.sys.or.b64
  %tmp91 = tail call i64 @llvm.nvvm.atomic.or.gen.i.sys.i64.p0(ptr %llp, i64 %ll)

; CHECK: atom.cta.xor.b32
  %tmp92 = tail call i32 @llvm.nvvm.atomic.xor.gen.i.cta.i32.p0(ptr %ip, i32 %i)
; CHECK: atom.cta.xor.b64
  %tmp94 = tail call i64 @llvm.nvvm.atomic.xor.gen.i.cta.i64.p0(ptr %llp, i64 %ll)
; CHECK: atom.sys.xor.b32
  %tmp95 = tail call i32 @llvm.nvvm.atomic.xor.gen.i.sys.i32.p0(ptr %ip, i32 %i)
; CHECK: atom.sys.xor.b64
  %tmp97 = tail call i64 @llvm.nvvm.atomic.xor.gen.i.sys.i64.p0(ptr %llp, i64 %ll)

; CHECK: atom.cta.cas.b32
  %tmp98 = tail call i32 @llvm.nvvm.atomic.cas.gen.i.cta.i32.p0(ptr %ip, i32 %i, i32 %i)
; CHECK: atom.cta.cas.b64
  %tmp100 = tail call i64 @llvm.nvvm.atomic.cas.gen.i.cta.i64.p0(ptr %llp, i64 %ll, i64 %ll)
; CHECK: atom.sys.cas.b32
  %tmp101 = tail call i32 @llvm.nvvm.atomic.cas.gen.i.sys.i32.p0(ptr %ip, i32 %i, i32 %i)
; CHECK: atom.sys.cas.b64
  %tmp103 = tail call i64 @llvm.nvvm.atomic.cas.gen.i.sys.i64.p0(ptr %llp, i64 %ll, i64 %ll)

  ; CHECK: ret
  ret void
}

; Make sure we use constants as operands to our scoped atomic calls, where appropriate.
; CHECK-LABEL: .func test_atomics_scope_imm(
define void @test_atomics_scope_imm(ptr %fp, float %f,
                                    ptr %dfp, double %df,
                                    ptr %ip, i32 %i,
                                    ptr %uip, i32 %ui,
                                    ptr %llp, i64 %ll) #0 {

; CHECK: atom.cta.add.s32{{.*}} %r{{[0-9]+}};
  %tmp1r = tail call i32 @llvm.nvvm.atomic.add.gen.i.cta.i32.p0(ptr %ip, i32 %i)
; CHECK: atom.cta.add.s32{{.*}}, 1;
  %tmp1i = tail call i32 @llvm.nvvm.atomic.add.gen.i.cta.i32.p0(ptr %ip, i32 1)
; CHECK: atom.cta.add.u64{{.*}}, %rd{{[0-9]+}};
  %tmp2r = tail call i64 @llvm.nvvm.atomic.add.gen.i.cta.i64.p0(ptr %llp, i64 %ll)
; CHECK: atom.cta.add.u64{{.*}}, 2;
  %tmp2i = tail call i64 @llvm.nvvm.atomic.add.gen.i.cta.i64.p0(ptr %llp, i64 2)

; CHECK: atom.cta.add.f32{{.*}}, %r{{[0-9]+}};
  %tmp3r = tail call float @llvm.nvvm.atomic.add.gen.f.cta.f32.p0(ptr %fp, float %f)
; CHECK: atom.cta.add.f32{{.*}}, 0f40400000;
  %tmp3i = tail call float @llvm.nvvm.atomic.add.gen.f.cta.f32.p0(ptr %fp, float 3.0)
; CHECK: atom.cta.add.f64{{.*}}, %rd{{[0-9]+}};
  %tmp4r = tail call double @llvm.nvvm.atomic.add.gen.f.cta.f64.p0(ptr %dfp, double %df)
; CHECK: atom.cta.add.f64{{.*}}, 0d4010000000000000;
  %tmp4i = tail call double @llvm.nvvm.atomic.add.gen.f.cta.f64.p0(ptr %dfp, double 4.0)

; CAS is implemented separately and has more arguments
; CHECK: atom.cta.cas.b32{{.*}}], %r{{[0-9+]}}, %r{{[0-9+]}};
  %tmp5rr = tail call i32 @llvm.nvvm.atomic.cas.gen.i.cta.i32.p0(ptr %ip, i32 %i, i32 %i)
; For some reason in 64-bit mode we end up passing 51 via a register.
; CHECK32: atom.cta.cas.b32{{.*}}], %r{{[0-9+]}}, 51;
  %tmp5ri = tail call i32 @llvm.nvvm.atomic.cas.gen.i.cta.i32.p0(ptr %ip, i32 %i, i32 51)
; CHECK: atom.cta.cas.b32{{.*}}], 52, %r{{[0-9+]}};
  %tmp5ir = tail call i32 @llvm.nvvm.atomic.cas.gen.i.cta.i32.p0(ptr %ip, i32 52, i32 %i)
; CHECK: atom.cta.cas.b32{{.*}}], 53, 54;
  %tmp5ii = tail call i32 @llvm.nvvm.atomic.cas.gen.i.cta.i32.p0(ptr %ip, i32 53, i32 54)

  ; CHECK: ret
  ret void
}

declare i32 @llvm.nvvm.atomic.add.gen.i.cta.i32.p0(ptr nocapture, i32) #1
declare i64 @llvm.nvvm.atomic.add.gen.i.cta.i64.p0(ptr nocapture, i64) #1
declare i32 @llvm.nvvm.atomic.add.gen.i.sys.i32.p0(ptr nocapture, i32) #1
declare i64 @llvm.nvvm.atomic.add.gen.i.sys.i64.p0(ptr nocapture, i64) #1
declare float @llvm.nvvm.atomic.add.gen.f.cta.f32.p0(ptr nocapture, float) #1
declare double @llvm.nvvm.atomic.add.gen.f.cta.f64.p0(ptr nocapture, double) #1
declare float @llvm.nvvm.atomic.add.gen.f.sys.f32.p0(ptr nocapture, float) #1
declare double @llvm.nvvm.atomic.add.gen.f.sys.f64.p0(ptr nocapture, double) #1
declare i32 @llvm.nvvm.atomic.exch.gen.i.cta.i32.p0(ptr nocapture, i32) #1
declare i64 @llvm.nvvm.atomic.exch.gen.i.cta.i64.p0(ptr nocapture, i64) #1
declare i32 @llvm.nvvm.atomic.exch.gen.i.sys.i32.p0(ptr nocapture, i32) #1
declare i64 @llvm.nvvm.atomic.exch.gen.i.sys.i64.p0(ptr nocapture, i64) #1
declare i32 @llvm.nvvm.atomic.max.gen.i.cta.i32.p0(ptr nocapture, i32) #1
declare i64 @llvm.nvvm.atomic.max.gen.i.cta.i64.p0(ptr nocapture, i64) #1
declare i32 @llvm.nvvm.atomic.max.gen.i.sys.i32.p0(ptr nocapture, i32) #1
declare i64 @llvm.nvvm.atomic.max.gen.i.sys.i64.p0(ptr nocapture, i64) #1
declare i32 @llvm.nvvm.atomic.min.gen.i.cta.i32.p0(ptr nocapture, i32) #1
declare i64 @llvm.nvvm.atomic.min.gen.i.cta.i64.p0(ptr nocapture, i64) #1
declare i32 @llvm.nvvm.atomic.min.gen.i.sys.i32.p0(ptr nocapture, i32) #1
declare i64 @llvm.nvvm.atomic.min.gen.i.sys.i64.p0(ptr nocapture, i64) #1
declare i32 @llvm.nvvm.atomic.inc.gen.i.cta.i32.p0(ptr nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.inc.gen.i.sys.i32.p0(ptr nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.dec.gen.i.cta.i32.p0(ptr nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.dec.gen.i.sys.i32.p0(ptr nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.and.gen.i.cta.i32.p0(ptr nocapture, i32) #1
declare i64 @llvm.nvvm.atomic.and.gen.i.cta.i64.p0(ptr nocapture, i64) #1
declare i32 @llvm.nvvm.atomic.and.gen.i.sys.i32.p0(ptr nocapture, i32) #1
declare i64 @llvm.nvvm.atomic.and.gen.i.sys.i64.p0(ptr nocapture, i64) #1
declare i32 @llvm.nvvm.atomic.or.gen.i.cta.i32.p0(ptr nocapture, i32) #1
declare i64 @llvm.nvvm.atomic.or.gen.i.cta.i64.p0(ptr nocapture, i64) #1
declare i32 @llvm.nvvm.atomic.or.gen.i.sys.i32.p0(ptr nocapture, i32) #1
declare i64 @llvm.nvvm.atomic.or.gen.i.sys.i64.p0(ptr nocapture, i64) #1
declare i32 @llvm.nvvm.atomic.xor.gen.i.cta.i32.p0(ptr nocapture, i32) #1
declare i64 @llvm.nvvm.atomic.xor.gen.i.cta.i64.p0(ptr nocapture, i64) #1
declare i32 @llvm.nvvm.atomic.xor.gen.i.sys.i32.p0(ptr nocapture, i32) #1
declare i64 @llvm.nvvm.atomic.xor.gen.i.sys.i64.p0(ptr nocapture, i64) #1
declare i32 @llvm.nvvm.atomic.cas.gen.i.cta.i32.p0(ptr nocapture, i32, i32) #1
declare i64 @llvm.nvvm.atomic.cas.gen.i.cta.i64.p0(ptr nocapture, i64, i64) #1
declare i32 @llvm.nvvm.atomic.cas.gen.i.sys.i32.p0(ptr nocapture, i32, i32) #1
declare i64 @llvm.nvvm.atomic.cas.gen.i.sys.i64.p0(ptr nocapture, i64, i64) #1

attributes #1 = { argmemonly nounwind }
