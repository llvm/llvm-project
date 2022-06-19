; RUN: not llc < %s -mtriple=x86_64-unknown-unknown -mattr=avx512f -stop-after=finalize-isel > %t 2> %t.err
; RUN: FileCheck < %t %s
; RUN: FileCheck --check-prefix=CHECK-STDERR < %t.err %s
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=avx512fp16 -stop-after=finalize-isel | FileCheck --check-prefixes=CHECK,FP16 %s

; CHECK-LABEL: name: mask_Yk_i8
; CHECK: %[[REG1:.*]]:vr512_0_15 = COPY %1
; CHECK: %[[REG2:.*]]:vr512_0_15 = COPY %2
; CHECK: INLINEASM &"vpaddq\09$3, $2, $0 {$1}", 0 /* attdialect */, {{.*}}, def %{{.*}}, {{.*}}, %{{.*}}, {{.*}}, %[[REG1]], {{.*}}, %[[REG2]], 12 /* clobber */, implicit-def early-clobber $df, 12 /* clobber */, implicit-def early-clobber $fpsw, 12 /* clobber */, implicit-def early-clobber $eflags

define <8 x i64> @mask_Yk_i8(i8 signext %msk, <8 x i64> %x, <8 x i64> %y) {
entry:
  %0 = tail call <8 x i64> asm "vpaddq\09$3, $2, $0 {$1}", "=x,^Yk,x,x,~{dirflag},~{fpsr},~{flags}"(i8 %msk, <8 x i64> %x, <8 x i64> %y)
  ret <8 x i64> %0
}

; FP16-LABEL: name: mask_Yk_f16
; FP16: %[[REG1:.*]]:vr512_0_15 = COPY %1
; FP16: %[[REG2:.*]]:vr512_0_15 = COPY %2
; FP16: INLINEASM &"vaddph\09$3, $2, $0 {$1}", 0 /* attdialect */, {{.*}}, def %{{.*}}, {{.*}}, %{{.*}}, {{.*}}, %[[REG1]], {{.*}}, %[[REG2]], 12 /* clobber */, implicit-def early-clobber $df, 12 /* clobber */, implicit-def early-clobber $fpsw, 12 /* clobber */, implicit-def early-clobber $eflags
; CHECK-STDERR: couldn't allocate output register for constraint 'x'
define <32 x half> @mask_Yk_f16(i8 signext %msk, <32 x half> %x, <32 x half> %y) {
entry:
  %0 = tail call <32 x half> asm "vaddph\09$3, $2, $0 {$1}", "=x,^Yk,x,x,~{dirflag},~{fpsr},~{flags}"(i8 %msk, <32 x half> %x, <32 x half> %y)
  ret <32 x half> %0
}
