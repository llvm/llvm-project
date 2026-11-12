; Given a library call that is represented as an llvm intrinsic call, but
; later transformed to an actual call, if an overriding definition of that
; library routine is provided indirectly via an alias, verify that LTO
; does not eliminate the definition.  This is a test for PR38547.
;
; RUN: llvm-as -o %t1 %s
; RUN: llvm-lto -exported-symbol=main -save-merged-module -filetype=asm -o %t2 %t1
; RUN: llvm-dis -o - %t2.merged.bc | FileCheck --check-prefix=CHECK_IR %s
;
; Check that the call is represented as an llvm intrinsic in the IR after LTO:
; CHECK_IR-LABEL: main
; CHECK_IR: call float @llvm.log.f32
;
; Check that the IR contains the overriding definition of the library routine
; in the IR after LTO:
; CHECK_IR: define internal float @logf(float
; CHECK_IR-NEXT:   [[TMP:%.*]] = fadd float
; CHECK_IR-NEXT:   ret float [[TMP]]
;
; Check that the assembly code from LTO contains the call to the expected
; library routine, and that the overriding definition of the library routine
; is present:
; RUN: FileCheck --check-prefix=CHECK_ASM %s < %t2
; CHECK_ASM-LABEL: main:
; CHECK_ASM: callq logf
; CHECK_ASM-LABEL: logf:
; CHECK_ASM-NEXT: add
; CHECK_ASM-NEXT: ret

; Produced from the following source-code:
;
;extern float logf(float);
;// 'src' and 'dst' are 'volatile' to prohibit optimization.
;volatile float src = 3.14f;
;volatile float dst;
;
;int main() {
;  dst = logf(src);
;  return 0;
;}
;
;extern float fname(float x);
;float fname(float x) {
;  return x + x;
;}
;
;float logf(float x) __attribute__((alias("fname")));
;
target triple = "x86_64-unknown-linux-gnu"

@src = global float 0x40091EB860000000, align 4
@dst = common global float 0.000000e+00, align 4

@logf = alias float (float), ptr @fname

define i32 @main() local_unnamed_addr {
entry:
  %0 = load volatile float, ptr @src, align 4
  %1 = tail call float @llvm.log.f32(float %0)
  store volatile float %1, ptr @dst, align 4
  ret i32 0
}

declare float @llvm.log.f32(float)

define float @fname(float %x) {
  %add = fadd float %x, %x
  ret float %add
}
