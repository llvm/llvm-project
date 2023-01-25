; RUN: llc < %s -mtriple armv7a-none-linux-gnueabihf -mcpu=cortex-a5 | FileCheck %s -check-prefix=CHECK-LINUXA5
; RUN: llc < %s -mtriple armv7a-none-linux-gnueabihf -mcpu=cortex-a8 | FileCheck %s -check-prefix=CHECK-LINUXA8
; RUN: llc < %s -mtriple armv7a-none-linux-gnueabihf -mcpu=cortex-a9 | FileCheck %s -check-prefix=CHECK-LINUXA9
; RUN: llc < %s -mtriple armv7a-none-linux-gnueabihf -mcpu=cortex-a15 | FileCheck %s -check-prefix=CHECK-LINUXA15
; RUN: llc < %s -mtriple armv7a-none-linux-gnueabihf -mcpu=swift | FileCheck %s -check-prefix=CHECK-LINUXSWIFT

; RUN: llc < %s -mtriple armv7a-none-linux-gnueabihf -mcpu=cortex-a5 --enable-unsafe-fp-math | FileCheck %s -check-prefix=CHECK-UNSAFEA5
; RUN: llc < %s -mtriple armv7a-none-linux-gnueabihf -mcpu=cortex-a8 --enable-unsafe-fp-math | FileCheck %s -check-prefix=CHECK-UNSAFEA8
; RUN: llc < %s -mtriple armv7a-none-linux-gnueabihf -mcpu=cortex-a9 --enable-unsafe-fp-math | FileCheck %s -check-prefix=CHECK-UNSAFEA9
; RUN: llc < %s -mtriple armv7a-none-linux-gnueabihf -mcpu=cortex-a15 --enable-unsafe-fp-math | FileCheck %s -check-prefix=CHECK-UNSAFEA15
; RUN: llc < %s -mtriple armv7a-none-linux-gnueabihf -mcpu=swift --enable-unsafe-fp-math | FileCheck %s -check-prefix=CHECK-UNSAFESWIFT

; RUN: llc < %s -mtriple armv7a-none-darwin -mcpu=cortex-a5 | FileCheck %s -check-prefix=CHECK-DARWINA5
; RUN: llc < %s -mtriple armv7a-none-darwin -mcpu=cortex-a8 | FileCheck %s -check-prefix=CHECK-DARWINA8
; RUN: llc < %s -mtriple armv7a-none-darwin -mcpu=cortex-a9 | FileCheck %s -check-prefix=CHECK-DARWINA9
; RUN: llc < %s -mtriple armv7a-none-darwin -mcpu=cortex-a15 | FileCheck %s -check-prefix=CHECK-DARWINA15
; RUN: llc < %s -mtriple armv7a-none-darwin -mcpu=swift | FileCheck %s -check-prefix=CHECK-DARWINSWIFT

; This test makes sure we're not lowering VMUL.f32 D* (aka. NEON) for single-prec. FP ops, since
; NEON is not fully IEEE 754 compliant, unless unsafe-math is selected.

@.str = private unnamed_addr constant [12 x i8] c"S317\09%.5g \0A\00", align 1

; CHECK-LINUXA5-LABEL: main:
; CHECK-LINUXA8-LABEL: main:
; CHECK-LINUXA9-LABEL: main:
; CHECK-LINUXA15-LABEL: main:
; CHECK-LINUXSWIFT-LABEL: main:
; CHECK-UNSAFEA5-LABEL: main:
; CHECK-UNSAFEA8-LABEL: main:
; CHECK-UNSAFEA9-LABEL: main:
; CHECK-UNSAFEA15-LABEL: main:
; CHECK-UNSAFESWIFT-LABEL: main:
; CHECK-DARWINA5-LABEL: main:
; CHECK-DARWINA8-LABEL: main:
; CHECK-DARWINA9-LABEL: main:
; CHECK-DARWINA15-LABEL: main:
; CHECK-DARWINSWIFT-LABEL: main:
define i32 @main() {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.04 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %q.03 = phi float [ 1.000000e+00, %entry ], [ %mul, %for.body ]
  %mul = fmul float %q.03, 0x3FEFAE1480000000
; CHECK-LINUXA5: vmul.f32 s{{[0-9]*}}
; CHECK-LINUXA8: vmul.f32 s{{[0-9]*}}
; CHECK-LINUXA9: vmul.f32 s{{[0-9]*}}
; CHECK-LINUXA15: vmul.f32 s{{[0-9]*}}
; Swift is *always* unsafe
; CHECK-LINUXSWIFT: vmul.f32 d{{[0-9]*}}

; CHECK-UNSAFEA5: vmul.f32 d{{[0-9]*}}
; CHECK-UNSAFEA8: vmul.f32 d{{[0-9]*}}
; A9 and A15 don't need this
; CHECK-UNSAFEA9: vmul.f32 s{{[0-9]*}}
; CHECK-UNSAFEA15: vmul.f32 s{{[0-9]*}}
; CHECK-UNSAFESWIFT: vmul.f32 d{{[0-9]*}}

; CHECK-DARWINA5: vmul.f32 d{{[0-9]*}}
; CHECK-DARWINA8: vmul.f32 d{{[0-9]*}}
; CHECK-DARWINA9: vmul.f32 s{{[0-9]*}}
; CHECK-DARWINA15: vmul.f32 s{{[0-9]*}}
; CHECK-DARWINSWIFT: vmul.f32 d{{[0-9]*}}
  %conv = fpext float %mul to double
  %call = tail call i32 (ptr, ...) @printf(ptr @.str, double %conv) #1
  %inc = add nsw i32 %i.04, 1
  %exitcond = icmp eq i32 %inc, 16000
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret i32 0
}

declare i32 @printf(ptr nocapture, ...)
