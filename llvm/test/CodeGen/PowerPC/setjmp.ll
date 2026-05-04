; RUN: llc < %s -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr8 \
; RUN:   -verify-machineinstrs | FileCheck --check-prefix=CHECK64 %s

; RUN: llc < %s -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr8 \
; RUN:   -ppc-always-use-base-pointer -verify-machineinstrs | \
; RUN: FileCheck --check-prefix=BP64 %s

; RUN: llc < %s -mtriple=powerpc-unknown-linux-gnu -mcpu=pwr7 \
; RUN:   -verify-machineinstrs | FileCheck --check-prefix=CHECK32 %s

; RUN: llc < %s -mtriple=powerpc-unknown-linux-gnu -mcpu=pwr7 \
; RUN:   -ppc-always-use-base-pointer -verify-machineinstrs | \
; RUN: FileCheck --check-prefix=BP32 %s

; RUN: llc < %s -mtriple=powerpc-unknown-linux-gnu -mcpu=pwr7 \
; RUN:   -ppc-always-use-base-pointer -verify-machineinstrs \
; RUN:   -relocation-model=pic | FileCheck --check-prefix=PIC32 %s

@buf = internal global [5 x ptr] zeroinitializer, align 8

declare i32 @llvm.eh.sjlj.setjmp(ptr) nounwind

define i32 @setjmp_with_fp() nounwind #0 {
  %r = call i32 @llvm.eh.sjlj.setjmp(ptr @buf)
  ret i32 %r
}

; CHECK64-LABEL: setjmp_with_fp:
; CHECK64:         addis [[SCRATCH:[0-9]+]], 2, buf@toc@ha
; CHECK64:         addi [[BUF:[0-9]+]], [[SCRATCH]], buf@toc@l
; CHECK64:         std 2, 24([[BUF]])
; CHECK64:         std 31, 32([[BUF]])

; BP64-LABEL: setjmp_with_fp:
; BP64:         addis [[SCRATCH:[0-9]+]], 2, buf@toc@ha
; BP64:         addi [[BUF:[0-9]+]], [[SCRATCH]], buf@toc@l
; BP64:         std 2, 24([[BUF]])
; BP64:         std 30, 32([[BUF]])

; CHECK32-LABEL: setjmp_with_fp:
; CHECK32:         lis [[SCRATCH:[0-9]+]], buf@ha
; CHECK32:         la [[BUF:[0-9]+]], buf@l([[SCRATCH]])
; CHECK32:         stw 31, 16([[BUF]])

; BP32-LABEL: setjmp_with_fp:
; BP32:         lis [[SCRATCH:[0-9]+]], buf@ha
; BP32:         la [[BUF:[0-9]+]], buf@l([[SCRATCH]])
; BP32:         stw 30, 16([[BUF]])

; PIC32-LABEL: setjmp_with_fp:
; PIC32:         lwz [[SCRATCH:[0-9]+]], .L0$poff-.L0$pb(30)
; PIC32:         add 30, [[SCRATCH]], 30
; PIC32:         lwz [[BUF:[0-9]+]], .LC0-.LTOC(30)
; PIC32:         stw 29, 16([[BUF]])

define i32 @setjmp_without_fp() nounwind #1 {
  %r = call i32 @llvm.eh.sjlj.setjmp(ptr @buf)
  ret i32 %r
}

; CHECK64-LABEL: setjmp_without_fp:
; CHECK64:         addis [[SCRATCH:[0-9]+]], 2, buf@toc@ha
; CHECK64:         addi [[BUF:[0-9]+]], [[SCRATCH]], buf@toc@l
; CHECK64:         std 2, 24([[BUF]])
; CHECK64:         std 1, 32([[BUF]])

; BP64-LABEL: setjmp_without_fp:
; BP64:         addis [[SCRATCH:[0-9]+]], 2, buf@toc@ha
; BP64:         addi [[BUF:[0-9]+]], [[SCRATCH]], buf@toc@l
; BP64:         std 2, 24([[BUF]])
; BP64:         std 30, 32([[BUF]])

; CHECK32-LABEL: setjmp_without_fp:
; CHECK32:         lis [[SCRATCH:[0-9]+]], buf@ha
; CHECK32:         la [[BUF:[0-9]+]], buf@l([[SCRATCH]])
; CHECK32:         stw 1, 16(3)

; BP32-LABEL: setjmp_without_fp:
; BP32:         lis [[SCRATCH:[0-9]+]], buf@ha
; BP32:         la [[BUF:[0-9]+]], buf@l([[SCRATCH]])
; BP32:         stw 30, 16([[BUF]])

; PIC32-LABEL: setjmp_without_fp:
; PIC32:         lwz [[SCRATCH:[0-9]+]], .L1$poff-.L1$pb(30)
; PIC32:         add 30, [[SCRATCH]], 30
; PIC32:         lwz [[BUF:[0-9]+]], .LC0-.LTOC(30)
; PIC32:         stw 29, 16([[BUF]])

attributes #0 = { "frame-pointer"="all" }
attributes #1 = { "frame-pointer"="none" }
