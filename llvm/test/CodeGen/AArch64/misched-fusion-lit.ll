; RUN: llc %s -o - -mtriple=aarch64-unknown -mattr=-fuse-adrp-add,-fuse-literals | FileCheck %s --check-prefix=CHECK --check-prefix=CHECKDONT
; RUN: llc %s -o - -mtriple=aarch64-unknown -mattr=+fuse-adrp-add,+fuse-literals | FileCheck %s --check-prefix=CHECK --check-prefix=CHECKFUSE
; RUN: llc %s -o - -mtriple=aarch64-unknown -mcpu=cortex-a57      | FileCheck %s --check-prefix=CHECK --check-prefix=CHECKFUSE
; RUN: llc %s -o - -mtriple=aarch64-unknown -mcpu=cortex-a65      | FileCheck %s --check-prefix=CHECK --check-prefix=CHECKFUSE
; RUN: llc %s -o - -mtriple=aarch64-unknown -mcpu=cortex-a72      | FileCheck %s --check-prefix=CHECK --check-prefix=CHECKFUSE
; RUN: llc %s -o - -mtriple=aarch64-unknown -mcpu=exynos-m3       | FileCheck %s --check-prefix=CHECK --check-prefix=CHECKFUSE
; RUN: llc %s -o - -mtriple=aarch64-unknown -mcpu=exynos-m4       | FileCheck %s --check-prefix=CHECK --check-prefix=CHECKFUSE
; RUN: llc %s -o - -mtriple=aarch64-unknown -mcpu=exynos-m5       | FileCheck %s --check-prefix=CHECK --check-prefix=CHECKFUSE

@g = common local_unnamed_addr global i8* null, align 8

define dso_local i8* @litp(i32 %a, i32 %b) {
entry:
  %add = add nsw i32 %b, %a
  %idx.ext = sext i32 %add to i64
  %add.ptr = getelementptr i8, i8* bitcast (i8* (i32, i32)* @litp to i8*), i64 %idx.ext
  store i8* %add.ptr, i8** @g, align 8
  ret i8* %add.ptr

; CHECK-LABEL: litp:
; CHECK: adrp [[R:x[0-9]+]], litp
; CHECKFUSE-NEXT: add {{x[0-9]+}}, [[R]], :lo12:litp
}

define dso_local i8* @litp_tune_generic(i32 %a, i32 %b) "tune-cpu"="generic" {
entry:
  %add = add nsw i32 %b, %a
  %idx.ext = sext i32 %add to i64
  %add.ptr = getelementptr i8, i8* bitcast (i8* (i32, i32)* @litp_tune_generic to i8*), i64 %idx.ext
  store i8* %add.ptr, i8** @g, align 8
  ret i8* %add.ptr

; CHECK-LABEL: litp_tune_generic:
; CHECK:         adrp [[R:x[0-9]+]], litp_tune_generic
; CHECK-NEXT:    add {{x[0-9]+}}, [[R]], :lo12:litp_tune_generic
}

define dso_local i32 @liti(i32 %a, i32 %b) {
entry:
  %add = add i32 %a, -262095121
  %add1 = add i32 %add, %b
  ret i32 %add1

; CHECK-LABEL: liti:
; CHECK: mov [[R:w[0-9]+]], {{#[0-9]+}}
; CHECKDONT-NEXT: add {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECKFUSE-NEXT: movk [[R]], {{#[0-9]+}}, lsl #16
}

; Function Attrs: norecurse nounwind readnone
define dso_local i64 @litl(i64 %a, i64 %b) {
entry:
  %add = add i64 %a, 2208998440489107183
  %add1 = add i64 %add, %b
  ret i64 %add1

; CHECK-LABEL: litl:
; CHECK: mov [[R:x[0-9]+]], {{#[0-9]+}}
; CHECKDONT-NEXT: add {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}
; CHECK-NEXT: movk [[R]], {{#[0-9]+}}, lsl #16
; CHECK: movk [[R]], {{#[0-9]+}}, lsl #32
; CHECK-NEXT: movk [[R]], {{#[0-9]+}}, lsl #48
}

; Function Attrs: norecurse nounwind readnone
define dso_local double @litf() {
entry:
  ret double 0x400921FB54442D18

; CHECK-LABEL: litf:
; CHECK-DONT:      adrp [[ADDR:x[0-9]+]], [[CSTLABEL:.LCP.*]]
; CHECK-DONT-NEXT: ldr  {{d[0-9]+}}, {{[[]}}[[ADDR]], :lo12:[[CSTLABEL]]{{[]]}}
; CHECK-FUSE:      mov  [[R:x[0-9]+]], #11544
; CHECK-FUSE:      movk [[R]], #21572, lsl #16
; CHECK-FUSE:      movk [[R]], #8699, lsl #32
; CHECK-FUSE:      movk [[R]], #16393, lsl #48
; CHECK-FUSE:      fmov {{d[0-9]+}}, [[R]]
}
