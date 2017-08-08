; RUN: llc %s -o - -mtriple=aarch64-unknown -mattr=-fuse-literals | FileCheck %s --check-prefix=CHECK --check-prefix=CHECKDONT
; RUN: llc %s -o - -mtriple=aarch64-unknown -mattr=+fuse-literals | FileCheck %s --check-prefix=CHECK --check-prefix=CHECKFUSE
; RUN: llc %s -o - -mtriple=aarch64-unknown -mcpu=cortex-a57      | FileCheck %s --check-prefix=CHECK --check-prefix=CHECKFUSE

@g = common local_unnamed_addr global i8* null, align 8

define i8* @litp(i32 %a, i32 %b) {
entry:
  %add = add nsw i32 %b, %a
  %idx.ext = sext i32 %add to i64
  %add.ptr = getelementptr i8, i8* bitcast (i8* (i32, i32)* @litp to i8*), i64 %idx.ext
  store i8* %add.ptr, i8** @g, align 8
  ret i8* %add.ptr

; CHECK-LABEL: litp:
; CHECK: adrp [[R:x[0-9]+]], litp
; CHECKDONT-NEXT: add {{w[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
; CHECKFUSE-NEXT: add {{x[0-9]+}}, [[R]], :lo12:litp
}

define i32 @liti(i32 %a, i32 %b) {
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
define i64 @litl(i64 %a, i64 %b) {
entry:
  %add = add i64 %a, 2208998440489107183
  %add1 = add i64 %add, %b
  ret i64 %add1

; CHECK-LABEL: litl:
; CHECK: mov [[R:x[0-9]+]], {{#[0-9]+}}
; CHECK-NEXT: movk [[R]], {{#[0-9]+}}, lsl #16
; CHECK: movk [[R]], {{#[0-9]+}}, lsl #32
; CHECKDONT-NEXT: add {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}
; CHECKFUSE-NEXT: movk [[R]], {{#[0-9]+}}, lsl #48
}
