; RUN: llc -mtriple=thumbv4t-eabi %s -o - | FileCheck --check-prefix=CHECK -check-prefix=CHECK-V4T %s
; RUN: llc -mtriple=thumbv5t-eabi %s -o - | FileCheck --check-prefix=CHECK -check-prefix=CHECK-V5T %s

@f = common global ptr null, align 4

; CHECK-LABEL: foo:
define void @foo(i32 %x) {
entry:
  %0 = load ptr, ptr @f, align 4
  tail call void %0(i32 %x)
  ret void

; CHECK: ldr [[TMP:r[0-3]]], [[F:\.[A-Z0-9_]+]]
; CHECK: ldr [[CALLEE:r[0-3]]], [[[TMP]]]

; CHECK-V4T-NOT: blx
; CHECK-V4T: bl [[INDIRECT_PAD:\.Ltmp[0-9]+]]
; CHECK-V4T: [[F]]:
; CHECK-V4T: [[INDIRECT_PAD]]:
; CHECK-V4T-NEXT: bx [[CALLEE]]
; CHECK-V5T: blx [[CALLEE]]
}

; CHECK-LABEL: bar:
define void @bar(ptr nocapture %g, i32 %x, ptr nocapture %h) {
entry:
  tail call void %g(i32 %x)
  tail call void %h(i32 %x)
  ret void

; CHECK-V4T: bl [[INDIRECT_PAD1:\.Ltmp[0-9]+]]
; CHECK-V4T: bl [[INDIRECT_PAD2:\.Ltmp[0-9]+]]
; CHECK-V4T: [[INDIRECT_PAD1]]:
; CHECK-V4T-NEXT: bx
; CHECK-V4T: [[INDIRECT_PAD2]]:
; CHECK-V4T-NEXT: bx
; CHECK-V5T: blx
; CHECK-V5T: blx
}
