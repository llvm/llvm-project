; RUN: llc -verify-machineinstrs -enable-machine-outliner -mtriple aarch64 %s -o - | \
; RUN:   FileCheck %s --check-prefixes CHECK,V8A
; RUN: llc -verify-machineinstrs -enable-machine-outliner -mtriple aarch64 -mattr=+v8.3a %s -o - | \
; RUN:   FileCheck %s --check-prefixes CHECK,V83A

; Leaf functions with a_key
define dso_local i32 @foo_leaf(i32 noundef %x) minsize "sign-return-address"="all" "sign-return-address-key"="a_key" "zero-call-used-regs"="all" {
; CHECK-LABEL: foo_leaf:
; V8A:         hint #25
; V83A:        paciasp
entry:
  %add = add nsw i32 %x, 1
; V8A:         hint #29
; V83A:        autiasp
; CHECK-NOT:   bl
; CHECK-NOT:   str {{.*}}x30
; CHECK:       mov x1, #0
; CHECK:       ret
  ret i32 %add
}

define dso_local i32 @bar_leaf(i32 noundef %x) minsize "sign-return-address"="all" "sign-return-address-key"="a_key" "zero-call-used-regs"="all" {
; CHECK-LABEL: bar_leaf:
; V8A:         hint #25
; V83A:        paciasp
entry:
  %add = add nsw i32 %x, 2
; V8A:         hint #29
; V83A:        autiasp
; CHECK-NOT:   bl
; CHECK-NOT:   str {{.*}}x30
; CHECK:       mov x1, #0
; CHECK:       ret
  ret i32 %add
}

; Leaf functions with b_key
define dso_local i32 @foo_b_leaf(i32 noundef %x) minsize "sign-return-address"="all" "sign-return-address-key"="b_key" "zero-call-used-regs"="all" {
; CHECK-LABEL: foo_b_leaf:
; V8A:         hint #27
; V83A:        pacibsp
entry:
  %add = add nsw i32 %x, 1
; V8A:         hint #31
; V83A:        autibsp
; CHECK-NOT:   bl
; CHECK-NOT:   str {{.*}}x30
; CHECK:       mov x1, #0
; CHECK:       ret
  ret i32 %add
}

define dso_local i32 @bar_b_leaf(i32 noundef %x) minsize "sign-return-address"="all" "sign-return-address-key"="b_key" "zero-call-used-regs"="all" {
; CHECK-LABEL: bar_b_leaf:
; V8A:         hint #27
; V83A:        pacibsp
entry:
  %add = add nsw i32 %x, 2
; V8A:         hint #31
; V83A:        autibsp
; CHECK-NOT:   bl
; CHECK-NOT:   str {{.*}}x30
; CHECK:       mov x1, #0
; CHECK:       ret
  ret i32 %add
}

; Non-leaf functions
declare void @ext()

define void @nonleaf1() minsize "sign-return-address"="non-leaf" "sign-return-address-key"="a_key" "zero-call-used-regs"="all" {
; CHECK-LABEL: nonleaf1:
; V8A:         hint #25
; V83A:        paciasp
entry:
  call void @ext()
; CHECK:       bl ext
; CHECK:       ldr x30, [sp], #16
; V8A:         hint #29
; V83A:        autiasp
; CHECK-NOT:   bl
; CHECK-NOT:   str {{.*}}x30
; CHECK:       mov x0, #0
; CHECK:       ret
  ret void
}

define void @nonleaf2() minsize "sign-return-address"="non-leaf" "sign-return-address-key"="a_key" "zero-call-used-regs"="all" {
; CHECK-LABEL: nonleaf2:
; V8A:         hint #25
; V83A:        paciasp
entry:
  call void @ext()
; CHECK:       bl ext
; CHECK:       ldr x30, [sp], #16
; V8A:         hint #29
; V83A:        autiasp
; CHECK-NOT:   bl
; CHECK-NOT:   str {{.*}}x30
; CHECK:       mov x0, #0
; CHECK:       ret
  ret void
}

; CHECK-NOT: OUTLINED_FUNCTION
