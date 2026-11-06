; Check outlining before authentication independently of post-RA scheduling.
; RUN: llc -verify-machineinstrs -enable-post-misched=false -enable-machine-outliner -mtriple aarch64 %s -o - | \
; RUN:   FileCheck %s --check-prefixes CHECK,V8A
; RUN: llc -verify-machineinstrs -enable-post-misched=false -enable-machine-outliner -mtriple aarch64 -mattr=+v8.3a %s -o - | \
; RUN:   FileCheck %s --check-prefixes CHECK,V83A
; RUN: llc -verify-machineinstrs -enable-machine-outliner -mtriple aarch64 %s -o - | \
; RUN:   FileCheck %s --check-prefixes DEFAULT,DEFAULT-V8A
; RUN: llc -verify-machineinstrs -enable-machine-outliner -mtriple aarch64 -mattr=+v8.3a %s -o - | \
; RUN:   FileCheck %s --check-prefixes DEFAULT,DEFAULT-V83A

define i32 @foo(i32 %x) minsize "sign-return-address"="all" "sign-return-address-key"="a_key" "zero-call-used-regs"="all" nounwind {
; CHECK-LABEL: foo:
; V8A:         hint #25
; V83A:        paciasp
  %add = add i32 %x, 1
; CHECK:       str x30, [sp, #-16]!
; CHECK-NEXT:  bl OUTLINED_FUNCTION_0
; CHECK-NEXT:  ldr x30, [sp], #16
; V8A-NEXT:    hint #29
; V8A-NEXT:    ret
; V83A-NEXT:   retaa
;
; DEFAULT-LABEL: foo:
; DEFAULT-V8A:   hint #29
; DEFAULT-V83A:  autiasp
; DEFAULT-NEXT:  b OUTLINED_FUNCTION_0
  ret i32 %add
}

define i32 @bar(i32 %x) minsize "sign-return-address"="all" "sign-return-address-key"="a_key" "zero-call-used-regs"="all" nounwind {
; CHECK-LABEL: bar:
; V8A:         hint #25
; V83A:        paciasp
  %add = add i32 %x, 2
; CHECK:       str x30, [sp, #-16]!
; CHECK-NEXT:  bl OUTLINED_FUNCTION_0
; CHECK-NEXT:  ldr x30, [sp], #16
; V8A-NEXT:    hint #29
; V8A-NEXT:    ret
; V83A-NEXT:   retaa
;
; DEFAULT-LABEL: bar:
; DEFAULT-V8A:   hint #29
; DEFAULT-V83A:  autiasp
; DEFAULT-NEXT:  b OUTLINED_FUNCTION_0
  ret i32 %add
}

; CHECK-LABEL: OUTLINED_FUNCTION_0:
; V8A:         hint #25
; V83A:        paciasp
; CHECK:       mov x1, #0
; CHECK:       movi v31.2d, #0000000000000000
; V8A-NEXT:    hint #29
; V8A-NEXT:    ret
; V83A-NEXT:   retaa
;
; DEFAULT-LABEL: OUTLINED_FUNCTION_0:
; DEFAULT-V8A:   hint #25
; DEFAULT-V83A:  paciasp
; DEFAULT:       mov x1, #0
; DEFAULT:       movi v31.2d, #0000000000000000
; DEFAULT-V8A-NEXT: hint #29
; DEFAULT-V8A-NEXT: ret
; DEFAULT-V83A-NEXT: retaa
