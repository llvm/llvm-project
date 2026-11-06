; RUN: llc -verify-machineinstrs -enable-machine-outliner=never -mtriple aarch64 %s -o - | \
; RUN:   FileCheck %s --check-prefixes CHECK,V8A
; RUN: llc -verify-machineinstrs -enable-machine-outliner=never -mtriple aarch64 -mattr=+v8.3a %s -o - | \
; RUN:   FileCheck %s --check-prefixes CHECK,V83A

define i32 @leaf_a(i32 %x) "sign-return-address"="all" "sign-return-address-key"="a_key" "zero-call-used-regs"="all" nounwind {
; CHECK-LABEL: leaf_a:
; V8A:         hint #25
; V83A:        paciasp
  %add = add i32 %x, 1
; CHECK:       mov x1, #0
; CHECK:       movi v31.2d, #0000000000000000
; V8A-NEXT:    hint #29
; V8A-NEXT:    ret
; V83A-NEXT:   retaa
  ret i32 %add
}

define i32 @leaf_b(i32 %x) "sign-return-address"="all" "sign-return-address-key"="b_key" "zero-call-used-regs"="all" nounwind {
; CHECK-LABEL: leaf_b:
; V8A:         hint #27
; V83A:        pacibsp
  %add = add i32 %x, 1
; CHECK:       mov x1, #0
; CHECK:       movi v31.2d, #0000000000000000
; V8A-NEXT:    hint #31
; V8A-NEXT:    ret
; V83A-NEXT:   retab
  ret i32 %add
}

declare void @ext()

define void @nonleaf_a() "sign-return-address"="non-leaf" "sign-return-address-key"="a_key" "zero-call-used-regs"="all" nounwind {
; CHECK-LABEL: nonleaf_a:
; V8A:         hint #25
; V83A:        paciasp
  call void @ext()
; CHECK:       bl ext
; CHECK:       ldr x30, [sp], #16
; CHECK:       mov x0, #0
; CHECK:       movi v31.2d, #0000000000000000
; V8A-NEXT:    hint #29
; V8A-NEXT:    ret
; V83A-NEXT:   retaa
  ret void
}

define void @nonleaf_b() "sign-return-address"="non-leaf" "sign-return-address-key"="b_key" "zero-call-used-regs"="all" nounwind {
; CHECK-LABEL: nonleaf_b:
; V8A:         hint #27
; V83A:        pacibsp
  call void @ext()
; CHECK:       bl ext
; CHECK:       ldr x30, [sp], #16
; CHECK:       mov x0, #0
; CHECK:       movi v31.2d, #0000000000000000
; V8A-NEXT:    hint #31
; V8A-NEXT:    ret
; V83A-NEXT:   retab
  ret void
}

define i32 @leaf_sve(i32 %x) "target-features"="+sve" "sign-return-address"="all" "sign-return-address-key"="a_key" "zero-call-used-regs"="all" nounwind {
; CHECK-LABEL: leaf_sve:
  %add = add i32 %x, 1
; CHECK:       mov x1, #0
; CHECK:       mov z31.d, #0
; CHECK:       pfalse p15.b
; V8A-NEXT:    hint #29
; V8A-NEXT:    ret
; V83A-NEXT:   retaa
  ret i32 %add
}

define i32 @leaf_pauthlr(i32 %x) "branch-protection-pauth-lr" "sign-return-address"="all" "zero-call-used-regs"="all-gpr" nounwind {
; CHECK-LABEL: leaf_pauthlr:
  %add = add i32 %x, 1
; CHECK:       mov x1, #0
; CHECK:       mov x18, #0
; V8A:         hint #29
; V83A:        autiasp
; CHECK-NEXT:  mov x16, #0
; CHECK-NEXT:  ret
  ret i32 %add
}
