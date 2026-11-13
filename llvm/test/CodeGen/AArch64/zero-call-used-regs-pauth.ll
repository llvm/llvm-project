; Check epilogue insertion order independently of post-RA scheduling.
; RUN: llc -verify-machineinstrs -enable-post-misched=false -enable-machine-outliner=never -mtriple aarch64 %s -o - | \
; RUN:   FileCheck %s --check-prefixes CHECK,V8A
; RUN: llc -verify-machineinstrs -enable-post-misched=false -enable-machine-outliner=never -mtriple aarch64 -mattr=+v8.3a %s -o - | \
; RUN:   FileCheck %s --check-prefixes CHECK,V83A
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

%large = type { [9 x i64] }

define i32 @leaf_harden_a(i32 %x) "sign-return-address"="all" "sign-return-address-key"="a_key" "sign-return-address-harden"="load-return-address" "zero-call-used-regs"="all" nounwind {
; CHECK-LABEL: leaf_harden_a:
; V8A:         hint #25
; V83A:        paciasp
  %add = add i32 %x, 1
; CHECK:       mov x1, #0
; CHECK:       movi v31.2d, #0000000000000000
; V8A-NEXT:    hint #29
; V8A-NEXT:    mov x8, x30
; V8A-NEXT:    hint #7
; V8A-NEXT:    ldr w30, [x30]
; V8A-NEXT:    ret x8
; V83A-NEXT:   autiasp
; V83A-NEXT:   mov x8, x30
; V83A-NEXT:   xpaci x8
; V83A-NEXT:   ldr w8, [x8]
; V83A-NEXT:   ret
  ret i32 %add
}

define i32 @leaf_harden_b(i32 %x) "sign-return-address"="all" "sign-return-address-key"="b_key" "sign-return-address-harden"="load-return-address" "zero-call-used-regs"="all" nounwind {
; CHECK-LABEL: leaf_harden_b:
; V8A:         hint #27
; V83A:        pacibsp
  %add = add i32 %x, 1
; CHECK:       mov x1, #0
; CHECK:       movi v31.2d, #0000000000000000
; V8A-NEXT:    hint #31
; V8A-NEXT:    mov x8, x30
; V8A-NEXT:    hint #7
; V8A-NEXT:    ldr w30, [x30]
; V8A-NEXT:    ret x8
; V83A-NEXT:   autibsp
; V83A-NEXT:   mov x8, x30
; V83A-NEXT:   xpaci x8
; V83A-NEXT:   ldr w8, [x8]
; V83A-NEXT:   ret
  ret i32 %add
}

define void @nonleaf_harden_a() "sign-return-address"="non-leaf" "sign-return-address-key"="a_key" "sign-return-address-harden"="load-return-address" "zero-call-used-regs"="all" nounwind {
; CHECK-LABEL: nonleaf_harden_a:
; V8A:         hint #25
; V83A:        paciasp
  call void @ext()
; CHECK:       bl ext
; CHECK:       ldr x30, [sp], #16
; CHECK:       mov x0, #0
; CHECK:       movi v31.2d, #0000000000000000
; V8A-NEXT:    hint #29
; V8A-NEXT:    mov x8, x30
; V8A-NEXT:    hint #7
; V8A-NEXT:    ldr w30, [x30]
; V8A-NEXT:    ret x8
; V83A-NEXT:   autiasp
; V83A-NEXT:   mov x8, x30
; V83A-NEXT:   xpaci x8
; V83A-NEXT:   ldr w8, [x8]
; V83A-NEXT:   ret
  ret void
}

define i32 @leaf_pauthlr_b(i32 %x) "branch-protection-pauth-lr" "sign-return-address"="all" "sign-return-address-key"="b_key" "zero-call-used-regs"="all-gpr" nounwind {
; CHECK-LABEL: leaf_pauthlr_b:
  %add = add i32 %x, 1
; CHECK:       mov x1, #0
; CHECK:       mov x18, #0
; V8A:         hint #31
; V83A:        autibsp
; CHECK-NEXT:  mov x16, #0
; CHECK-NEXT:  ret
  ret i32 %add
}

define void @nonleaf_pauthlr() "branch-protection-pauth-lr" "sign-return-address"="non-leaf" "sign-return-address-key"="a_key" "zero-call-used-regs"="all-gpr" nounwind {
; CHECK-LABEL: nonleaf_pauthlr:
  call void @ext()
; CHECK:       bl ext
; CHECK:       ldr x30, [sp], #16
; CHECK:       mov x0, #0
; CHECK:       mov x18, #0
; V8A:         hint #29
; V83A:        autiasp
; CHECK-NEXT:  mov x16, #0
; CHECK-NEXT:  ret
  ret void
}

define i32 @leaf_pauthlr_feat(i32 %x) "target-features"="+pauth-lr" "branch-protection-pauth-lr" "sign-return-address"="all" "zero-call-used-regs"="all-gpr" nounwind {
; CHECK-LABEL: leaf_pauthlr_feat:
  %add = add i32 %x, 1
; CHECK:       mov x1, #0
; CHECK:       mov x18, #0
; V8A-NEXT:    autiasppc .Ltmp[[#]]
; V8A-NEXT:    ret
; V83A-NEXT:   retaasppc .Ltmp[[#]]
  ret i32 %add
}

define swifttailcc void @tail_stack(%large %s) "sign-return-address"="all" "sign-return-address-key"="a_key" "zero-call-used-regs"="all-gpr" nounwind {
; CHECK-LABEL: tail_stack:
; V8A:         hint #25
; V83A:        paciasp
  call void @ext()
; CHECK:       bl ext
; CHECK:       ldr x30, [sp], #16
; CHECK:       add sp, sp, #16
; CHECK:       mov x0, #0
; CHECK:       mov x18, #0
; V8A:         sub x16, sp, #16
; V8A-NEXT:    mov x17, x30
; V8A-NEXT:    hint #12
; V8A-NEXT:    mov x30, x17
; V83A:        sub x16, sp, #16
; V83A-NEXT:   autia x30, x16
; CHECK-NEXT:  mov x16, #0
; CHECK-NEXT:  mov x17, #0
; CHECK-NEXT:  ret
  ret void
}

define swifttailcc void @tail_stack_b(%large %s) "sign-return-address"="all" "sign-return-address-key"="b_key" "zero-call-used-regs"="all-gpr" nounwind {
; CHECK-LABEL: tail_stack_b:
; V8A:         hint #27
; V83A:        pacibsp
  call void @ext()
; CHECK:       bl ext
; CHECK:       ldr x30, [sp], #16
; CHECK:       add sp, sp, #16
; CHECK:       mov x0, #0
; CHECK:       mov x18, #0
; V8A:         sub x16, sp, #16
; V8A-NEXT:    mov x17, x30
; V8A-NEXT:    hint #14
; V8A-NEXT:    mov x30, x17
; V83A:        sub x16, sp, #16
; V83A-NEXT:   autib x30, x16
; CHECK-NEXT:  mov x16, #0
; CHECK-NEXT:  mov x17, #0
; CHECK-NEXT:  ret
  ret void
}

define swifttailcc void @tail_stack_pauthlr(%large %s) "branch-protection-pauth-lr" "sign-return-address"="all" "sign-return-address-key"="a_key" "zero-call-used-regs"="all-gpr" nounwind {
; CHECK-LABEL: tail_stack_pauthlr:
  call void @ext()
; CHECK:       bl ext
; CHECK:       ldr x30, [sp], #16
; CHECK:       add sp, sp, #16
; CHECK:       mov x0, #0
; CHECK:       mov x18, #0
; CHECK:       sub x16, sp, #16
; CHECK-NEXT:  mov x17, x30
; CHECK-NEXT:  adrp x15, .Ltmp[[#]]
; CHECK-NEXT:  add x15, x15, :lo12:.Ltmp[[#]]
; V8A-NEXT:    hint #39
; V8A-NEXT:    hint #12
; V83A-NEXT:   hint #39
; V83A-NEXT:   autia1716
; CHECK-NEXT:  mov x30, x17
; CHECK-NEXT:  mov x15, #0
; CHECK-NEXT:  mov x16, #0
; CHECK-NEXT:  mov x17, #0
; CHECK-NEXT:  ret
  ret void
}

define swifttailcc void @tail_stack_pauthlr_feat(%large %s) "target-features"="+pauth-lr" "branch-protection-pauth-lr" "sign-return-address"="all" "sign-return-address-key"="a_key" "zero-call-used-regs"="all-gpr" nounwind {
; CHECK-LABEL: tail_stack_pauthlr_feat:
  call void @ext()
; CHECK:       bl ext
; CHECK:       ldr x30, [sp], #16
; CHECK:       add sp, sp, #16
; CHECK:       mov x0, #0
; CHECK:       mov x18, #0
; CHECK:       sub x16, sp, #16
; CHECK-NEXT:  mov x17, x30
; CHECK-NEXT:  adrp x15, .Ltmp[[#]]
; CHECK-NEXT:  add x15, x15, :lo12:.Ltmp[[#]]
; CHECK-NEXT:  autia171615
; CHECK-NEXT:  mov x30, x17
; CHECK-NEXT:  mov x15, #0
; CHECK-NEXT:  mov x16, #0
; CHECK-NEXT:  mov x17, #0
; CHECK-NEXT:  ret
  ret void
}

define swifttailcc void @tail_stack_harden(%large %s) "sign-return-address"="all" "sign-return-address-key"="a_key" "sign-return-address-harden"="load-return-address" "zero-call-used-regs"="all-gpr" nounwind {
; CHECK-LABEL: tail_stack_harden:
  call void @ext()
; CHECK:       bl ext
; CHECK:       ldr x30, [sp], #16
; CHECK:       add sp, sp, #16
; CHECK:       mov x0, #0
; CHECK:       mov x18, #0
; V8A:         sub x16, sp, #16
; V8A-NEXT:    mov x17, x30
; V8A-NEXT:    hint #12
; V8A-NEXT:    mov x30, x17
; V83A:        sub x16, sp, #16
; V83A-NEXT:   autia x30, x16
; CHECK-NEXT:  mov x16, #0
; CHECK-NEXT:  mov x17, #0
; V8A-NEXT:    mov x8, x30
; V8A-NEXT:    hint #7
; V8A-NEXT:    ldr w30, [x30]
; V8A-NEXT:    ret x8
; V83A-NEXT:   mov x8, x30
; V83A-NEXT:   xpaci x8
; V83A-NEXT:   ldr w8, [x8]
; V83A-NEXT:   ret
  ret void
}
