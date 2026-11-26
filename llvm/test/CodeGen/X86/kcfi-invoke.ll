; RUN: llc -mtriple=x86_64-unknown-linux-gnu -verify-machineinstrs < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -O0 -verify-machineinstrs < %s | FileCheck %s --check-prefix=ASM-O0
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -stop-after=finalize-isel < %s | FileCheck %s --check-prefixes=MIR,ISEL
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -stop-after=kcfi < %s | FileCheck %s --check-prefixes=MIR,KCFI
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -stop-after=kcfi -enable-new-pm < %s | FileCheck %s --check-prefixes=MIR,KCFI

;; An indirect call that is an invoke has a KCFI check like any other, and the
;; check is inside of the invoke's try-range. The landing pad does something
;; before it resumes, or the invoke would be turned into a call before
;; instruction selection.

declare i32 @__gxx_personality_v0(...)
declare void @cleanup()

@g = external local_unnamed_addr global ptr, align 8

define void @f1(ptr noundef %x) personality ptr @__gxx_personality_v0 {
; ASM-LABEL: f1:
; ASM:       .Ltmp{{[0-9]+}}: # EH_LABEL
; ASM-NEXT:    movl $4282621618, %r10d # imm = 0xFF439EB2
; ASM-NEXT:    addl -4(%rdi), %r10d
; ASM-NEXT:    je [[PASS:.Ltmp[0-9]+]]
; ASM-NEXT:  [[TRAP:.Ltmp[0-9]+]]:
; ASM-NEXT:    ud2
; ASM-NEXT:    .section .kcfi_traps,"ao",@progbits,.text
; ASM-NEXT:  [[ENTRY:.Ltmp[0-9]+]]:
; ASM-NEXT:    .long [[TRAP]]-[[ENTRY]]
; ASM-NEXT:    .text
; ASM-NEXT:  [[PASS]]:
; ASM-NEXT:    callq *%rdi
; ASM-NEXT:  .Ltmp{{[0-9]+}}: # EH_LABEL

; ASM-O0-LABEL: f1:
; ASM-O0:       .Ltmp{{[0-9]+}}: # EH_LABEL
; ASM-O0:         movl $4282621618, %r10d # imm = 0xFF439EB2
; ASM-O0-NEXT:    addl -4([[REG:%r[a-z0-9]+]]), %r10d
; ASM-O0-NEXT:    je [[PASS:.Ltmp[0-9]+]]
; ASM-O0-NEXT:  [[TRAP:.Ltmp[0-9]+]]:
; ASM-O0-NEXT:    ud2
; ASM-O0:       [[PASS]]:
; ASM-O0-NEXT:    callq *[[REG]]
; ASM-O0-NEXT:  .Ltmp{{[0-9]+}}: # EH_LABEL

; MIR-LABEL: name: f1
; MIR: body:

; ISEL:       EH_LABEL
; ISEL:       CALL64r %0, csr_64, implicit $rsp, implicit $ssp, implicit-def $rsp, implicit-def $ssp, cfi-type 12345678
; ISEL:       EH_LABEL

; KCFI:       EH_LABEL
; KCFI-NEXT:  BUNDLE{{.*}} {
; KCFI-NEXT:    KCFI_CHECK $rdi, 12345678, implicit-def $r10, implicit-def $r11, implicit-def $eflags
; KCFI-NEXT:    CALL64r killed $rdi, csr_64, implicit $rsp, implicit $ssp, implicit-def $rsp, implicit-def $ssp
; KCFI-NEXT:  }
; KCFI-NEXT:  EH_LABEL
  invoke void %x() [ "kcfi"(i32 12345678) ]
    to label %cont
    unwind label %err
cont:
  ret void
err:
  %exn = landingpad { ptr, i32 }
          cleanup
  call void @cleanup()
  resume { ptr, i32 } %exn
}

define void @f2() personality ptr @__gxx_personality_v0 {
; ASM-LABEL: f2:
; ASM:         movq g@GOTPCREL(%rip), %rax
; ASM-NEXT:    movq (%rax), %rax
; ASM-NEXT:  .Ltmp{{[0-9]+}}: # EH_LABEL
; ASM-NEXT:    movl $4282621618, %r10d # imm = 0xFF439EB2
; ASM-NEXT:    addl -4(%rax), %r10d
; ASM-NEXT:    je [[PASS:.Ltmp[0-9]+]]
; ASM-NEXT:  [[TRAP:.Ltmp[0-9]+]]:
; ASM-NEXT:    ud2
; ASM-NEXT:    .section .kcfi_traps,"ao",@progbits,.text
; ASM-NEXT:  [[ENTRY:.Ltmp[0-9]+]]:
; ASM-NEXT:    .long [[TRAP]]-[[ENTRY]]
; ASM-NEXT:    .text
; ASM-NEXT:  [[PASS]]:
; ASM-NEXT:    callq *%rax
; ASM-NEXT:  .Ltmp{{[0-9]+}}: # EH_LABEL

; ASM-O0-LABEL: f2:
; ASM-O0:       .Ltmp{{[0-9]+}}: # EH_LABEL
; ASM-O0:         movl $4282621618, %r10d # imm = 0xFF439EB2
; ASM-O0-NEXT:    addl -4([[REG:%r[a-z0-9]+]]), %r10d
; ASM-O0-NEXT:    je [[PASS:.Ltmp[0-9]+]]
; ASM-O0-NEXT:  [[TRAP:.Ltmp[0-9]+]]:
; ASM-O0-NEXT:    ud2
; ASM-O0:       [[PASS]]:
; ASM-O0-NEXT:    callq *[[REG]]
; ASM-O0-NEXT:  .Ltmp{{[0-9]+}}: # EH_LABEL

; MIR-LABEL: name: f2
; MIR: body:

; ISEL:       EH_LABEL
; ISEL:       CALL64r %1, csr_64, implicit $rsp, implicit $ssp, implicit-def $rsp, implicit-def $ssp, cfi-type 12345678
; ISEL:       EH_LABEL

; KCFI:       renamable $rax = MOV64rm killed renamable $rax, 1, $noreg, 0, $noreg
; KCFI-NEXT:  EH_LABEL
; KCFI-NEXT:  BUNDLE{{.*}} {
; KCFI-NEXT:    KCFI_CHECK $rax, 12345678, implicit-def $r10, implicit-def $r11, implicit-def $eflags
; KCFI-NEXT:    CALL64r killed $rax, csr_64, implicit $rsp, implicit $ssp, implicit-def $rsp, implicit-def $ssp
; KCFI-NEXT:  }
; KCFI-NEXT:  EH_LABEL
  %1 = load ptr, ptr @g, align 8
  invoke void %1() [ "kcfi"(i32 12345678) ]
    to label %cont
    unwind label %err
cont:
  ret void
err:
  %exn = landingpad { ptr, i32 }
          cleanup
  call void @cleanup()
  resume { ptr, i32 } %exn
}

define void @f3(ptr noundef %x) #0 personality ptr @__gxx_personality_v0 {
; ASM-LABEL: f3:
; ASM:       .Ltmp{{[0-9]+}}: # EH_LABEL
; ASM-NEXT:    movq %rdi, %r11
; ASM-NEXT:    movl $4282621618, %r10d # imm = 0xFF439EB2
; ASM-NEXT:    addl -4(%r11), %r10d
; ASM-NEXT:    je [[PASS:.Ltmp[0-9]+]]
; ASM-NEXT:  [[TRAP:.Ltmp[0-9]+]]:
; ASM-NEXT:    ud2
; ASM-NEXT:    .section .kcfi_traps,"ao",@progbits,.text
; ASM-NEXT:  [[ENTRY:.Ltmp[0-9]+]]:
; ASM-NEXT:    .long [[TRAP]]-[[ENTRY]]
; ASM-NEXT:    .text
; ASM-NEXT:  [[PASS]]:
; ASM-NEXT:    callq __llvm_retpoline_r11
; ASM-NEXT:  .Ltmp{{[0-9]+}}: # EH_LABEL

; ASM-O0-LABEL: f3:
; ASM-O0:       .Ltmp{{[0-9]+}}: # EH_LABEL
; ASM-O0:         movl $4282621618, %r10d # imm = 0xFF439EB2
; ASM-O0-NEXT:    addl -4(%r11), %r10d
; ASM-O0-NEXT:    je [[PASS:.Ltmp[0-9]+]]
; ASM-O0-NEXT:  [[TRAP:.Ltmp[0-9]+]]:
; ASM-O0-NEXT:    ud2
; ASM-O0:       [[PASS]]:
; ASM-O0-NEXT:    callq __llvm_retpoline_r11
; ASM-O0-NEXT:  .Ltmp{{[0-9]+}}: # EH_LABEL

; MIR-LABEL: name: f3
; MIR: body:

; ISEL:       EH_LABEL
; ISEL:       CALL64pcrel32 &__llvm_retpoline_r11, csr_64, implicit $rsp, implicit $ssp, implicit-def $rsp, implicit-def $ssp, implicit killed $r11, cfi-type 12345678
; ISEL:       EH_LABEL

; KCFI:       EH_LABEL
; KCFI-NEXT:  $r11 = MOV64rr killed $rdi
; KCFI-NEXT:  BUNDLE{{.*}} {
; KCFI-NEXT:    KCFI_CHECK $r11, 12345678, implicit-def $r10, implicit-def $r11, implicit-def $eflags
; KCFI-NEXT:    CALL64pcrel32 &__llvm_retpoline_r11, csr_64, implicit $rsp, implicit $ssp, implicit-def $rsp, implicit-def $ssp, implicit internal $r11
; KCFI-NEXT:  }
; KCFI-NEXT:  EH_LABEL
  invoke void %x() [ "kcfi"(i32 12345678) ]
    to label %cont
    unwind label %err
cont:
  ret void
err:
  %exn = landingpad { ptr, i32 }
          cleanup
  call void @cleanup()
  resume { ptr, i32 } %exn
}

attributes #0 = { "target-features"="+retpoline-indirect-branches,+retpoline-indirect-calls" }

!llvm.module.flags = !{!0}
!0 = !{i32 4, !"kcfi", i32 1}
