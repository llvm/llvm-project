; RUN: llc -mtriple=x86_64-unknown-linux-gnu -verify-machineinstrs < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -verify-machineinstrs -stop-after=finalize-isel < %s | FileCheck %s --check-prefixes=MIR,ISEL
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -verify-machineinstrs -stop-after=kcfi < %s | FileCheck %s --check-prefixes=MIR,KCFI

; ASM:       .p2align 4, 0x90
; ASM:       .type __cfi_f1,@function
; ASM-LABEL: __cfi_f1:
; ASM-NEXT:    nop
; ASM-NEXT:    nop
; ASM-NEXT:    nop
; ASM-NEXT:    nop
; ASM-NEXT:    nop
; ASM-NEXT:    nop
; ASM-NEXT:    nop
; ASM-NEXT:    nop
; ASM-NEXT:    nop
; ASM-NEXT:    nop
; ASM-NEXT:    nop
; ASM-NEXT:    movl $12345678, %eax
; ASM-LABEL: .Lcfi_func_end0:
; ASM-NEXT:  .size   __cfi_f1, .Lcfi_func_end0-__cfi_f1
define void @f1(ptr noundef %x) !kcfi_type !1 {
; ASM-LABEL: f1:
; ASM:       # %bb.0:
; ASM:         movl $4282621618, %r10d # imm = 0xFF439EB2
; ASM-NEXT:    addl -4(%rdi), %r10d
; ASM-NEXT:    je .Ltmp0
; ASM-NEXT:  .Ltmp1:
; ASM-NEXT:    ud2
; ASM-NEXT:    .section .kcfi_traps,"ao",@progbits,.text
; ASM-NEXT:  .Ltmp2:
; ASM-NEXT:    .long .Ltmp1-.Ltmp2
; ASM-NEXT:    .text
; ASM-NEXT:  .Ltmp0:
; ASM-NEXT:    callq *%rdi

; MIR-LABEL: name: f1
; MIR: body:
; ISEL: CALL64r %0, csr_64, implicit $rsp, implicit $ssp, implicit-def $rsp, implicit-def $ssp, cfi-type 12345678
; KCFI:       BUNDLE{{.*}} {
; KCFI-NEXT:    KCFI_CHECK $rdi, 12345678, implicit-def $r10, implicit-def $r11, implicit-def $eflags
; KCFI-NEXT:    CALL64r killed $rdi, csr_64, implicit $rsp, implicit $ssp, implicit-def $rsp, implicit-def $ssp
; KCFI-NEXT:  }
  call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}

; ASM-NOT: __cfi_f2:
define void @f2(ptr noundef %x) {
; ASM-LABEL: f2:

; MIR-LABEL: name: f2
; MIR: body:
; ISEL: TCRETURNri64 %0, 0, csr_64, implicit $rsp, implicit $ssp, cfi-type 12345678
; KCFI:       BUNDLE{{.*}} {
; KCFI-NEXT:    KCFI_CHECK $rdi, 12345678, implicit-def $r10, implicit-def $r11, implicit-def $eflags
; KCFI-NEXT:    TAILJMPr64 killed $rdi, csr_64, implicit $rsp, implicit $ssp, implicit $rsp, implicit $ssp
; KCFI-NEXT:  }
  tail call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}

; ASM-NOT: __cfi_f3:
define void @f3(ptr noundef %x) #0 {
; ASM-LABEL: f3:
; MIR-LABEL: name: f3
; MIR: body:
; ISEL: CALL64pcrel32 &__llvm_retpoline_r11, csr_64, implicit $rsp, implicit $ssp, implicit-def $rsp, implicit-def $ssp, implicit killed $r11, cfi-type 12345678
; KCFI:       BUNDLE{{.*}} {
; KCFI-NEXT:    KCFI_CHECK $r11, 12345678, implicit-def $r10, implicit-def $r11, implicit-def $eflags
; KCFI-NEXT:    CALL64pcrel32 &__llvm_retpoline_r11, csr_64, implicit $rsp, implicit $ssp, implicit-def $rsp, implicit-def $ssp, implicit internal killed $r11
; KCFI-NEXT:  }
  call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}

; ASM-NOT: __cfi_f4:
define void @f4(ptr noundef %x) #0 {
; ASM-LABEL: f4:
; MIR-LABEL: name: f4
; MIR: body:
; ISEL: TCRETURNdi64 &__llvm_retpoline_r11, 0, csr_64, implicit $rsp, implicit $ssp, implicit killed $r11, cfi-type 12345678
; KCFI:       BUNDLE{{.*}} {
; KCFI-NEXT:    KCFI_CHECK $r11, 12345678, implicit-def $r10, implicit-def $r11, implicit-def $eflags
; KCFI-NEXT:    TAILJMPd64 &__llvm_retpoline_r11, csr_64, implicit $rsp, implicit $ssp, implicit $rsp, implicit $ssp, implicit internal killed $r11
; KCFI-NEXT:  }
  tail call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}

;; Ensure we emit Value + 1 for unwanted values (e.g. endbr64 == 4196274163).
; ASM-LABEL: __cfi_f5:
; ASM: movl $4196274164, %eax # imm = 0xFA1E0FF4
define void @f5(ptr noundef %x) !kcfi_type !2 {
; ASM-LABEL: f5:
; ASM: movl $98693132, %r10d # imm = 0x5E1F00C
  tail call void %x() [ "kcfi"(i32 4196274163) ]
  ret void
}

;; Ensure we emit Value + 1 for unwanted values (e.g. -endbr64 == 98693133).
; ASM-LABEL: __cfi_f6:
; ASM: movl $98693134, %eax # imm = 0x5E1F00E
define void @f6(ptr noundef %x) !kcfi_type !3 {
; ASM-LABEL: f6:
; ASM: movl $4196274162, %r10d # imm = 0xFA1E0FF2
  tail call void %x() [ "kcfi"(i32 98693133) ]
  ret void
}

@g = external local_unnamed_addr global ptr, align 8

define void @f7() {
; MIR-LABEL: name: f7
; MIR: body:
; ISEL: TCRETURNmi64 killed %0, 1, $noreg, 0, $noreg, 0, csr_64, implicit $rsp, implicit $ssp, cfi-type 12345678
; KCFI: $r11 = MOV64rm killed renamable $rax, 1, $noreg, 0, $noreg
; KCFI-NEXT:  BUNDLE{{.*}} {
; KCFI-NEXT:    KCFI_CHECK $r11, 12345678, implicit-def $r10, implicit-def $r11, implicit-def $eflags
; KCFI-NEXT:    TAILJMPr64 internal $r11, csr_64, implicit $rsp, implicit $ssp, implicit $rsp, implicit $ssp
; KCFI-NEXT:  }
  %1 = load ptr, ptr @g, align 8
  tail call void %1() [ "kcfi"(i32 12345678) ]
  ret void
}

define void @f8() {
; MIR-LABEL: name: f8
; MIR: body:
; ISEL: CALL64m killed %0, 1, $noreg, 0, $noreg, csr_64, implicit $rsp, implicit $ssp, implicit-def $rsp, implicit-def $ssp, cfi-type 12345678
; KCFI: $r11 = MOV64rm killed renamable $rax, 1, $noreg, 0, $noreg
; KCFI-NEXT:  BUNDLE{{.*}} {
; KCFI-NEXT:    KCFI_CHECK $r11, 12345678, implicit-def $r10, implicit-def $r11, implicit-def $eflags
; KCFI-NEXT:    CALL64r internal $r11, csr_64, implicit $rsp, implicit $ssp, implicit-def $rsp, implicit-def $ssp
; KCFI-NEXT:  }
  %1 = load ptr, ptr @g, align 8
  call void %1() [ "kcfi"(i32 12345678) ]
  ret void
}

attributes #0 = { "target-features"="+retpoline-indirect-branches,+retpoline-indirect-calls" }

!llvm.module.flags = !{!0}
!0 = !{i32 4, !"kcfi", i32 1}
!1 = !{i32 12345678}
!2 = !{i32 4196274163}
!3 = !{i32 98693133}
