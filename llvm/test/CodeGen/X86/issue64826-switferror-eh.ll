; RUN: llc %s -filetype=obj -o - | llvm-readobj -r - | FileCheck %s --check-prefix=RELOC
; RUN: llc %s -o - | FileCheck %s --check-prefix=ASM

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx"

declare void @issue64826(i64, ptr, ptr swifterror)

define swiftcc void @rdar113994760() personality ptr @__gcc_personality_v0 {
entry:
  %swifterror = alloca swifterror ptr, align 8
  invoke swiftcc void @issue64826(i64 0, ptr null, ptr swifterror %swifterror)
          to label %.noexc unwind label %tsan_cleanup

.noexc:                                           ; preds = %entry
  ret void

tsan_cleanup:                                     ; preds = %entry
  %cleanup.lpad = landingpad { ptr, i32 }
          cleanup
  resume { ptr, i32 } zeroinitializer
}

declare i32 @__gcc_personality_v0(...)

; RELOC-LABEL: Relocations [
; RELOC-NEXT:    Section __text {
; RELOC-NEXT:      0x19 1 2 1 X86_64_RELOC_BRANCH 0 __Unwind_Resume
; RELOC-NEXT:      0xB 1 2 1 X86_64_RELOC_BRANCH 0 _issue64826
; RELOC-NEXT:    }
; RELOC-NEXT:    Section __eh_frame {
; RELOC-NEXT:      0x13 1 2 1 X86_64_RELOC_GOT 0 ___gcc_personality_v0
; RELOC-NEXT:    }
; RELOC-NEXT:  ]

; ASM-LABEL: rdar113994760:
; ASM:       ## %bb.0: ## %entry
; ASM-NEXT:    pushq %r12
; ASM-NEXT:    .cfi_def_cfa_offset 16
; ASM-NEXT:    subq $16, %rsp
; ASM-NEXT:    .cfi_def_cfa_offset 32
; ASM-NEXT:    .cfi_offset %r12, -16
; ASM-NEXT:  Ltmp0:
; ASM-NEXT:    xorl %edi, %edi
; ASM-NEXT:    xorl %esi, %esi
; ASM-NEXT:    callq   _issue64826
; ASM-NEXT:  Ltmp1:
; ASM-NEXT:  ## %bb.1: ## %.noexc
; ASM-NEXT:    addq $16, %rsp
; ASM-NEXT:    popq %r12
; ASM-NEXT:    retq
; ASM-NEXT:  LBB0_2: ## %tsan_cleanup
; ASM-NEXT:  Ltmp2:
; ASM-NEXT:    xorl %edi, %edi
; ASM-NEXT:    callq __Unwind_Resume
; ASM-NEXT:  Lfunc_end0:
; ASM-NEXT:    .cfi_endproc
; ASM-NEXT:    .section        __TEXT,__gcc_except_tab
; ASM-NEXT:    .p2align        2, 0x0
; ASM-NEXT: GCC_except_table0:
; ASM-NEXT: Lexception0:
; ASM-NEXT:    .byte   255                             ## @LPStart Encoding = omit
; ASM-NEXT:    .byte   255                             ## @TType Encoding = omit
; ASM-NEXT:    .byte   1                               ## Call site Encoding = uleb128
; ASM-NEXT:    .uleb128 Lcst_end0-Lcst_begin0
; ASM-NEXT: Lcst_begin0:
; ASM-NEXT:    .uleb128 Ltmp0-Lfunc_begin0             ## >> Call Site 1 <<
; ASM-NEXT:    .uleb128 Ltmp1-Ltmp0                    ##   Call between Ltmp0 and Ltmp1
; ASM-NEXT:    .uleb128 Ltmp2-Lfunc_begin0             ##     jumps to Ltmp2
; ASM-NEXT:    .byte   0                               ##   On action: cleanup
; ASM-NEXT:    .uleb128 Ltmp1-Lfunc_begin0             ## >> Call Site 2 <<
; ASM-NEXT:    .uleb128 Lfunc_end0-Ltmp1               ##   Call between Ltmp1 and Lfunc_end0
; ASM-NEXT:    .byte   0                               ##     has no landing pad
; ASM-NEXT:    .byte   0                               ##   On action: cleanup
; ASM-NEXT: Lcst_end0:
; ASM-NEXT:    .p2align        2, 0x0
; ASM-NEXT:                                         ## -- End function
