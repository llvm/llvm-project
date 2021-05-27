; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:     -mattr=-altivec -xcoff-traceback-table=true < %s | \
; RUN:   FileCheck --check-prefixes=CHECK-ASM,COMMON %s

; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -function-sections \
; RUN:     -mcpu=pwr4 -mattr=-altivec < %s | \
; RUN:   FileCheck --check-prefixes=CHECK-FUNC,COMMON %s

define float @bar() #0 {
entry:
  %fvalue = alloca float, align 4
  %taken = alloca i32, align 4
  %data = alloca i32, align 4
  store float 1.000000e+00, float* %fvalue, align 4
  %0 = load float, float* %fvalue, align 4
  %1 = call float asm "fneg $0,$1\0A\09", "=b,b,~{f31},~{f30},~{f29},~{f28},~{f27}"(float %0)
  store float %1, float* %fvalue, align 4
  store i32 123, i32* %data, align 4
  %2 = load i32, i32* %data, align 4
  %3 = call i32 asm "cntlzw $0, $1\0A\09", "=b,b,~{r31},~{r30},~{r29},~{r28}"(i32 %2)
  store i32 %3, i32* %taken, align 4
  %4 = load i32, i32* %taken, align 4
  %conv = sitofp i32 %4 to float
  %5 = load float, float* %fvalue, align 4
  %add = fadd float %conv, %5
  ret float %add
}

; COMMON:       .vbyte  4, 0x00000000                   # Traceback table begin
; COMMON-NEXT:  .byte   0x00                            # Version = 0
; COMMON-NEXT:  .byte   0x09                            # Language = CPlusPlus
; COMMON-NEXT:  .byte   0x22                            # -IsGlobaLinkage, -IsOutOfLineEpilogOrPrologue
; COMMON-NEXT:                                        # +HasTraceBackTableOffset, -IsInternalProcedure
; COMMON-NEXT:                                        # -HasControlledStorage, -IsTOCless
; COMMON-NEXT:                                        # +IsFloatingPointPresent
; COMMON-NEXT:                                        # -IsFloatingPointOperationLogOrAbortEnabled
; COMMON-NEXT:  .byte   0x60                            # -IsInterruptHandler, +IsFunctionNamePresent, +IsAllocaUsed
; COMMON-NEXT:                                        # OnConditionDirective = 0, -IsCRSaved, -IsLRSaved
; COMMON-NEXT:  .byte   0x85                            # +IsBackChainStored, -IsFixup, NumOfFPRsSaved = 5
; COMMON-NEXT:  .byte   0x04                            # -HasVectorInfo, -HasExtensionTable, NumOfGPRsSaved = 4
; COMMON-NEXT:  .byte   0x00                            # NumberOfFixedParms = 0
; COMMON-NEXT:  .byte   0x01                            # NumberOfFPParms = 0, +HasParmsOnStack
; CHECK-ASM-NEXT:   .vbyte  4, L..bar0-.bar                 # Function size
; CHECK-FUNC-NEXT:  .vbyte  4, L..bar0-.bar[PR]             # Function size
; COMMON-NEXT:  .vbyte  2, 0x0003                       # Function name len = 3
; COMMON-NEXT:  .byte   "bar"                           # Function Name
; COMMON-NEXT:  .byte   0x1f                            # AllocaUsed
; COMMON-NEXT:                                        # -- End function
