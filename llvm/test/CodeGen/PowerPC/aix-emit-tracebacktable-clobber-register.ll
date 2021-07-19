; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:     -mattr=+altivec -vec-extabi -xcoff-traceback-table=true < %s | \
; RUN:   FileCheck --check-prefixes=CHECK-ASM,COMMON %s

; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -function-sections \
; RUN:     -mcpu=pwr4 -mattr=+altivec -vec-extabi  < %s | \
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

define <4 x i32> @foov() #0 {
entry:
  %taken = alloca <4 x i32>, align 16
  %data = alloca <4 x i32>, align 16
  store <4 x i32> <i32 123, i32 0, i32 0, i32 0>, <4 x i32>* %data, align 16
  call void asm sideeffect "", "~{v31},~{v30},~{v29},~{v28}"() 
  %0 = load <4 x i32>, <4 x i32>* %taken, align 16
  ret <4 x i32> %0
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
; COMMON-NEXT:  .byte   0x04                            # -HasExtensionTable, -HasVectorInfo, NumOfGPRsSaved = 4
; COMMON-NEXT:  .byte   0x00                            # NumberOfFixedParms = 0
; COMMON-NEXT:  .byte   0x01                            # NumberOfFPParms = 0, +HasParmsOnStack
; CHECK-ASM-NEXT:   .vbyte  4, L..bar0-.bar                 # Function size
; CHECK-FUNC-NEXT:  .vbyte  4, L..bar0-.bar[PR]             # Function size
; COMMON-NEXT:  .vbyte  2, 0x0003                       # Function name len = 3
; COMMON-NEXT:  .byte   "bar"                           # Function Name
; COMMON-NEXT:  .byte   0x1f                            # AllocaUsed
; COMMON-NEXT:                                        # -- End function

; COMMON:     L..foov0:
; COMMON-NEXT:  .vbyte  4, 0x00000000                   # Traceback table begin
; COMMON-NEXT:  .byte   0x00                            # Version = 0
; COMMON-NEXT:  .byte   0x09                            # Language = CPlusPlus
; COMMON-NEXT:  .byte   0x20                            # -IsGlobaLinkage, -IsOutOfLineEpilogOrPrologue
; COMMON-NEXT:                                         # +HasTraceBackTableOffset, -IsInternalProcedure
; COMMON-NEXT:                                         # -HasControlledStorage, -IsTOCless
; COMMON-NEXT:                                         # -IsFloatingPointPresent
; COMMON-NEXT:                                         # -IsFloatingPointOperationLogOrAbortEnabled
; COMMON-NEXT:  .byte   0x40                            # -IsInterruptHandler, +IsFunctionNamePresent, -IsAllocaUsed
; COMMON-NEXT:                                         # OnConditionDirective = 0, -IsCRSaved, -IsLRSaved
; COMMON-NEXT:  .byte   0x80                            # +IsBackChainStored, -IsFixup, NumOfFPRsSaved = 0
; COMMON-NEXT:  .byte   0xc0                            # +HasExtensionTable, +HasVectorInfo, NumOfGPRsSaved = 0
; COMMON-NEXT:  .byte   0x00                            # NumberOfFixedParms = 0
; COMMON-NEXT:  .byte   0x01                            # NumberOfFPParms = 0, +HasParmsOnStack
; CHECK-ASM-NEXT:   .vbyte  4, L..foov0-.foov               # Function size
; CHECK-FUNC-NEXT:  .vbyte  4, L..foov0-.foov[PR]           # Function size
; COMMON-NEXT:  .vbyte  2, 0x0004                       # Function name len = 4
; COMMON-NEXT:  .byte   "foov"                          # Function Name 
; COMMON-NEXT:  .byte   0x12                            # NumOfVRsSaved = 4, +IsVRSavedOnStack, -HasVarArgs
; COMMON-NEXT:  .byte   0x01                            # NumOfVectorParams = 0, +HasVMXInstruction
; COMMON-NEXT:  .vbyte  4, 0x00000000                   # Vector Parameter type =
; COMMON-NEXT:  .vbyte  2, 0x0000                       # Padding 
; COMMON-NEXT:  .byte   0x08                            # ExtensionTableFlag = TB_EH_INFO
; COMMON-NEXT:  .align  2
; COMMON-NEXT:  .vbyte  4, L..C2-TOC[TC0]               # EHInfo Table

; COMMON:       .csect .eh_info_table[RW],2
; COMMON-NEXT:__ehinfo.1:
; COMMON-NEXT:  .vbyte  4, 0
; COMMON-NEXT:  .align  2
; COMMON-NEXT:  .vbyte  4, 0
; COMMON-NEXT:  .vbyte  4, 0
; CHECK-ASM-NEXT:  .csect .text[PR],2
; CHECK-FUNC-NEXT:  .csect .foov[PR],2
; COMMON-NEXT:                                         # -- End function
; COMMON:       .toc
; COMMON:      L..C2:
; COMMON-NEXT:  .tc __ehinfo.1[TC],__ehinfo.1
