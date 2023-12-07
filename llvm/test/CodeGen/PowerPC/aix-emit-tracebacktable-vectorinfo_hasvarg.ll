; RUN:  llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr7 \
; RUN:     -mattr=+altivec  -vec-extabi -xcoff-traceback-table=true 2>&1 < %s | \
; RUN:   FileCheck --check-prefixes=CHECK-ASM %s 

;; void f(vector float vf, ...) {
;;}

define void @f(<4 x float> %vf, ...) #0 {
entry:
  %vf.addr = alloca <4 x float>, align 16
  store <4 x float> %vf, ptr %vf.addr, align 16
  ret void
}

;CHECK-ASM:             .vbyte  4, 0x00000000                   # Traceback table begin
;CHECK-ASM-NEXT:        .byte   0x00                            # Version = 0
;CHECK-ASM-NEXT:        .byte   0x09                            # Language = CPlusPlus
;CHECK-ASM-NEXT:        .byte   0x20                            # -IsGlobaLinkage, -IsOutOfLineEpilogOrPrologue
;CHECK-ASM-NEXT:                                         # +HasTraceBackTableOffset, -IsInternalProcedure
;CHECK-ASM-NEXT:                                         # -HasControlledStorage, -IsTOCless
;CHECK-ASM-NEXT:                                         # -IsFloatingPointPresent
;CHECK-ASM-NEXT:                                         # -IsFloatingPointOperationLogOrAbortEnabled
;CHECK-ASM-NEXT:        .byte   0x40                            # -IsInterruptHandler, +IsFunctionNamePresent, -IsAllocaUsed
;CHECK-ASM-NEXT:                                         # OnConditionDirective = 0, -IsCRSaved, -IsLRSaved
;CHECK-ASM-NEXT:        .byte   0x00                            # -IsBackChainStored, -IsFixup, NumOfFPRsSaved = 0
;CHECK-ASM-NEXT:        .byte   0x40                            # -HasExtensionTable, +HasVectorInfo, NumOfGPRsSaved = 0
;CHECK-ASM-NEXT:        .byte   0x00                            # NumberOfFixedParms = 0
;CHECK-ASM-NEXT:        .byte   0x01                            # NumberOfFPParms = 0, +HasParmsOnStack
;CHECK-ASM-NEXT:        .vbyte  4, L..f0-.f                     # Function size
;CHECK-ASM-NEXT:        .vbyte  2, 0x0001                       # Function name len = 1
;CHECK-ASM-NEXT:        .byte   102                             # Function Name
;CHECK-ASM-NEXT:        .byte   0x01                            # NumOfVRsSaved = 0, -IsVRSavedOnStack, +HasVarArgs
;CHECK-ASM-NEXT:        .byte   0x03                            # NumOfVectorParams = 1, +HasVMXInstruction
;CHECK-ASM-NEXT:        .vbyte  4, 0xc0000000                   # Vector Parameter type = vf
;CHECK-ASM-NEXT:        .vbyte  2, 0x0000                       # Padding 
;CHECK-ASM-NEXT:                                         # -- End function
