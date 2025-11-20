; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:     -mattr=+altivec -vec-extabi -xcoff-traceback-table=true < %s | \
; RUN:   FileCheck --check-prefixes=CHECK-ASM,COMMON %s

; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -function-sections \
; RUN:     -mcpu=pwr4 -mattr=+altivec -vec-extabi  < %s | \
; RUN:   FileCheck --check-prefixes=CHECK-FUNC,COMMON %s

; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:     -mattr=+altivec -vec-extabi -xcoff-traceback-table=true -filetype=obj -o %t.o  < %s
; RUN: llvm-objdump -d --traceback-table --symbol-description %t.o | \
; RUN:   FileCheck --match-full-lines --strict-whitespace --check-prefixes=OBJ-DIS,NO-FUNC-SEC %s

; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:     -mattr=+altivec -vec-extabi -xcoff-traceback-table=true -function-sections -filetype=obj -o %t_func.o  < %s
; RUN: llvm-objdump -d --traceback-table --symbol-description %t_func.o | \
; RUN:   FileCheck --match-full-lines --strict-whitespace --check-prefixes=OBJ-DIS,FUNC-SEC %s

define float @bar() #0 {
entry:
  %fvalue = alloca float, align 4
  %taken = alloca i32, align 4
  %data = alloca i32, align 4
  store float 1.000000e+00, ptr %fvalue, align 4
  %0 = load float, ptr %fvalue, align 4
  %1 = call float asm "fneg $0,$1\0A\09", "=b,b,~{f31},~{f30},~{f29},~{f28},~{f27}"(float %0)
  store float %1, ptr %fvalue, align 4
  store i32 123, ptr %data, align 4
  %2 = load i32, ptr %data, align 4
  %3 = call i32 asm "cntlzw $0, $1\0A\09", "=b,b,~{r31},~{r30},~{r29},~{r28}"(i32 %2)
  store i32 %3, ptr %taken, align 4
  %4 = load i32, ptr %taken, align 4
  %conv = sitofp i32 %4 to float
  %5 = load float, ptr %fvalue, align 4
  %add = fadd float %conv, %5
  ret float %add
}

define <4 x i32> @foov() #0 {
entry:
  %taken = alloca <4 x i32>, align 16
  %data = alloca <4 x i32>, align 16
  store <4 x i32> <i32 123, i32 0, i32 0, i32 0>, ptr %data, align 16
  call void asm sideeffect "", "~{v31},~{v30},~{v29},~{v28}"() 
  %0 = load <4 x i32>, ptr %taken, align 16
  ret <4 x i32> %0
}

; COMMON:       .vbyte  4, 0x00000000                   # Traceback table begin
; COMMON-NEXT:  .byte   0x00                            # Version = 0
; COMMON-NEXT:  .byte   0x09                            # Language = CPlusPlus
; COMMON-NEXT:  .byte   0x22                            # -IsGlobalLinkage, -IsOutOfLineEpilogOrPrologue
; COMMON-NEXT:                                        # +HasTraceBackTableOffset, -IsInternalProcedure
; COMMON-NEXT:                                        # -HasControlledStorage, -IsTOCless
; COMMON-NEXT:                                        # +IsFloatingPointPresent
; COMMON-NEXT:                                        # -IsFloatingPointOperationLogOrAbortEnabled
; COMMON-NEXT:  .byte   0x40                            # -IsInterruptHandler, +IsFunctionNamePresent, -IsAllocaUsed
; COMMON-NEXT:                                        # OnConditionDirective = 0, -IsCRSaved, -IsLRSaved
; COMMON-NEXT:  .byte   0x05                            # -IsBackChainStored, -IsFixup, NumOfFPRsSaved = 5
; COMMON-NEXT:  .byte   0x04                            # -HasExtensionTable, -HasVectorInfo, NumOfGPRsSaved = 4
; COMMON-NEXT:  .byte   0x00                            # NumberOfFixedParms = 0
; COMMON-NEXT:  .byte   0x01                            # NumberOfFPParms = 0, +HasParmsOnStack
; CHECK-ASM-NEXT:   .vbyte  4, L..bar0-.bar                 # Function size
; CHECK-FUNC-NEXT:  .vbyte  4, L..bar0-.bar[PR]             # Function size
; COMMON-NEXT:  .vbyte  2, 0x0003                       # Function name len = 3
; COMMON-NEXT:  .byte   "bar"                           # Function Name
; COMMON-NEXT:                                        # -- End function

; COMMON:     L..foov0:
; COMMON-NEXT:  .vbyte  4, 0x00000000                   # Traceback table begin
; COMMON-NEXT:  .byte   0x00                            # Version = 0
; COMMON-NEXT:  .byte   0x09                            # Language = CPlusPlus
; COMMON-NEXT:  .byte   0x20                            # -IsGlobalLinkage, -IsOutOfLineEpilogOrPrologue
; COMMON-NEXT:                                         # +HasTraceBackTableOffset, -IsInternalProcedure
; COMMON-NEXT:                                         # -HasControlledStorage, -IsTOCless
; COMMON-NEXT:                                         # -IsFloatingPointPresent
; COMMON-NEXT:                                         # -IsFloatingPointOperationLogOrAbortEnabled
; COMMON-NEXT:  .byte   0x40                            # -IsInterruptHandler, +IsFunctionNamePresent, -IsAllocaUsed
; COMMON-NEXT:                                         # OnConditionDirective = 0, -IsCRSaved, -IsLRSaved
; COMMON-NEXT:  .byte   0x00                            # -IsBackChainStored, -IsFixup, NumOfFPRsSaved = 0
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
; CHECK-ASM-NEXT:   .csect ..text..[PR],5
; CHECK-FUNC-NEXT:  .csect .foov[PR],5
; COMMON-NEXT:                                         # -- End function
; COMMON:       .toc
; COMMON:      L..C2:
; COMMON-NEXT:  .tc __ehinfo.1[TE],__ehinfo.1


; OBJ-DIS:      9c: 00 00 00 00  	# Traceback table start
; OBJ-DIS-NEXT:      a0: 00           	# Version = 0
; OBJ-DIS-NEXT:      a1: 09           	# Language = CPlusPlus
; OBJ-DIS-NEXT:      a2: 22           	# -isGlobalLinkage, -isOutOfLineEpilogOrPrologue
; OBJ-DIS-NEXT:                       	  +hasTraceBackTableOffset, -isInternalProcedure
; OBJ-DIS-NEXT:                       	  -hasControlledStorage, -isTOCless
; OBJ-DIS-NEXT:                       	  +isFloatingPointPresent
; OBJ-DIS-NEXT:                       	  -isFloatingPointOperationLogOrAbortEnabled
; OBJ-DIS-NEXT:      a3: 40           	# -isInterruptHandler, +isFuncNamePresent, -isAllocaUsed
; OBJ-DIS-NEXT:                       	  OnConditionDirective = 0, -isCRSaved, -isLRSaved
; OBJ-DIS-NEXT:      a4: 05           	# -isBackChainStored, -isFixup, NumOfFPRsSaved = 5
; OBJ-DIS-NEXT:      a5: 04           	# -hasExtensionTable, -hasVectorInfo, NumOfGPRsSaved = 4
; OBJ-DIS-NEXT:      a6: 00           	# NumberOfFixedParms = 0
; OBJ-DIS-NEXT:      a7: 01           	# NumberOfFPParms = 0, +hasParmsOnStack
; OBJ-DIS-NEXT:      a8: 00 00 00 9c  	# TraceBackTableOffset = 156
; OBJ-DIS-NEXT:      ac: 00 03        	# FunctionNameLen = 3
; OBJ-DIS-NEXT:      ae: 62 61 72     	# FunctionName = bar
; NO-FUNC-SEC-NEXT:      b1: 60 00 00     	# Padding
; NO-FUNC-SEC-NEXT:      b4: 00 60 00 00  
; NO-FUNC-SEC-NEXT:      b8: 00 60 00 00  
; NO-FUNC-SEC-NEXT:      bc: 00 00 00 00  
; FUNC-SEC-NEXT:      b1: 00 00 00     	# Padding
; FUNC-SEC-NEXT:        ...

; OBJ-DIS:     11c: 00 00 00 00  	# Traceback table start
; OBJ-DIS-NEXT:     120: 00           	# Version = 0
; OBJ-DIS-NEXT:     121: 09           	# Language = CPlusPlus
; OBJ-DIS-NEXT:     122: 20           	# -isGlobalLinkage, -isOutOfLineEpilogOrPrologue
; OBJ-DIS-NEXT:                       	  +hasTraceBackTableOffset, -isInternalProcedure
; OBJ-DIS-NEXT:                       	  -hasControlledStorage, -isTOCless
; OBJ-DIS-NEXT:                       	  -isFloatingPointPresent
; OBJ-DIS-NEXT:                       	  -isFloatingPointOperationLogOrAbortEnabled
; OBJ-DIS-NEXT:     123: 40           	# -isInterruptHandler, +isFuncNamePresent, -isAllocaUsed
; OBJ-DIS-NEXT:                       	  OnConditionDirective = 0, -isCRSaved, -isLRSaved
; OBJ-DIS-NEXT:     124: 00           	# -isBackChainStored, -isFixup, NumOfFPRsSaved = 0
; OBJ-DIS-NEXT:     125: c0           	# +hasExtensionTable, +hasVectorInfo, NumOfGPRsSaved = 0
; OBJ-DIS-NEXT:     126: 00           	# NumberOfFixedParms = 0
; OBJ-DIS-NEXT:     127: 01           	# NumberOfFPParms = 0, +hasParmsOnStack
; OBJ-DIS-NEXT:     128: 00 00 00 5c  	# TraceBackTableOffset = 92
; OBJ-DIS-NEXT:     12c: 00 04        	# FunctionNameLen = 4
; OBJ-DIS-NEXT:     12e: 66 6f 6f 76  	# FunctionName = foov
; OBJ-DIS-NEXT:     132: 12           	# NumberOfVRSaved = 4, +isVRSavedOnStack, -hasVarArgs
; OBJ-DIS-NEXT:     133: 01           	# NumberOfVectorParms = 0, +hasVMXInstruction
; OBJ-DIS-NEXT:     134: 00 00 00 00  	# VectorParmsInfoString = 
; OBJ-DIS-NEXT:     138: 00 00        	# Padding
; OBJ-DIS-NEXT:     13a: 08           	# ExtensionTable = TB_EH_INFO
; OBJ-DIS-NEXT:     13b: 00           	# Alignment padding for eh info displacement
; OBJ-DIS-NEXT:     13c: 00 00 00 08  	# EH info displacement
