; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:     -mattr=-altivec -simplifycfg-require-and-preserve-domtree=1 < %s | \
; RUN:   FileCheck --check-prefixes=ASM,ASMNFS,ASM32 %s

; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:     -mattr=-altivec -simplifycfg-require-and-preserve-domtree=1 < %s | \
; RUN:   FileCheck --check-prefixes=ASM,ASMNFS,ASM64 %s

; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:     -mattr=-altivec -simplifycfg-require-and-preserve-domtree=1 \
; RUN:     -function-sections < %s | \
; RUN:   FileCheck --check-prefixes=ASM,ASMFS,ASM32 %s

; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:     -mattr=-altivec -simplifycfg-require-and-preserve-domtree=1 \
; RUN:     -function-sections < %s | \
; RUN:   FileCheck --check-prefixes=ASM,ASMFS,ASM64 %s

@_ZTIi = external constant i8*

define void @_Z9throwFuncv() {
entry:
  %exception = call i8* @__cxa_allocate_exception(i32 4) #2
  %0 = bitcast i8* %exception to i32*
  store i32 1, i32* %0, align 16
  call void @__cxa_throw(i8* %exception, i8* bitcast (i8** @_ZTIi to i8*), i8* null) #3
  unreachable
}

; ASMNFS: ._Z9throwFuncv:
; ASMFS:    .csect ._Z9throwFuncv[PR],2
; ASM:      bl .__cxa_allocate_exception[PR]
; ASM:      nop
; ASM32:    lwz 4, L..C0(2)
; ASM64:    ld 4, L..C0(2)
; ASM:      bl .__cxa_throw[PR]
; ASM:      nop

define i32 @_Z9catchFuncv() personality i8* bitcast (i32 (...)* @__xlcxx_personality_v1 to i8*) {
entry:
  %retval = alloca i32, align 4
  %exn.slot = alloca i8*, align 4
  %ehselector.slot = alloca i32, align 4
  %0 = alloca i32, align 4
  invoke void @_Z9throwFuncv()
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %entry
  br label %try.cont

lpad:                                             ; preds = %entry
  %1 = landingpad { i8*, i32 }
          catch i8* bitcast (i8** @_ZTIi to i8*)
  %2 = extractvalue { i8*, i32 } %1, 0
  store i8* %2, i8** %exn.slot, align 4
  %3 = extractvalue { i8*, i32 } %1, 1
  store i32 %3, i32* %ehselector.slot, align 4
  br label %catch.dispatch

catch.dispatch:                                   ; preds = %lpad
  %sel = load i32, i32* %ehselector.slot, align 4
  %4 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*)) #2
  %matches = icmp eq i32 %sel, %4
  br i1 %matches, label %catch, label %eh.resume

catch:                                            ; preds = %catch.dispatch
  %exn = load i8*, i8** %exn.slot, align 4
  %5 = call i8* @__cxa_begin_catch(i8* %exn) #2
  %6 = bitcast i8* %5 to i32*
  %7 = load i32, i32* %6, align 4
  store i32 %7, i32* %0, align 4
  store i32 2, i32* %retval, align 4
  call void @__cxa_end_catch() #2
  br label %return

try.cont:                                         ; preds = %invoke.cont
  store i32 1, i32* %retval, align 4
  br label %return

return:                                           ; preds = %try.cont, %catch
  %8 = load i32, i32* %retval, align 4
  ret i32 %8

eh.resume:                                        ; preds = %catch.dispatch
  %exn1 = load i8*, i8** %exn.slot, align 4
  %sel2 = load i32, i32* %ehselector.slot, align 4
  %lpad.val = insertvalue { i8*, i32 } undef, i8* %exn1, 0
  %lpad.val3 = insertvalue { i8*, i32 } %lpad.val, i32 %sel2, 1
  resume { i8*, i32 } %lpad.val3
}

; ASMNFS: ._Z9catchFuncv:
; ASMFS:        .csect ._Z9catchFuncv[PR],2
; ASM:  L..func_begin0:
; ASM:  # %bb.0:                                # %entry
; ASM:  	mflr 0
; ASM:  L..tmp0:
; ASM:  	bl ._Z9throwFuncv
; ASM:  	nop
; ASM:  L..tmp1:
; ASM:  # %bb.1:                                # %invoke.cont
; ASM:  	li 3, 1
; ASM:  L..BB1_2:                               # %return
; ASM:  	mtlr 0
; ASM:  	blr
; ASM:  L..BB1_3:                               # %lpad
; ASM:  L..tmp2:
; ASM:  	bl .__cxa_begin_catch[PR]
; ASM:  	nop
; ASM:  	bl .__cxa_end_catch[PR]
; ASM:  	nop
; ASM:  	b L..BB1_2

; ASM:  L.._Z9catchFuncv0:
; ASM:    .vbyte  4, 0x00000000                   # Traceback table begin
; ASM:    .byte   0x00                            # Version = 0
; ASM:    .byte   0x09                            # Language = CPlusPlus
; ASM:    .byte   0x20                            # -IsGlobaLinkage, -IsOutOfLineEpilogOrPrologue
; ASM:                                    # +HasTraceBackTableOffset, -IsInternalProcedure
; ASM:                                    # -HasControlledStorage, -IsTOCless
; ASM:                                    # -IsFloatingPointPresent
; ASM:                                    # -IsFloatingPointOperationLogOrAbortEnabled
; ASM:    .byte   0x41                            # -IsInterruptHandler, +IsFunctionNamePresent, -IsAllocaUsed
; ASM:                                    # OnConditionDirective = 0, -IsCRSaved, +IsLRSaved
; ASM:    .byte   0x80                            # +IsBackChainStored, -IsFixup, NumOfFPRsSaved = 0
; ASM:    .byte   0x80                            # +HasExtensionTable, -HasVectorInfo, NumOfGPRsSaved = 0
; ASM:    .byte   0x00                            # NumberOfFixedParms = 0
; ASM:    .byte   0x01                            # NumberOfFPParms = 0, +HasParmsOnStack
; ASMNFS: .vbyte  4, L.._Z9catchFuncv0-._Z9catchFuncv # Function size
; ASMFS:  .vbyte  4, L.._Z9catchFuncv0-._Z9catchFuncv[PR] # Function size
; ASM:    .vbyte  2, 0x000d                       # Function name len = 13
; ASM:    .byte   "_Z9catchFuncv"                 # Function Name
; ASM:    .byte   0x08                            # ExtensionTableFlag = TB_EH_INFO
; ASM:    .align  2
; ASM32:  .vbyte  4, L..C1-TOC[TC0]               # EHInfo Table
; ASM64:  .vbyte  8, L..C1-TOC[TC0]               # EHInfo Table
; ASM:  L..func_end0:

; ASMNFS:  	.csect .gcc_except_table[RO],2
; ASMFS:  	.csect .gcc_except_table._Z9catchFuncv[RO],2
; ASM:  	.align	2
; ASM:  GCC_except_table1:
; ASM:  L..exception0:
; ASM:  	.byte	255                             # @LPStart Encoding = omit
; ASM32:	.byte	187                             # @TType Encoding = indirect datarel sdata4
; ASM64:  .byte	188                             # @TType Encoding = indirect datarel sdata8
; ASM32: 	.byte 37
; ASM64:  .byte	41
; ASM:  	.byte	3                               # Call site Encoding = udata4
; ASM:  	.byte 26
; ASM:  	.vbyte	4, L..tmp0-L..func_begin0       # >> Call Site 1 <<
; ASM:  	.vbyte	4, L..tmp1-L..tmp0              #   Call between L..tmp0 and L..tmp1
; ASM:  	.vbyte	4, L..tmp2-L..func_begin0       #     jumps to L..tmp2
; ASM:  	.byte	1                               #   On action: 1
; ASM:  	.vbyte	4, L..tmp1-L..func_begin0       # >> Call Site 2 <<
; ASM:  	.vbyte	4, L..func_end0-L..tmp1         #   Call between L..tmp1 and L..func_end0
; ASM:  	.vbyte	4, 0                            #     has no landing pad
; ASM:  	.byte	0                               #   On action: cleanup
; ASM:  L..cst_end0:
; ASM:  	.byte	1                               # >> Action Record 1 <<
; ASM:                                          #   Catch TypeInfo 1
; ASM:  	.byte	0                               #   No further actions
; ASM:  	.align	2
; ASM:                                          # >> Catch TypeInfos <<
; ASM32:	.vbyte	4, L..C0-TOC[TC0]               # TypeInfo 1
; ASM64: 	.vbyte	8, L..C0-TOC[TC0]               # TypeInfo 1
; ASM:  L..ttbase0:
; ASM:  	.align	2

; ASMNFS:  	.csect .eh_info_table[RW],2
; ASMFS:  	.csect .eh_info_table._Z9catchFuncv[RW],2
; ASM:  __ehinfo.1:
; ASM:  	.vbyte	4, 0
; ASM32:  .align  2
; ASM32:  .vbyte	4, GCC_except_table1
; ASM32:  .vbyte	4, __xlcxx_personality_v1[DS]
; ASM64:  .align	3
; ASM64:  .vbyte	8, GCC_except_table1
; ASM64:  .vbyte	8, __xlcxx_personality_v1[DS]

; ASM:    .toc
; ASM:  L..C0:
; ASM:    .tc _ZTIi[TC],_ZTIi[UA]
; ASM:  L..C1:
; ASM:    .tc __ehinfo.1[TC],__ehinfo.1

declare i8* @__cxa_allocate_exception(i32)
declare void @__cxa_throw(i8*, i8*, i8*)
declare i32 @__xlcxx_personality_v1(...)
declare i32 @llvm.eh.typeid.for(i8*)
declare i8* @__cxa_begin_catch(i8*)
declare void @__cxa_end_catch()
