; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr7 \
; RUN:     -mattr=+altivec  -vec-extabi -xcoff-traceback-table=true < %s | \
; RUN:   FileCheck --check-prefixes=CHECK-ASM,COMMON %s

; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -function-sections \
; RUN:     -mcpu=pwr7 -mattr=+altivec -vec-extabi < %s | \
; RUN:   FileCheck --check-prefixes=CHECK-FUNC,COMMON %s

;; #include <altivec.h>
;; vector float f(vector int vi1, int i1, int i2, float f1, vector float vf,double d1, vector char vc1) {
;;   return vec_abs(vf);
;; }
;; vector float fin(int x) {
;;  vector float vf ={1.0,1.0,1.0,1.0};
;;  if (x) return vf;
;;  return vec_abs(vf);
;; }

define dso_local <4 x float> @f(<4 x i32> %vi1, i32 signext %i1, i32 signext %i2, float %f1, <4 x float> %vf, double %d1, <16 x i8> %vc1) #0 {
entry:
  %__a.addr.i = alloca <4 x float>, align 16
  %vi1.addr = alloca <4 x i32>, align 16
  %i1.addr = alloca i32, align 4
  %i2.addr = alloca i32, align 4
  %f1.addr = alloca float, align 4
  %vf.addr = alloca <4 x float>, align 16
  %d1.addr = alloca double, align 8
  %vc1.addr = alloca <16 x i8>, align 16
  store <4 x i32> %vi1, ptr %vi1.addr, align 16
  store i32 %i1, ptr %i1.addr, align 4
  store i32 %i2, ptr %i2.addr, align 4
  store float %f1, ptr %f1.addr, align 4
  store <4 x float> %vf, ptr %vf.addr, align 16
  store double %d1, ptr %d1.addr, align 8
  store <16 x i8> %vc1, ptr %vc1.addr, align 16
  %0 = load <4 x float>, ptr %vf.addr, align 16
  store <4 x float> %0, ptr %__a.addr.i, align 16
  %1 = load <4 x float>, ptr %__a.addr.i, align 16
  %2 = load <4 x float>, ptr %__a.addr.i, align 16
  %3 = call <4 x float> @llvm.fabs.v4f32(<4 x float> %2) #2
  ret <4 x float> %3
}

define <4 x float> @fin(i32 %x) #0 {
entry:
  %__a.addr.i = alloca <4 x float>, align 16
  %__res.i = alloca <4 x i32>, align 16
  %retval = alloca <4 x float>, align 16
  %x.addr = alloca i32, align 4
  %vf = alloca <4 x float>, align 16
  store i32 %x, ptr %x.addr, align 4
  store <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, ptr %vf, align 16
  %0 = load i32, ptr %x.addr, align 4
  %tobool = icmp ne i32 %0, 0
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %1 = load <4 x float>, ptr %vf, align 16
  store <4 x float> %1, ptr %retval, align 16
  br label %return

if.end:                                           ; preds = %entry
  %2 = load <4 x float>, ptr %vf, align 16
  store <4 x float> %2, ptr %__a.addr.i, align 16
  %3 = load <4 x float>, ptr %__a.addr.i, align 16
  %4 = bitcast <4 x float> %3 to <4 x i32>
  %and.i = and <4 x i32> %4, <i32 2147483647, i32 2147483647, i32 2147483647, i32 2147483647>
  store <4 x i32> %and.i, ptr %__res.i, align 16
  %5 = load <4 x i32>, ptr %__res.i, align 16
  %6 = bitcast <4 x i32> %5 to <4 x float>
  store <4 x float> %6, ptr %retval, align 16
  br label %return

return:                                           ; preds = %if.end, %if.then
  %7 = load <4 x float>, ptr %retval, align 16
  ret <4 x float> %7
}

declare <4 x float> @llvm.fabs.v4f32(<4 x float>) #1

; COMMON:      L..f0:
; COMMON-NEXT:  .vbyte  4, 0x00000000                   # Traceback table begin
; COMMON-NEXT:  .byte   0x00                            # Version = 0
; COMMON-NEXT:  .byte   0x09                            # Language = CPlusPlus
; COMMON-NEXT:  .byte   0x22                            # -IsGlobaLinkage, -IsOutOfLineEpilogOrPrologue
; COMMON-NEXT:                                         # +HasTraceBackTableOffset, -IsInternalProcedure
; COMMON-NEXT:                                         # -HasControlledStorage, -IsTOCless
; COMMON-NEXT:                                         # +IsFloatingPointPresent
; COMMON-NEXT:                                         # -IsFloatingPointOperationLogOrAbortEnabled
; COMMON-NEXT:  .byte   0x40                            # -IsInterruptHandler, +IsFunctionNamePresent, -IsAllocaUsed
; COMMON-NEXT:                                         # OnConditionDirective = 0, -IsCRSaved, -IsLRSaved
; COMMON-NEXT:  .byte   0x80                            # +IsBackChainStored, -IsFixup, NumOfFPRsSaved = 0
; COMMON-NEXT:  .byte   0x40                            # -HasExtensionTable, +HasVectorInfo, NumOfGPRsSaved = 0
; COMMON-NEXT:  .byte   0x02                            # NumberOfFixedParms = 2
; COMMON-NEXT:  .byte   0x05                            # NumberOfFPParms = 2, +HasParmsOnStack
; COMMON-NEXT:  .vbyte  4, 0x42740000                   # Parameter type = v, i, i, f, v, d, v
; CHECK-ASM-NEXT:  .vbyte  4, L..f0-.f                     # Function size
; CHECK-FUNC-NEXT: .vbyte  4, L..f0-.f[PR]                 # Function size
; COMMON-NEXT:  .vbyte  2, 0x0001                       # Function name len = 1
; COMMON-NEXT:  .byte   102                             # Function Name
; COMMON-NEXT:  .byte   0x00                            # NumOfVRsSaved = 0, -IsVRSavedOnStack, -HasVarArgs
; COMMON-NEXT:  .byte   0x07                            # NumOfVectorParams = 3, +HasVMXInstruction
; COMMON-NEXT:  .vbyte  4, 0xb0000000                   # Vector Parameter type = vi, vf, vc
; COMMON-NEXT:  .vbyte  2, 0x0000                       # Padding 

; COMMON:     L..fin0:
; COMMON-NEXT:  .vbyte  4, 0x00000000                   # Traceback table begin
; COMMON-NEXT:  .byte   0x00                            # Version = 0
; COMMON-NEXT:  .byte   0x09                            # Language = CPlusPlus
; COMMON-NEXT:  .byte   0x22                            # -IsGlobaLinkage, -IsOutOfLineEpilogOrPrologue
; COMMON-NEXT:                                         # +HasTraceBackTableOffset, -IsInternalProcedure
; COMMON-NEXT:                                         # -HasControlledStorage, -IsTOCless
; COMMON-NEXT:                                         # +IsFloatingPointPresent
; COMMON-NEXT:                                         # -IsFloatingPointOperationLogOrAbortEnabled
; COMMON-NEXT:  .byte   0x40                            # -IsInterruptHandler, +IsFunctionNamePresent, -IsAllocaUsed
; COMMON-NEXT:                                         # OnConditionDirective = 0, -IsCRSaved, -IsLRSaved
; COMMON-NEXT:  .byte   0x80                            # +IsBackChainStored, -IsFixup, NumOfFPRsSaved = 0
; COMMON-NEXT:  .byte   0x40                            # -HasExtensionTable, +HasVectorInfo, NumOfGPRsSaved = 0
; COMMON-NEXT:  .byte   0x01                            # NumberOfFixedParms = 1
; COMMON-NEXT:  .byte   0x01                            # NumberOfFPParms = 0, +HasParmsOnStack
; COMMON-NEXT:  .vbyte  4, 0x00000000                   # Parameter type = i
; CHECK-ASM-NEXT:       .vbyte  4, L..fin0-.fin                 # Function size
; CHECK-FUNC-NEXT:      .vbyte  4, L..fin0-.fin[PR]             # Function size
; COMMON-NEXT:  .vbyte  2, 0x0003                       # Function name len = 3
; COMMON-NEXT:  .byte   "fin"                           # Function Name 
; COMMON-NEXT:  .byte   0x00                            # NumOfVRsSaved = 0, -IsVRSavedOnStack, -HasVarArgs
; COMMON-NEXT:  .byte   0x01                            # NumOfVectorParams = 0, +HasVMXInstruction
; COMMON-NEXT:  .vbyte  4, 0x00000000                   # Vector Parameter type =
; COMMON-NEXT:  .vbyte  2, 0x0000                       # Padding
