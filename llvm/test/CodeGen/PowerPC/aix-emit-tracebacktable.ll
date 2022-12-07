; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:     -mattr=-altivec -xcoff-traceback-table=true < %s | \
; RUN:   FileCheck --check-prefixes=CHECK-ASM,COMMON %s

; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -function-sections \
; RUN:     -mcpu=pwr4 -mattr=-altivec < %s | \
; RUN:   FileCheck --check-prefixes=CHECK-FUNC,COMMON %s


%struct.S = type { i32, i32 }
%struct.D = type { float, double }
%struct.SD = type { ptr, %struct.D }

@__const.main.s = private unnamed_addr constant %struct.S { i32 10, i32 20 }, align 4
@__const.main.d = private unnamed_addr constant %struct.D { float 1.000000e+01, double 2.000000e+01 }, align 8

define double @_Z10add_structifd1SP2SD1Di(i32 %value, float %fvalue, double %dvalue, ptr byval(%struct.S) align 4 %s, ptr %dp, ptr byval(%struct.D) align 4 %0, i32 %v2) #0 {
entry:
  %d = alloca %struct.D, align 8
  %value.addr = alloca i32, align 4
  %fvalue.addr = alloca float, align 4
  %dvalue.addr = alloca double, align 8
  %dp.addr = alloca ptr, align 4
  %v2.addr = alloca i32, align 4
  call void @llvm.memcpy.p0.p0.i32(ptr align 8 %d, ptr align 4 %0, i32 16, i1 false)
  store i32 %value, ptr %value.addr, align 4
  store float %fvalue, ptr %fvalue.addr, align 4
  store double %dvalue, ptr %dvalue.addr, align 8
  store ptr %dp, ptr %dp.addr, align 4
  store i32 %v2, ptr %v2.addr, align 4
  %1 = load double, ptr %dvalue.addr, align 8
  %2 = load float, ptr %fvalue.addr, align 4
  %conv = fpext float %2 to double
  %add = fadd double %1, %conv
  %3 = load i32, ptr %value.addr, align 4
  %conv1 = sitofp i32 %3 to double
  %add2 = fadd double %add, %conv1
  %4 = load i32, ptr %s, align 4
  %conv3 = sitofp i32 %4 to double
  %add4 = fadd double %add2, %conv3
  %5 = load ptr, ptr %dp.addr, align 4
  %d5 = getelementptr inbounds %struct.SD, ptr %5, i32 0, i32 1
  %d1 = getelementptr inbounds %struct.D, ptr %d5, i32 0, i32 1
  %6 = load double, ptr %d1, align 8
  %add6 = fadd double %add4, %6
  %7 = load float, ptr %d, align 8
  %conv7 = fpext float %7 to double
  %add8 = fadd double %add6, %conv7
  %8 = load i32, ptr %v2.addr, align 4
  %conv9 = sitofp i32 %8 to double
  %add10 = fadd double %add8, %conv9
  ret double %add10
}

declare void @llvm.memcpy.p0.p0.i32(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i32, i1 immarg) #1

define i32 @main() {
entry:
  %retval = alloca i32, align 4
  %s = alloca %struct.S, align 4
  %d = alloca %struct.D, align 8
  %sd = alloca %struct.SD, align 8
  %agg.tmp = alloca %struct.S, align 4
  %agg.tmp4 = alloca %struct.D, align 8
  store i32 0, ptr %retval, align 4
  call void @llvm.memcpy.p0.p0.i32(ptr align 4 %s, ptr align 4 @__const.main.s, i32 8, i1 false)
  call void @llvm.memcpy.p0.p0.i32(ptr align 8 %d, ptr align 8 @__const.main.d, i32 16, i1 false)
  store ptr %s, ptr %sd, align 8
  %d1 = getelementptr inbounds %struct.SD, ptr %sd, i32 0, i32 1
  store float 1.000000e+02, ptr %d1, align 8
  %d2 = getelementptr inbounds %struct.SD, ptr %sd, i32 0, i32 1
  %d13 = getelementptr inbounds %struct.D, ptr %d2, i32 0, i32 1
  store double 2.000000e+02, ptr %d13, align 8
  call void @llvm.memcpy.p0.p0.i32(ptr align 4 %agg.tmp, ptr align 4 %s, i32 8, i1 false)
  call void @llvm.memcpy.p0.p0.i32(ptr align 8 %agg.tmp4, ptr align 8 %d, i32 16, i1 false)
  %call = call double @_Z10add_structifd1SP2SD1Di(i32 1, float 2.000000e+00, double 3.000000e+00, ptr byval(%struct.S) align 4 %agg.tmp, ptr %sd, ptr byval(%struct.D) align 4 %agg.tmp4, i32 7)
  %add = fadd double %call, 1.000000e+00
  %conv = fptosi double %add to i32
  ret i32 %conv
}

define double @_Z7add_bari1SfdP2SD1Di(i32 %value, ptr byval(%struct.S) align 4 %s, float %fvalue, double %dvalue, ptr %dp, ptr byval(%struct.D) align 4 %0, i32 %v2) #0 {
entry:
  %d = alloca %struct.D, align 8
  %value.addr = alloca i32, align 4
  %fvalue.addr = alloca float, align 4
  %dvalue.addr = alloca double, align 8
  %dp.addr = alloca ptr, align 4
  %v2.addr = alloca i32, align 4
  call void @llvm.memcpy.p0.p0.i32(ptr align 8 %d, ptr align 4 %0, i32 16, i1 false)
  store i32 %value, ptr %value.addr, align 4
  store float %fvalue, ptr %fvalue.addr, align 4
  store double %dvalue, ptr %dvalue.addr, align 8
  store ptr %dp, ptr %dp.addr, align 4
  store i32 %v2, ptr %v2.addr, align 4
  %1 = load double, ptr %dvalue.addr, align 8
  %2 = load float, ptr %fvalue.addr, align 4
  %conv = fpext float %2 to double
  %add = fadd double %1, %conv
  %3 = load i32, ptr %value.addr, align 4
  %conv1 = sitofp i32 %3 to double
  %add2 = fadd double %add, %conv1
  %4 = load i32, ptr %s, align 4
  %conv3 = sitofp i32 %4 to double
  %add4 = fadd double %add2, %conv3
  %5 = load ptr, ptr %dp.addr, align 4
  %d5 = getelementptr inbounds %struct.SD, ptr %5, i32 0, i32 1
  %d1 = getelementptr inbounds %struct.D, ptr %d5, i32 0, i32 1
  %6 = load double, ptr %d1, align 8
  %add6 = fadd double %add4, %6
  %7 = load float, ptr %d, align 8
  %conv7 = fpext float %7 to double
  %add8 = fadd double %add6, %conv7
  %8 = load i32, ptr %v2.addr, align 4
  %conv9 = sitofp i32 %8 to double
  %add10 = fadd double %add8, %conv9
  ret double %add10
}

define i32 @foo(i32 %i1, i32 %i2, i32 %i3, i32 %i4, i32 %i5, i32 %i6, i32 %i7, float %f1, float %f2, float %f3, float %f4, float %f5, float %f6, float %f7, float %f8, float %f9, float %f10, float %f11, float %f12, float %f13, float %f14, i32 %i8) {
entry:
  %i1.addr = alloca i32, align 4
  store i32 %i1, ptr %i1.addr, align 4
  ret i32 %i1
}

; CHECK-ASM-LABEL:  ._Z10add_structifd1SP2SD1Di:{{[[:space:]] *}}# %bb.0:
; CHECK-FUNC-LABEL: csect ._Z10add_structifd1SP2SD1Di[PR],5{{[[:space:]] *}}# %bb.0:
; COMMON-NEXT:   lwz 4, L..C0(2)
; COMMON-NEXT:   stfs 1, -24(1)
; COMMON-NEXT:   lfs 0, 0(4)
; COMMON-NEXT:   lwz 4, 56(1)
; COMMON:        fsub 0, 2, 0
; COMMON-NEXT:   stw 9, -36(1)
; COMMON-NEXT:   fadd 1, 1, 0
; COMMON-NEXT:   blr
; COMMON-NEXT: L.._Z10add_structifd1SP2SD1Di0:
; COMMON-NEXT:  .vbyte  4, 0x00000000                   # Traceback table begin
; COMMON-NEXT:  .byte   0x00                            # Version = 0
; COMMON-NEXT:  .byte   0x09                            # Language = CPlusPlus
; COMMON-NEXT:  .byte   0x22                            # -IsGlobaLinkage, -IsOutOfLineEpilogOrPrologue
; COMMON-NEXT:                                        # +HasTraceBackTableOffset, -IsInternalProcedure
; COMMON-NEXT:                                        # -HasControlledStorage, -IsTOCless
; COMMON-NEXT:                                        # +IsFloatingPointPresent
; COMMON-NEXT:                                        # -IsFloatingPointOperationLogOrAbortEnabled
; COMMON-NEXT:  .byte   0x40                            # -IsInterruptHandler, +IsFunctionNamePresent, -IsAllocaUsed
; COMMON-NEXT:                                        # OnConditionDirective = 0, -IsCRSaved, -IsLRSaved
; COMMON-NEXT:  .byte   0x80                            # +IsBackChainStored, -IsFixup, NumOfFPRsSaved = 0
; COMMON-NEXT:  .byte   0x00                            # -HasExtensionTable, -HasVectorInfo, NumOfGPRsSaved = 0
; COMMON-NEXT:  .byte   0x05                            # NumberOfFixedParms = 5
; COMMON-NEXT:  .byte   0x05                            # NumberOfFPParms = 2, +HasParmsOnStack
; COMMON-NEXT:  .vbyte  4, 0x58000000                   # Parameter type = i, f, d, i, i, i, i
; CHECK-ASM-NEXT:   .vbyte  4, L.._Z10add_structifd1SP2SD1Di0-._Z10add_structifd1SP2SD1Di # Function size
; CHECK-FUNC-NEXT:   .vbyte  4, L.._Z10add_structifd1SP2SD1Di0-._Z10add_structifd1SP2SD1Di[PR] # Function size
; COMMON-NEXT:  .vbyte  2, 0x001a                       # Function name len = 26
; COMMON-NEXT:  .byte   "_Z10add_structifd1SP2SD1Di"    # Function Name
; COMMON-NEXT:                                        # -- End function


; CHECK-ASM-LABEL:     .main:{{[[:space:]] *}}# %bb.0:
; CHECK-FUNC-LABEL:    .csect .main[PR],5{{[[:space:]] *}}# %bb.0
; COMMON-NEXT:   mflr 0
; COMMON:        stw 0, 168(1)
; COMMON:        mtlr 0
; COMMON-NEXT:   blr
; COMMON-NEXT: L..main0:
; COMMON-NEXT:  .vbyte  4, 0x00000000                   # Traceback table begin
; COMMON-NEXT:  .byte   0x00                            # Version = 0
; COMMON-NEXT:  .byte   0x09                            # Language = CPlusPlus
; COMMON-NEXT:  .byte   0x22                            # -IsGlobaLinkage, -IsOutOfLineEpilogOrPrologue
; COMMON-NEXT:                                        # +HasTraceBackTableOffset, -IsInternalProcedure
; COMMON-NEXT:                                        # -HasControlledStorage, -IsTOCless
; COMMON-NEXT:                                        # +IsFloatingPointPresent
; COMMON-NEXT:                                        # -IsFloatingPointOperationLogOrAbortEnabled
; COMMON-NEXT:  .byte   0x41                            # -IsInterruptHandler, +IsFunctionNamePresent, -IsAllocaUsed
; COMMON-NEXT:                                        # OnConditionDirective = 0, -IsCRSaved, +IsLRSaved
; COMMON-NEXT:  .byte   0x80                            # +IsBackChainStored, -IsFixup, NumOfFPRsSaved = 0
; COMMON-NEXT:  .byte   0x00                            # -HasExtensionTable, -HasVectorInfo, NumOfGPRsSaved = 0
; COMMON-NEXT:  .byte   0x00                            # NumberOfFixedParms = 0
; COMMON-NEXT:  .byte   0x01                            # NumberOfFPParms = 0, +HasParmsOnStack
; CHECK-ASM-NEXT:   .vbyte  4, L..main0-.main               # Function size
; CHECK-FUNC-NEXT:   .vbyte  4, L..main0-.main[PR]               # Function size
; COMMON-NEXT:  .vbyte  2, 0x0004                       # Function name len = 4
; COMMON-NEXT:  .byte   "main"                        # Function Name
; COMMON-NEXT:                                        # -- End function


; CHECK-ASM-LABEL:    ._Z7add_bari1SfdP2SD1Di:{{[[:space:]] *}}# %bb.0:
; CHECK-FUNC-LABEL:   .csect ._Z7add_bari1SfdP2SD1Di[PR],5{{[[:space:]] *}}# %bb.0:
; COMMON:       .vbyte  4, 0x00000000                   # Traceback table begin
; COMMON-NEXT:  .byte   0x00                            # Version = 0
; COMMON-NEXT:  .byte   0x09                            # Language = CPlusPlus
; COMMON-NEXT:  .byte   0x22                            # -IsGlobaLinkage, -IsOutOfLineEpilogOrPrologue
; COMMON-NEXT:                                        # +HasTraceBackTableOffset, -IsInternalProcedure
; COMMON-NEXT:                                        # -HasControlledStorage, -IsTOCless
; COMMON-NEXT:                                        # +IsFloatingPointPresent
; COMMON-NEXT:                                        # -IsFloatingPointOperationLogOrAbortEnabled
; COMMON-NEXT:  .byte   0x40                            # -IsInterruptHandler, +IsFunctionNamePresent, -IsAllocaUsed
; COMMON-NEXT:                                        # OnConditionDirective = 0, -IsCRSaved, -IsLRSaved
; COMMON-NEXT:  .byte   0x80                            # +IsBackChainStored, -IsFixup, NumOfFPRsSaved = 0
; COMMON-NEXT:  .byte   0x00                            # -HasExtensionTable, -HasVectorInfo, NumOfGPRsSaved = 0
; COMMON-NEXT:  .byte   0x05                            # NumberOfFixedParms = 5
; COMMON-NEXT:  .byte   0x05                            # NumberOfFPParms = 2, +HasParmsOnStack
; COMMON-NEXT:  .vbyte  4, 0x16000000                   # Parameter type = i, i, i, f, d, i, i
; CHECK-ASM-NEXT:  .vbyte  4, L.._Z7add_bari1SfdP2SD1Di0-._Z7add_bari1SfdP2SD1Di # Function size
; CHECK-FUNC-NEXT:  .vbyte  4, L.._Z7add_bari1SfdP2SD1Di0-._Z7add_bari1SfdP2SD1Di[PR] # Function size
; COMMON-NEXT:  .vbyte  2, 0x0016                       # Function name len = 22
; COMMON-NEXT:  .byte   "_Z7add_bari1SfdP2SD1Di"        # Function Name
; COMMON-NEXT:                                        # -- End function


; CHECK-ASM-LABEL:    .foo:{{[[:space:]] *}}# %bb.0:
; CHECK-FUNC-LABEL:   .csect .foo[PR],5{{[[:space:]] *}}# %bb.0:
; COMMON:       stw 3, -4(1)
; COMMON-NEXT:  blr
; COMMON-NEXT:L..foo0:
; COMMON-NEXT:  .vbyte  4, 0x00000000                   # Traceback table begin
; COMMON-NEXT:  .byte   0x00                            # Version = 0
; COMMON-NEXT:  .byte   0x09                            # Language = CPlusPlus
; COMMON-NEXT:  .byte   0x20                            # -IsGlobaLinkage, -IsOutOfLineEpilogOrPrologue
; COMMON-NEXT:                                        # +HasTraceBackTableOffset, -IsInternalProcedure
; COMMON-NEXT:                                        # -HasControlledStorage, -IsTOCless
; COMMON-NEXT:                                        # -IsFloatingPointPresent
; COMMON-NEXT:                                        # -IsFloatingPointOperationLogOrAbortEnabled
; COMMON-NEXT:  .byte   0x40                            # -IsInterruptHandler, +IsFunctionNamePresent, -IsAllocaUsed
; COMMON-NEXT:                                        # OnConditionDirective = 0, -IsCRSaved, -IsLRSaved
; COMMON-NEXT:  .byte   0x80                            # +IsBackChainStored, -IsFixup, NumOfFPRsSaved = 0
; COMMON-NEXT:  .byte   0x00                            # -HasExtensionTable, -HasVectorInfo, NumOfGPRsSaved = 0
; COMMON-NEXT:  .byte   0x07                            # NumberOfFixedParms = 7
; COMMON-NEXT:  .byte   0x1b                            # NumberOfFPParms = 13, +HasParmsOnStack
; COMMON-NEXT:  .vbyte  4, 0x01555554                   # Parameter type = i, i, i, i, i, i, i, f, f, f, f, f, f, f, f, f, f, f, f, ...
; CHECK-ASM-NEXT:  .vbyte  4, L..foo0-.foo                 # Function size
; CHECK-FUNC-NEXT: .vbyte  4, L..foo0-.foo[PR]                 # Function size
; COMMON-NEXT:  .vbyte  2, 0x0003                       # Function name len = 3
; COMMON-NEXT:  .byte   "foo"                           # Function Name
; COMMON-NEXT:                                        # -- End function
