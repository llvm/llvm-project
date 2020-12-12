; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr4 \
; RUN:     -mattr=-altivec -xcoff-traceback-table=true < %s | \
; RUN:   FileCheck --check-prefixes=CHECK-ASM,COMMON %s

; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -function-sections \
; RUN:     -mcpu=pwr4 -mattr=-altivec < %s | \
; RUN:   FileCheck --check-prefixes=CHECK-FUNC,COMMON %s


%struct.S = type { i32, i32 }
%struct.D = type { float, double }
%struct.SD = type { %struct.S*, %struct.D }

@__const.main.s = private unnamed_addr constant %struct.S { i32 10, i32 20 }, align 4
@__const.main.d = private unnamed_addr constant %struct.D { float 1.000000e+01, double 2.000000e+01 }, align 8

define double @_Z10add_structifd1SP2SD1Di(i32 %value, float %fvalue, double %dvalue, %struct.S* byval(%struct.S) align 4 %s, %struct.SD* %dp, %struct.D* byval(%struct.D) align 4 %0, i32 %v2) #0 {
entry:
  %d = alloca %struct.D, align 8
  %value.addr = alloca i32, align 4
  %fvalue.addr = alloca float, align 4
  %dvalue.addr = alloca double, align 8
  %dp.addr = alloca %struct.SD*, align 4
  %v2.addr = alloca i32, align 4
  %1 = bitcast %struct.D* %d to i8*
  %2 = bitcast %struct.D* %0 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 8 %1, i8* align 4 %2, i32 16, i1 false)
  store i32 %value, i32* %value.addr, align 4
  store float %fvalue, float* %fvalue.addr, align 4
  store double %dvalue, double* %dvalue.addr, align 8
  store %struct.SD* %dp, %struct.SD** %dp.addr, align 4
  store i32 %v2, i32* %v2.addr, align 4
  %3 = load double, double* %dvalue.addr, align 8
  %4 = load float, float* %fvalue.addr, align 4
  %conv = fpext float %4 to double
  %add = fadd double %3, %conv
  %5 = load i32, i32* %value.addr, align 4
  %conv1 = sitofp i32 %5 to double
  %add2 = fadd double %add, %conv1
  %i1 = getelementptr inbounds %struct.S, %struct.S* %s, i32 0, i32 0
  %6 = load i32, i32* %i1, align 4
  %conv3 = sitofp i32 %6 to double
  %add4 = fadd double %add2, %conv3
  %7 = load %struct.SD*, %struct.SD** %dp.addr, align 4
  %d5 = getelementptr inbounds %struct.SD, %struct.SD* %7, i32 0, i32 1
  %d1 = getelementptr inbounds %struct.D, %struct.D* %d5, i32 0, i32 1
  %8 = load double, double* %d1, align 8
  %add6 = fadd double %add4, %8
  %f1 = getelementptr inbounds %struct.D, %struct.D* %d, i32 0, i32 0
  %9 = load float, float* %f1, align 8
  %conv7 = fpext float %9 to double
  %add8 = fadd double %add6, %conv7
  %10 = load i32, i32* %v2.addr, align 4
  %conv9 = sitofp i32 %10 to double
  %add10 = fadd double %add8, %conv9
  ret double %add10
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i32, i1 immarg) #1

define i32 @main() {
entry:
  %retval = alloca i32, align 4
  %s = alloca %struct.S, align 4
  %d = alloca %struct.D, align 8
  %sd = alloca %struct.SD, align 8
  %agg.tmp = alloca %struct.S, align 4
  %agg.tmp4 = alloca %struct.D, align 8
  store i32 0, i32* %retval, align 4
  %0 = bitcast %struct.S* %s to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 4 %0, i8* align 4 bitcast (%struct.S* @__const.main.s to i8*), i32 8, i1 false)
  %1 = bitcast %struct.D* %d to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 8 %1, i8* align 8 bitcast (%struct.D* @__const.main.d to i8*), i32 16, i1 false)
  %sp = getelementptr inbounds %struct.SD, %struct.SD* %sd, i32 0, i32 0
  store %struct.S* %s, %struct.S** %sp, align 8
  %d1 = getelementptr inbounds %struct.SD, %struct.SD* %sd, i32 0, i32 1
  %f1 = getelementptr inbounds %struct.D, %struct.D* %d1, i32 0, i32 0
  store float 1.000000e+02, float* %f1, align 8
  %d2 = getelementptr inbounds %struct.SD, %struct.SD* %sd, i32 0, i32 1
  %d13 = getelementptr inbounds %struct.D, %struct.D* %d2, i32 0, i32 1
  store double 2.000000e+02, double* %d13, align 8
  %2 = bitcast %struct.S* %agg.tmp to i8*
  %3 = bitcast %struct.S* %s to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 4 %2, i8* align 4 %3, i32 8, i1 false)
  %4 = bitcast %struct.D* %agg.tmp4 to i8*
  %5 = bitcast %struct.D* %d to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 8 %4, i8* align 8 %5, i32 16, i1 false)
  %call = call double @_Z10add_structifd1SP2SD1Di(i32 1, float 2.000000e+00, double 3.000000e+00, %struct.S* byval(%struct.S) align 4 %agg.tmp, %struct.SD* %sd, %struct.D* byval(%struct.D) align 4 %agg.tmp4, i32 7)
  %add = fadd double %call, 1.000000e+00
  %conv = fptosi double %add to i32
  ret i32 %conv
}

define double @_Z7add_bari1SfdP2SD1Di(i32 %value, %struct.S* byval(%struct.S) align 4 %s, float %fvalue, double %dvalue, %struct.SD* %dp, %struct.D* byval(%struct.D) align 4 %0, i32 %v2) #0 {
entry:
  %d = alloca %struct.D, align 8
  %value.addr = alloca i32, align 4
  %fvalue.addr = alloca float, align 4
  %dvalue.addr = alloca double, align 8
  %dp.addr = alloca %struct.SD*, align 4
  %v2.addr = alloca i32, align 4
  %1 = bitcast %struct.D* %d to i8*
  %2 = bitcast %struct.D* %0 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 8 %1, i8* align 4 %2, i32 16, i1 false)
  store i32 %value, i32* %value.addr, align 4
  store float %fvalue, float* %fvalue.addr, align 4
  store double %dvalue, double* %dvalue.addr, align 8
  store %struct.SD* %dp, %struct.SD** %dp.addr, align 4
  store i32 %v2, i32* %v2.addr, align 4
  %3 = load double, double* %dvalue.addr, align 8
  %4 = load float, float* %fvalue.addr, align 4
  %conv = fpext float %4 to double
  %add = fadd double %3, %conv
  %5 = load i32, i32* %value.addr, align 4
  %conv1 = sitofp i32 %5 to double
  %add2 = fadd double %add, %conv1
  %i1 = getelementptr inbounds %struct.S, %struct.S* %s, i32 0, i32 0
  %6 = load i32, i32* %i1, align 4
  %conv3 = sitofp i32 %6 to double
  %add4 = fadd double %add2, %conv3
  %7 = load %struct.SD*, %struct.SD** %dp.addr, align 4
  %d5 = getelementptr inbounds %struct.SD, %struct.SD* %7, i32 0, i32 1
  %d1 = getelementptr inbounds %struct.D, %struct.D* %d5, i32 0, i32 1
  %8 = load double, double* %d1, align 8
  %add6 = fadd double %add4, %8
  %f1 = getelementptr inbounds %struct.D, %struct.D* %d, i32 0, i32 0
  %9 = load float, float* %f1, align 8
  %conv7 = fpext float %9 to double
  %add8 = fadd double %add6, %conv7
  %10 = load i32, i32* %v2.addr, align 4
  %conv9 = sitofp i32 %10 to double
  %add10 = fadd double %add8, %conv9
  ret double %add10
}

; CHECK-ASM-LABEL:  ._Z10add_structifd1SP2SD1Di:{{[[:space:]] *}}# %bb.0:
; CHECK-FUNC-LABEL: csect ._Z10add_structifd1SP2SD1Di[PR],2{{[[:space:]] *}}# %bb.0:
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
; COMMON-NEXT:  .byte   0x00                            # -HasVectorInfo, -HasExtensionTable, NumOfGPRsSaved = 0
; COMMON-NEXT:  .byte   0x05                            # NumberOfFixedParms = 5
; COMMON-NEXT:  .byte   0x05                            # NumberOfFPParms = 2, +HasParmsOnStack
; COMMON-NEXT:  .vbyte  4, 0x58000000                   # Parameter type = i, f, d, i, i, i, i
; CHECK-ASM-NEXT:   .vbyte  4, L.._Z10add_structifd1SP2SD1Di0-._Z10add_structifd1SP2SD1Di # Function size
; CHECK-FUNC-NEXT:   .vbyte  4, L.._Z10add_structifd1SP2SD1Di0-._Z10add_structifd1SP2SD1Di[PR] # Function size
; COMMON-NEXT:  .vbyte  2, 0x001a                       # Function name len = 26
; COMMON-NEXT:  .byte   '_,'Z,'1,'0,'a,'d,'d,'_,'s,'t,'r,'u,'c,'t,'i,'f,'d,'1,'S,'P,'2,'S,'D,'1,'D,'i # Function Name
; COMMON-NEXT:                                        # -- End function


; CHECK-ASM-LABEL:     .main:{{[[:space:]] *}}# %bb.0:
; CHECK-FUNC-LABEL:   .csect .main[PR],2{{[[:space:]] *}}# %bb.0
; COMMON-NEXT:   mflr 0
; COMMON-NEXT:   stw 0, 8(1)
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
; COMMON-NEXT:  .byte   0x00                            # -HasVectorInfo, -HasExtensionTable, NumOfGPRsSaved = 0
; COMMON-NEXT:  .byte   0x00                            # NumberOfFixedParms = 0
; COMMON-NEXT:  .byte   0x01                            # NumberOfFPParms = 0, +HasParmsOnStack
; CHECK-ASM-NEXT:   .vbyte  4, L..main0-.main               # Function size
; CHECK-FUNC-NEXT:   .vbyte  4, L..main0-.main[PR]               # Function size
; COMMON-NEXT:  .vbyte  2, 0x0004                       # Function name len = 4
; COMMON-NEXT:  .byte   'm,'a,'i,'n                     # Function Name
; COMMON-NEXT:                                        # -- End function


; CHECK-ASM-LABEL:    ._Z7add_bari1SfdP2SD1Di:{{[[:space:]] *}}# %bb.0:
; CHECK-FUNC-LABEL:   .csect ._Z7add_bari1SfdP2SD1Di[PR],2{{[[:space:]] *}}# %bb.0:
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
; COMMON-NEXT:  .byte   0x00                            # -HasVectorInfo, -HasExtensionTable, NumOfGPRsSaved = 0
; COMMON-NEXT:  .byte   0x05                            # NumberOfFixedParms = 5
; COMMON-NEXT:  .byte   0x05                            # NumberOfFPParms = 2, +HasParmsOnStack
; COMMON-NEXT:  .vbyte  4, 0x16000000                   # Parameter type = i, i, i, f, d, i, i
; CHECK-ASM-NEXT:  .vbyte  4, L.._Z7add_bari1SfdP2SD1Di0-._Z7add_bari1SfdP2SD1Di # Function size
; CHECK-FUNC-NEXT:  .vbyte  4, L.._Z7add_bari1SfdP2SD1Di0-._Z7add_bari1SfdP2SD1Di[PR] # Function size
; COMMON-NEXT:  .vbyte  2, 0x0016                       # Function name len = 22
; COMMON-NEXT:  .byte   '_,'Z,'7,'a,'d,'d,'_,'b,'a,'r,'i,'1,'S,'f,'d,'P,'2,'S,'D,'1,'D,'i # Function Name
; COMMON-NEXT:                                        # -- End function
