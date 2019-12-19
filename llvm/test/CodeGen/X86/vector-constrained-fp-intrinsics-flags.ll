; RUN: llc -O3 -mtriple=x86_64-pc-linux -stop-after=finalize-isel < %s | FileCheck %s

define <1 x float> @constrained_vector_fadd_v1f32() #0 {
; CHECK-LABEL: name: constrained_vector_fadd_v1f32
; CHECK: [[MOVSSrm_alt:%[0-9]+]]:fr32 = MOVSSrm_alt $rip, 1, $noreg, %const.0, $noreg :: (load 4 from constant-pool)
; CHECK: [[ADDSSrm:%[0-9]+]]:fr32 = ADDSSrm [[MOVSSrm_alt]], $rip, 1, $noreg, %const.1, $noreg, implicit $mxcsr :: (load 4 from constant-pool)
; CHECK: $xmm0 = COPY [[ADDSSrm]]
; CHECK: RET 0, $xmm0
entry:
  %add = call <1 x float> @llvm.experimental.constrained.fadd.v1f32(<1 x float> <float 0x7FF0000000000000>, <1 x float> <float 1.0>, metadata !"round.dynamic", metadata !"fpexcept.strict") #0
  ret <1 x float> %add
}

define <3 x float> @constrained_vector_fadd_v3f32() #0 {
; CHECK-LABEL: name: constrained_vector_fadd_v3f32
; CHECK: [[FsFLD0SS:%[0-9]+]]:fr32 = FsFLD0SS
; CHECK: [[MOVSSrm_alt:%[0-9]+]]:fr32 = MOVSSrm_alt $rip, 1, $noreg, %const.0, $noreg :: (load 4 from constant-pool)
; CHECK: [[ADDSSrr:%[0-9]+]]:fr32 = ADDSSrr [[MOVSSrm_alt]], killed [[FsFLD0SS]], implicit $mxcsr
; CHECK: [[ADDSSrm:%[0-9]+]]:fr32 = ADDSSrm [[MOVSSrm_alt]], $rip, 1, $noreg, %const.1, $noreg, implicit $mxcsr :: (load 4 from constant-pool)
; CHECK: [[ADDSSrm1:%[0-9]+]]:fr32 = ADDSSrm [[MOVSSrm_alt]], $rip, 1, $noreg, %const.2, $noreg, implicit $mxcsr :: (load 4 from constant-pool)
; CHECK: [[COPY:%[0-9]+]]:vr128 = COPY [[ADDSSrm1]]
; CHECK: [[COPY1:%[0-9]+]]:vr128 = COPY [[ADDSSrm]]
; CHECK: [[UNPCKLPSrr:%[0-9]+]]:vr128 = UNPCKLPSrr [[COPY1]], killed [[COPY]]
; CHECK: [[COPY2:%[0-9]+]]:vr128 = COPY [[ADDSSrr]]
; CHECK: [[UNPCKLPDrr:%[0-9]+]]:vr128 = UNPCKLPDrr [[UNPCKLPSrr]], killed [[COPY2]]
; CHECK: $xmm0 = COPY [[UNPCKLPDrr]]
; CHECK: RET 0, $xmm0
entry:
  %add = call <3 x float> @llvm.experimental.constrained.fadd.v3f32(
           <3 x float> <float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000,
                        float 0xFFFFFFFFE0000000>,
           <3 x float> <float 2.0, float 1.0, float 0.0>,
           metadata !"round.dynamic",
           metadata !"fpexcept.strict") #0
  ret <3 x float> %add
}

define <4 x double> @constrained_vector_fadd_v4f64() #0 {
; CHECK-LABEL: name: constrained_vector_fadd_v4f64
; CHECK: [[MOVAPDrm:%[0-9]+]]:vr128 = MOVAPDrm $rip, 1, $noreg, %const.0, $noreg :: (load 16 from constant-pool)
; CHECK: [[ADDPDrm:%[0-9]+]]:vr128 = ADDPDrm [[MOVAPDrm]], $rip, 1, $noreg, %const.1, $noreg, implicit $mxcsr :: (load 16 from constant-pool)
; CHECK: [[ADDPDrm1:%[0-9]+]]:vr128 = ADDPDrm [[MOVAPDrm]], $rip, 1, $noreg, %const.2, $noreg, implicit $mxcsr :: (load 16 from constant-pool)
; CHECK: $xmm0 = COPY [[ADDPDrm]]
; CHECK: $xmm1 = COPY [[ADDPDrm1]]
; CHECK: RET 0, $xmm0, $xmm1
entry:
  %add = call <4 x double> @llvm.experimental.constrained.fadd.v4f64(
           <4 x double> <double 0x7FEFFFFFFFFFFFFF, double 0x7FEFFFFFFFFFFFFF,
                         double 0x7FEFFFFFFFFFFFFF, double 0x7FEFFFFFFFFFFFFF>,
           <4 x double> <double 1.000000e+00, double 1.000000e-01,
                         double 2.000000e+00, double 2.000000e-01>,
           metadata !"round.dynamic",
           metadata !"fpexcept.strict") #0
  ret <4 x double> %add
}

declare <1 x float> @llvm.experimental.constrained.fadd.v1f32(<1 x float>, <1 x float>, metadata, metadata)
declare <3 x float> @llvm.experimental.constrained.fadd.v3f32(<3 x float>, <3 x float>, metadata, metadata)
declare <4 x double> @llvm.experimental.constrained.fadd.v4f64(<4 x double>, <4 x double>, metadata, metadata)
