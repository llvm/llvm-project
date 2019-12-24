; RUN: llc -O3 -mtriple=i686-pc-linux -mattr=sse2 -stop-after=finalize-isel < %s | FileCheck %s

define double @sifdb(i8 %x) #0 {
entry:
; CHECK-LABEL: name: sifdb
; CHECK: [[MOVSX32rm8_:%[0-9]+]]:gr32 = MOVSX32rm8 %fixed-stack.0, 1, $noreg, 0, $noreg :: (load 1 from %fixed-stack.0, align 16)
; CHECK: [[CVTSI2SDrr:%[0-9]+]]:fr64 = CVTSI2SDrr killed [[MOVSX32rm8_]]
; CHECK: MOVSDmr %stack.0, 1, $noreg, 0, $noreg, killed [[CVTSI2SDrr]] :: (store 8 into %stack.0, align 4)
; CHECK: [[LD_Fp64m80_:%[0-9]+]]:rfp80 = LD_Fp64m80 %stack.0, 1, $noreg, 0, $noreg, implicit-def dead $fpsw, implicit $fpcw :: (load 8 from %stack.0, align 4)
; CHECK: RET 0, killed [[LD_Fp64m80_]]
  %result = call double @llvm.experimental.constrained.sitofp.f64.i8(i8 %x, metadata !"round.dynamic", metadata !"fpexcept.strict") #0
  ret double %result
}

define double @sifdw(i16 %x) #0 {
entry:
; CHECK-LABEL: name: sifdw
; CHECK: [[MOVSX32rm16_:%[0-9]+]]:gr32 = MOVSX32rm16 %fixed-stack.0, 1, $noreg, 0, $noreg :: (load 2 from %fixed-stack.0, align 16)
; CHECK: [[CVTSI2SDrr:%[0-9]+]]:fr64 = CVTSI2SDrr killed [[MOVSX32rm16_]]
; CHECK: MOVSDmr %stack.0, 1, $noreg, 0, $noreg, killed [[CVTSI2SDrr]] :: (store 8 into %stack.0, align 4)
; CHECK: [[LD_Fp64m80_:%[0-9]+]]:rfp80 = LD_Fp64m80 %stack.0, 1, $noreg, 0, $noreg, implicit-def dead $fpsw, implicit $fpcw :: (load 8 from %stack.0, align 4)
; CHECK: RET 0, killed [[LD_Fp64m80_]]
  %result = call double @llvm.experimental.constrained.sitofp.f64.i16(i16 %x, metadata !"round.dynamic", metadata !"fpexcept.strict") #0
  ret double %result
}

define i64 @f20u64(double %x) #0 {
entry:
; CHECK-LABEL: name: f20u64
; CHECK: [[MOVSDrm_alt:%[0-9]+]]:fr64 = MOVSDrm_alt %fixed-stack.0, 1, $noreg, 0, $noreg :: (load 8 from %fixed-stack.0, align 16)
; CHECK: [[MOVSDrm_alt1:%[0-9]+]]:fr64 = MOVSDrm_alt $noreg, 1, $noreg, %const.0, $noreg :: (load 8 from constant-pool)
; CHECK: COMISDrr [[MOVSDrm_alt1]], [[MOVSDrm_alt]], implicit-def $eflags, implicit $mxcsr
; CHECK: [[FsFLD0SD:%[0-9]+]]:fr64 = FsFLD0SD
; CHECK: JCC_1
; CHECK: [[PHI:%[0-9]+]]:fr64 = PHI [[MOVSDrm_alt1]], {{.*}}, [[FsFLD0SD]], {{.*}}
; CHECK: [[SUBSDrr:%[0-9]+]]:fr64 = SUBSDrr [[MOVSDrm_alt]], killed [[PHI]], implicit $mxcsr
; CHECK: MOVSDmr %stack.0, 1, $noreg, 0, $noreg, killed [[SUBSDrr]] :: (store 8 into %stack.0)
; CHECK: [[SETCCr:%[0-9]+]]:gr8 = SETCCr 6, implicit $eflags
; CHECK: [[LD_Fp64m:%[0-9]+]]:rfp64 = LD_Fp64m %stack.0, 1, $noreg, 0, $noreg, implicit-def dead $fpsw, implicit $fpcw :: (load 8 from %stack.0)
; CHECK: FNSTCW16m %stack.1, 1, $noreg, 0, $noreg, implicit-def $fpsw, implicit $fpcw :: (store 2 into %stack.1)
; CHECK: [[MOVZX32rm16_:%[0-9]+]]:gr32 = MOVZX32rm16 %stack.1, 1, $noreg, 0, $noreg :: (load 2 from %stack.1)
; CHECK: [[OR32ri:%[0-9]+]]:gr32 = OR32ri killed [[MOVZX32rm16_]], 3072, implicit-def $eflags
; CHECK: [[COPY3:%[0-9]+]]:gr16 = COPY killed [[OR32ri]].sub_16bit
; CHECK: MOV16mr %stack.2, 1, $noreg, 0, $noreg, killed [[COPY3]] :: (store 2 into %stack.2)
; CHECK: FLDCW16m %stack.2, 1, $noreg, 0, $noreg, implicit-def $fpsw, implicit-def $fpcw :: (load 2 from %stack.2)
; CHECK: IST_Fp64m64 %stack.0, 1, $noreg, 0, $noreg, [[LD_Fp64m]], implicit-def $fpsw, implicit $fpcw
; CHECK: FLDCW16m %stack.1, 1, $noreg, 0, $noreg, implicit-def $fpsw, implicit-def $fpcw :: (load 2 from %stack.1)
; CHECK: [[MOVZX32rr8_:%[0-9]+]]:gr32 = MOVZX32rr8 killed [[SETCCr]]
; CHECK: [[SHL32ri:%[0-9]+]]:gr32 = SHL32ri [[MOVZX32rr8_]], 31, implicit-def dead $eflags
; CHECK: [[XOR32rm:%[0-9]+]]:gr32 = XOR32rm [[SHL32ri]], %stack.0, 1, $noreg, 4, $noreg, implicit-def dead $eflags :: (load 4 from %stack.0 + 4)
; CHECK: [[MOV32rm:%[0-9]+]]:gr32 = MOV32rm %stack.0, 1, $noreg, 0, $noreg :: (load 4 from %stack.0, align 8)
; CHECK: $eax = COPY [[MOV32rm]]
; CHECK: $edx = COPY [[XOR32rm]]
; CHECK: RET 0, $eax, $edx
  %result = call i64 @llvm.experimental.constrained.fptoui.i64.f64(double %x, metadata !"fpexcept.strict") #0
  ret i64 %result
}

define i8 @f20s8(double %x) #0 {
entry:
; CHECK-LABEL: name: f20s8
; CHECK: [[CVTTSD2SIrm:%[0-9]+]]:gr32 = CVTTSD2SIrm %fixed-stack.0, 1, $noreg, 0, $noreg, implicit $mxcsr :: (load 8 from %fixed-stack.0, align 16)
; CHECK: [[COPY:%[0-9]+]]:gr32_abcd = COPY [[CVTTSD2SIrm]]
; CHECK: [[COPY1:%[0-9]+]]:gr8 = COPY [[COPY]].sub_8bit
; CHECK: $al = COPY [[COPY1]]
; CHECK: RET 0, $al
  %result = call i8 @llvm.experimental.constrained.fptosi.i8.f64(double %x, metadata !"fpexcept.strict") #0
  ret i8 %result
}

define i16 @f20s16(double %x) #0 {
entry:
; CHECK-LABEL: name: f20s16
; CHECK: [[CVTTSD2SIrm:%[0-9]+]]:gr32 = CVTTSD2SIrm %fixed-stack.0, 1, $noreg, 0, $noreg, implicit $mxcsr :: (load 8 from %fixed-stack.0, align 16)
; CHECK: [[COPY:%[0-9]+]]:gr16 = COPY [[CVTTSD2SIrm]].sub_16bit
; CHECK: $ax = COPY [[COPY]]
; CHECK: RET 0, $ax
  %result = call i16 @llvm.experimental.constrained.fptosi.i16.f64(double %x, metadata !"fpexcept.strict") #0
  ret i16 %result
}

define i32 @f20u(double %x) #0 {
entry:
; CHECK-LABEL: name: f20u
; CHECK: [[MOVSDrm_alt:%[0-9]+]]:fr64 = MOVSDrm_alt %fixed-stack.0, 1, $noreg, 0, $noreg :: (load 8 from %fixed-stack.0, align 16)
; CHECK: [[MOVSDrm_alt1:%[0-9]+]]:fr64 = MOVSDrm_alt $noreg, 1, $noreg, %const.0, $noreg :: (load 8 from constant-pool)
; CHECK: COMISDrr [[MOVSDrm_alt1]], [[MOVSDrm_alt]], implicit-def $eflags, implicit $mxcsr
; CHECK: [[FsFLD0SD:%[0-9]+]]:fr64 = FsFLD0SD
; CHECK: JCC_1
; CHECK: [[PHI:%[0-9]+]]:fr64 = PHI [[MOVSDrm_alt1]], {{.*}}, [[FsFLD0SD]], {{.*}}
; CHECK: [[SETCCr:%[0-9]+]]:gr8 = SETCCr 6, implicit $eflags
; CHECK: [[MOVZX32rr8_:%[0-9]+]]:gr32 = MOVZX32rr8 killed [[SETCCr]]
; CHECK: [[SHL32ri:%[0-9]+]]:gr32 = SHL32ri [[MOVZX32rr8_]], 31, implicit-def dead $eflags
; CHECK: [[SUBSDrr:%[0-9]+]]:fr64 = SUBSDrr [[MOVSDrm_alt]], killed [[PHI]], implicit $mxcsr
; CHECK: [[CVTTSD2SIrr:%[0-9]+]]:gr32 = CVTTSD2SIrr killed [[SUBSDrr]], implicit $mxcsr
; CHECK: [[XOR32rr:%[0-9]+]]:gr32 = XOR32rr [[CVTTSD2SIrr]], killed [[SHL32ri]], implicit-def dead $eflags
; CHECK: $eax = COPY [[XOR32rr]]
; CHECK: RET 0, $eax
  %result = call i32 @llvm.experimental.constrained.fptoui.i32.f64(double %x, metadata !"fpexcept.strict") #0
  ret i32 %result
}

attributes #0 = { strictfp }

declare double @llvm.experimental.constrained.sitofp.f64.i8(i8, metadata, metadata)
declare double @llvm.experimental.constrained.sitofp.f64.i16(i16, metadata, metadata)
declare i32 @llvm.experimental.constrained.fptoui.i32.f64(double, metadata)
declare i64 @llvm.experimental.constrained.fptoui.i64.f64(double, metadata)
declare i8 @llvm.experimental.constrained.fptosi.i8.f64(double, metadata)
declare i16 @llvm.experimental.constrained.fptosi.i16.f64(double, metadata)
