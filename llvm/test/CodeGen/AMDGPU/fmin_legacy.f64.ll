; RUN: llc -march=amdgcn < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

declare i32 @llvm.amdgcn.workitem.id.x() #1

; FUNC-LABEL: @test_fmin_legacy_f64
define void @test_fmin_legacy_f64(<4 x double> addrspace(1)* %out, <4 x double> inreg %reg0) #0 {
   %r0 = extractelement <4 x double> %reg0, i32 0
   %r1 = extractelement <4 x double> %reg0, i32 1
   %r2 = fcmp uge double %r0, %r1
   %r3 = select i1 %r2, double %r1, double %r0
   %vec = insertelement <4 x double> undef, double %r3, i32 0
   store <4 x double> %vec, <4 x double> addrspace(1)* %out, align 16
   ret void
}

; FUNC-LABEL: @test_fmin_legacy_ule_f64
define void @test_fmin_legacy_ule_f64(double addrspace(1)* %out, double addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %gep.0 = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr double, double addrspace(1)* %gep.0, i32 1

  %a = load double, double addrspace(1)* %gep.0, align 8
  %b = load double, double addrspace(1)* %gep.1, align 8

  %cmp = fcmp ule double %a, %b
  %val = select i1 %cmp, double %a, double %b
  store double %val, double addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: @test_fmin_legacy_ole_f64
define void @test_fmin_legacy_ole_f64(double addrspace(1)* %out, double addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %gep.0 = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr double, double addrspace(1)* %gep.0, i32 1

  %a = load double, double addrspace(1)* %gep.0, align 8
  %b = load double, double addrspace(1)* %gep.1, align 8

  %cmp = fcmp ole double %a, %b
  %val = select i1 %cmp, double %a, double %b
  store double %val, double addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: @test_fmin_legacy_olt_f64
define void @test_fmin_legacy_olt_f64(double addrspace(1)* %out, double addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %gep.0 = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr double, double addrspace(1)* %gep.0, i32 1

  %a = load double, double addrspace(1)* %gep.0, align 8
  %b = load double, double addrspace(1)* %gep.1, align 8

  %cmp = fcmp olt double %a, %b
  %val = select i1 %cmp, double %a, double %b
  store double %val, double addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: @test_fmin_legacy_ult_f64
define void @test_fmin_legacy_ult_f64(double addrspace(1)* %out, double addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %gep.0 = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr double, double addrspace(1)* %gep.0, i32 1

  %a = load double, double addrspace(1)* %gep.0, align 8
  %b = load double, double addrspace(1)* %gep.1, align 8

  %cmp = fcmp ult double %a, %b
  %val = select i1 %cmp, double %a, double %b
  store double %val, double addrspace(1)* %out, align 8
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
