; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; Test that the new FP instruction intrinsics (llvm.fadd, llvm.fsub, etc.)
; parse correctly, round-trip through bitcode, and have the expected attributes.

define void @test_arithmetic(float %a, float %b, double %da, double %db) {
; CHECK-LABEL: define void @test_arithmetic

  %fadd  = call float @llvm.fadd.f32(float %a, float %b)
  %fsub  = call float @llvm.fsub.f32(float %a, float %b)
  %fmul  = call float @llvm.fmul.f32(float %a, float %b)
  %fdiv  = call float @llvm.fdiv.f32(float %a, float %b)
  %frem  = call float @llvm.frem.f32(float %a, float %b)
  %dfadd = call double @llvm.fadd.f64(double %da, double %db)
  %dfsub = call double @llvm.fsub.f64(double %da, double %db)
  %dfmul = call double @llvm.fmul.f64(double %da, double %db)
  %dfdiv = call double @llvm.fdiv.f64(double %da, double %db)

; CHECK: call float @llvm.fadd.f32(float %a, float %b)
; CHECK: call float @llvm.fsub.f32(float %a, float %b)
; CHECK: call float @llvm.fmul.f32(float %a, float %b)
; CHECK: call float @llvm.fdiv.f32(float %a, float %b)
; CHECK: call float @llvm.frem.f32(float %a, float %b)
; CHECK: call double @llvm.fadd.f64(double %da, double %db)

  ret void
}

define void @test_fast_math_flags(float %a, float %b) {
; CHECK-LABEL: define void @test_fast_math_flags

  %r1 = call fast float @llvm.fadd.f32(float %a, float %b)
  %r2 = call nnan nsz float @llvm.fsub.f32(float %a, float %b)
  %r3 = call reassoc float @llvm.fmul.f32(float %a, float %b)

; CHECK: call fast float @llvm.fadd.f32(float %a, float %b)
; CHECK: call nnan nsz float @llvm.fsub.f32(float %a, float %b)
; CHECK: call reassoc float @llvm.fmul.f32(float %a, float %b)

  ret void
}

define void @test_conversions(double %d, float %f, i32 %i) {
; CHECK-LABEL: define void @test_conversions

  %fptrunc = call float @llvm.fptrunc.f32.f64(double %d)
  %fpext   = call double @llvm.fpext.f64.f32(float %f)
  %sitofp  = call float @llvm.sitofp.f32.i32(i32 %i)
  %uitofp  = call float @llvm.uitofp.f32.i32(i32 %i)
  %fptosi  = call i32 @llvm.fptosi.i32.f32(float %f)
  %fptoui  = call i32 @llvm.fptoui.i32.f32(float %f)

; CHECK: call float @llvm.fptrunc.f32.f64(double %d)
; CHECK: call double @llvm.fpext.f64.f32(float %f)
; CHECK: call float @llvm.sitofp.f32.i32(i32 %i)
; CHECK: call float @llvm.uitofp.f32.i32(i32 %i)
; CHECK: call i32 @llvm.fptosi.i32.f32(float %f)
; CHECK: call i32 @llvm.fptoui.i32.f32(float %f)

  ret void
}

define void @test_fcmp(float %a, float %b) {
; CHECK-LABEL: define void @test_fcmp

  %oeq = call i1 @llvm.fcmp.f32(float %a, float %b, metadata !"oeq")
  %ogt = call i1 @llvm.fcmp.f32(float %a, float %b, metadata !"ogt")
  %oge = call i1 @llvm.fcmp.f32(float %a, float %b, metadata !"oge")
  %olt = call i1 @llvm.fcmp.f32(float %a, float %b, metadata !"olt")
  %ole = call i1 @llvm.fcmp.f32(float %a, float %b, metadata !"ole")
  %one = call i1 @llvm.fcmp.f32(float %a, float %b, metadata !"one")
  %ord = call i1 @llvm.fcmp.f32(float %a, float %b, metadata !"ord")
  %ueq = call i1 @llvm.fcmp.f32(float %a, float %b, metadata !"ueq")
  %ugt = call i1 @llvm.fcmp.f32(float %a, float %b, metadata !"ugt")
  %uge = call i1 @llvm.fcmp.f32(float %a, float %b, metadata !"uge")
  %ult = call i1 @llvm.fcmp.f32(float %a, float %b, metadata !"ult")
  %ule = call i1 @llvm.fcmp.f32(float %a, float %b, metadata !"ule")
  %une = call i1 @llvm.fcmp.f32(float %a, float %b, metadata !"une")
  %uno = call i1 @llvm.fcmp.f32(float %a, float %b, metadata !"uno")
  %t   = call i1 @llvm.fcmp.f32(float %a, float %b, metadata !"true")
  %f1  = call i1 @llvm.fcmp.f32(float %a, float %b, metadata !"false")

; CHECK: call i1 @llvm.fcmp.f32(float %a, float %b, metadata !"oeq")
; CHECK: call i1 @llvm.fcmp.f32(float %a, float %b, metadata !"ogt")

  ret void
}

define void @test_fcmps(float %a, float %b) {
; CHECK-LABEL: define void @test_fcmps

  %r = call i1 @llvm.fcmps.f32(float %a, float %b, metadata !"oeq")

; CHECK: call i1 @llvm.fcmps.f32(float %a, float %b, metadata !"oeq")

  ret void
}

; Arithmetic intrinsics are memory(none) and speculatable (no strictfp needed).
; CHECK-DAG: declare float @llvm.fadd.f32(float, float)
; CHECK-DAG: declare float @llvm.fsub.f32(float, float)
; CHECK-DAG: declare float @llvm.fmul.f32(float, float)
; CHECK-DAG: declare float @llvm.fdiv.f32(float, float)
; CHECK-DAG: declare float @llvm.frem.f32(float, float)
; CHECK-DAG: declare double @llvm.fadd.f64(double, double)
; CHECK-DAG: declare float @llvm.fptrunc.f32.f64(double)
; CHECK-DAG: declare double @llvm.fpext.f64.f32(float)
; CHECK-DAG: declare float @llvm.sitofp.f32.i32(i32)
; CHECK-DAG: declare float @llvm.uitofp.f32.i32(i32)
; CHECK-DAG: declare i32 @llvm.fptosi.i32.f32(float)
; CHECK-DAG: declare i32 @llvm.fptoui.i32.f32(float)
; CHECK-DAG: declare i1 @llvm.fcmp.f32(float, float, metadata)
; fcmps has memory(inaccessiblemem: readwrite) + willreturn
; CHECK-DAG: declare i1 @llvm.fcmps.f32(float, float, metadata) #[[ATTR:[0-9]+]]
; CHECK-DAG: attributes #[[ATTR]] = { nocreateundeforpoison nounwind willreturn memory(inaccessiblemem: readwrite) }
