; RUN: opt -S -passes=inline -mtriple=arm-eabi -pass-remarks=.* -pass-remarks-missed=.* < %s 2>&1 | FileCheck %s -check-prefix=NOFP
; RUN: opt -S -passes=inline -mtriple=arm-eabi -mattr=+vfp2 -pass-remarks=.* -pass-remarks-missed=.* < %s 2>&1 | FileCheck %s -check-prefix=FULLFP
; RUN: opt -S -passes=inline -mtriple=arm-eabi -mattr=+vfp2,-fp64 -pass-remarks=.* -pass-remarks-missed=.* < %s 2>&1 | FileCheck %s -check-prefix=SINGLEFP
; Make sure that soft float implementations are calculated as being more expensive
; to the inliner.

; NOFP-DAG: 'single' not inlined into 'test_single' because too costly to inline (cost=125, threshold=75)
; NOFP-DAG: 'single' not inlined into 'test_single' because too costly to inline (cost=125, threshold=75)
; NOFP-DAG: 'single_cheap' inlined into 'test_single_cheap' with (cost=-15, threshold=75)
; NOFP-DAG: 'single_cheap' inlined into 'test_single_cheap' with (cost=-15015, threshold=75)
; NOFP-DAG: 'double' not inlined into 'test_double' because too costly to inline (cost=125, threshold=75)
; NOFP-DAG: 'double' not inlined into 'test_double' because too costly to inline (cost=125, threshold=75)
; NOFP-DAG: 'single_force_soft' not inlined into 'test_single_force_soft' because too costly to inline (cost=125, threshold=75)
; NOFP-DAG: 'single_force_soft' not inlined into 'test_single_force_soft' because too costly to inline (cost=125, threshold=75)
; NOFP-DAG: 'single_force_soft_fneg' not inlined into 'test_single_force_soft_fneg' because too costly to inline (cost=100, threshold=75)
; NOFP-DAG: 'single_force_soft_fneg' not inlined into 'test_single_force_soft_fneg' because too costly to inline (cost=100, threshold=75)

; FULLFP-DAG: 'single' inlined into 'test_single' with (cost=0, threshold=75)
; FULLFP-DAG: 'single' inlined into 'test_single' with (cost=-15000, threshold=75)
; FULLFP-DAG: 'single_cheap' inlined into 'test_single_cheap' with (cost=-15, threshold=75)
; FULLFP-DAG: 'single_cheap' inlined into 'test_single_cheap' with (cost=-15015, threshold=75)
; FULLFP-DAG: 'double' inlined into 'test_double' with (cost=0, threshold=75)
; FULLFP-DAG: 'double' inlined into 'test_double' with (cost=-15000, threshold=75)
; FULLFP-DAG: 'single_force_soft' not inlined into 'test_single_force_soft' because too costly to inline (cost=125, threshold=75)
; FULLFP-DAG: 'single_force_soft' not inlined into 'test_single_force_soft' because too costly to inline (cost=125, threshold=75)
; FULLFP-DAG: 'single_force_soft_fneg' not inlined into 'test_single_force_soft_fneg' because too costly to inline (cost=100, threshold=75)
; FULLFP-DAG: 'single_force_soft_fneg' not inlined into 'test_single_force_soft_fneg' because too costly to inline (cost=100, threshold=75)

; SINGLEFP-DAG: 'single' inlined into 'test_single' with (cost=0, threshold=75)
; SINGLEFP-DAG: 'single' inlined into 'test_single' with (cost=-15000, threshold=75)
; SINGLEFP-DAG: 'single_cheap' inlined into 'test_single_cheap' with (cost=-15, threshold=75)
; SINGLEFP-DAG: 'single_cheap' inlined into 'test_single_cheap' with (cost=-15015, threshold=75)
; SINGLEFP-DAG: 'double' not inlined into 'test_double' because too costly to inline (cost=125, threshold=75)
; SINGLEFP-DAG: 'double' not inlined into 'test_double' because too costly to inline (cost=125, threshold=75)
; SINGLEFP-DAG: 'single_force_soft' not inlined into 'test_single_force_soft' because too costly to inline (cost=125, threshold=75)
; SINGLEFP-DAG: 'single_force_soft' not inlined into 'test_single_force_soft' because too costly to inline (cost=125, threshold=75)
; SINGLEFP-DAG: 'single_force_soft_fneg' not inlined into 'test_single_force_soft_fneg' because too costly to inline (cost=100, threshold=75)
; SINGLEFP-DAG: 'single_force_soft_fneg' not inlined into 'test_single_force_soft_fneg' because too costly to inline (cost=100, threshold=75)

define i32 @test_single(i32 %a, i8 %b, i32 %c, i8 %d) #0 {
  %call = call float @single(i32 %a, i8 zeroext %b)
  %call2 = call float @single(i32 %c, i8 zeroext %d)
  ret i32 0
}

define i32 @test_single_cheap(i32 %a, i8 %b, i32 %c, i8 %d) #0 {
  %call = call float @single_cheap(i32 %a, i8 zeroext %b)
  %call2 = call float @single_cheap(i32 %c, i8 zeroext %d)
  ret i32 0
}

define i32 @test_double(i32 %a, i8 %b, i32 %c, i8 %d) #0 {
  %call = call double @double(i32 %a, i8 zeroext %b)
  %call2 = call double @double(i32 %c, i8 zeroext %d)
  ret i32 0
}

define i32 @test_single_force_soft(i32 %a, i8 %b, i32 %c, i8 %d) #1 {
  %call = call float @single_force_soft(i32 %a, i8 zeroext %b) #1
  %call2 = call float @single_force_soft(i32 %c, i8 zeroext %d) #1
  ret i32 0
}

define i32 @test_single_force_soft_fneg(i32 %a, i8 %b, i32 %c, i8 %d) #1 {
  %call = call float @single_force_soft_fneg(i32 %a, i8 zeroext %b) #1
  %call2 = call float @single_force_soft_fneg(i32 %c, i8 zeroext %d) #1
  ret i32 0
}

define internal float @single(i32 %response, i8 zeroext %value1) #0 {
entry:
  %conv = zext i8 %value1 to i32
  %sub = add nsw i32 %conv, -1
  %conv1 = sitofp i32 %sub to float
  %0 = tail call float @llvm.pow.f32(float 0x3FF028F5C0000000, float %conv1)
  %mul = fmul float %0, 2.620000e+03
  %conv2 = sitofp i32 %response to float
  %sub3 = fsub float %conv2, %mul
  %div = fdiv float %sub3, %mul
  ret float %div
}

define internal float @single_cheap(i32 %response, i8 zeroext %value1) #0 {
entry:
  %conv = zext i8 %value1 to i32
  %sub = add nsw i32 %conv, -1
  %conv1 = bitcast i32 %sub to float
  %conv2 = bitcast i32 %response to float
  %0 = tail call float @llvm.pow.f32(float %conv2, float %conv1)
  %1 = tail call float @llvm.pow.f32(float %0, float %0)
  %2 = tail call float @llvm.pow.f32(float %1, float %1)
  ret float %2
}

define internal double @double(i32 %response, i8 zeroext %value1) #0 {
entry:
  %conv = zext i8 %value1 to i32
  %sub = add nsw i32 %conv, -1
  %conv1 = sitofp i32 %sub to double
  %0 = tail call double @llvm.pow.f64(double 0x3FF028F5C0000000, double %conv1)
  %mul = fmul double %0, 2.620000e+03
  %conv2 = sitofp i32 %response to double
  %sub3 = fsub double %conv2, %mul
  %div = fdiv double %sub3, %mul
  ret double %div
}

define internal float @single_force_soft(i32 %response, i8 zeroext %value1) #1 {
entry:
  %conv = zext i8 %value1 to i32
  %sub = add nsw i32 %conv, -1
  %conv1 = sitofp i32 %sub to float
  %0 = tail call float @llvm.pow.f32(float 0x3FF028F5C0000000, float %conv1)
  %mul = fmul float %0, 2.620000e+03
  %conv2 = sitofp i32 %response to float
  %sub3 = fsub float %conv2, %mul
  %div = fdiv float %sub3, %mul
  ret float %div
}

define internal float @single_force_soft_fneg(i32 %response, i8 zeroext %value1) #1 {
entry:
  %conv = zext i8 %value1 to i32
  %sub = add nsw i32 %conv, -1
  %conv1 = sitofp i32 %sub to float
  %0 = tail call float @llvm.pow.f32(float 0x3FF028F5C0000000, float %conv1)
  %mul = fsub float -0.0, %0
  %conv2 = sitofp i32 %response to float
  %sub3 = fsub float %conv2, %mul
  %div = fdiv float %sub3, %mul
  ret float %div
}

declare float @llvm.pow.f32(float, float) optsize minsize
declare double @llvm.pow.f64(double, double) optsize minsize

attributes #0 = { optsize }
attributes #1 = { optsize "use-soft-float"="true" "target-features"="+soft-float" }
