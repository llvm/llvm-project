; RUN: opt -S -inline < %s | FileCheck %s
; RUN: opt -S -passes='cgscc(inline)' < %s | FileCheck %s
; Make sure that soft float implementations are calculated as being more expensive
; to the inliner.

define i32 @test_nofp() #0 {
; f_nofp() has the "use-soft-float" attribute, so it should never get inlined.
; CHECK-LABEL: test_nofp
; CHECK: call float @f_nofp 
entry:
  %responseX = alloca i32, align 4
  %responseY = alloca i32, align 4
  %responseZ = alloca i32, align 4
  %valueX = alloca i8, align 1
  %valueY = alloca i8, align 1
  %valueZ = alloca i8, align 1

  call void @getX(i32* %responseX, i8* %valueX)
  call void @getY(i32* %responseY, i8* %valueY)
  call void @getZ(i32* %responseZ, i8* %valueZ)

  %0 = load i32, i32* %responseX
  %1 = load i8, i8* %valueX
  %call = call float @f_nofp(i32 %0, i8 zeroext %1)
  %2 = load i32, i32* %responseZ
  %3 = load i8, i8* %valueZ
  %call2 = call float @f_nofp(i32 %2, i8 zeroext %3)
  %call3 = call float @fabsf(float %call)
  %cmp = fcmp ogt float %call3, 0x3FC1EB8520000000
  br i1 %cmp, label %if.end12, label %if.else

if.else:                                          ; preds = %entry
  %4 = load i32, i32* %responseY
  %5 = load i8, i8* %valueY
  %call1 = call float @f_nofp(i32 %4, i8 zeroext %5)
  %call4 = call float @fabsf(float %call1)
  %cmp5 = fcmp ogt float %call4, 0x3FC1EB8520000000
  br i1 %cmp5, label %if.end12, label %if.else7

if.else7:                                         ; preds = %if.else
  %call8 = call float @fabsf(float %call2)
  %cmp9 = fcmp ogt float %call8, 0x3FC1EB8520000000
  br i1 %cmp9, label %if.then10, label %if.end12

if.then10:                                        ; preds = %if.else7
  br label %if.end12

if.end12:                                         ; preds = %if.else, %entry, %if.then10, %if.else7
  %success.0 = phi i32 [ 0, %if.then10 ], [ 1, %if.else7 ], [ 0, %entry ], [ 0, %if.else ]
  ret i32 %success.0
}

define i32 @test_hasfp() #0 {
; f_hasfp()  does not have the "use-soft-float" attribute, so it should get inlined.
; CHECK-LABEL: test_hasfp
; CHECK-NOT: call float @f_hasfp 
entry:
  %responseX = alloca i32, align 4
  %responseY = alloca i32, align 4
  %responseZ = alloca i32, align 4
  %valueX = alloca i8, align 1
  %valueY = alloca i8, align 1
  %valueZ = alloca i8, align 1

  call void @getX(i32* %responseX, i8* %valueX)
  call void @getY(i32* %responseY, i8* %valueY)
  call void @getZ(i32* %responseZ, i8* %valueZ)

  %0 = load i32, i32* %responseX
  %1 = load i8, i8* %valueX
  %call = call float @f_hasfp(i32 %0, i8 zeroext %1)
  %2 = load i32, i32* %responseZ
  %3 = load i8, i8* %valueZ
  %call2 = call float @f_hasfp(i32 %2, i8 zeroext %3)
  %call3 = call float @fabsf(float %call)
  %cmp = fcmp ogt float %call3, 0x3FC1EB8520000000
  br i1 %cmp, label %if.end12, label %if.else

if.else:                                          ; preds = %entry
  %4 = load i32, i32* %responseY
  %5 = load i8, i8* %valueY
  %call1 = call float @f_hasfp(i32 %4, i8 zeroext %5)
  %call4 = call float @fabsf(float %call1)
  %cmp5 = fcmp ogt float %call4, 0x3FC1EB8520000000
  br i1 %cmp5, label %if.end12, label %if.else7

if.else7:                                         ; preds = %if.else
  %call8 = call float @fabsf(float %call2)
  %cmp9 = fcmp ogt float %call8, 0x3FC1EB8520000000
  br i1 %cmp9, label %if.then10, label %if.end12

if.then10:                                        ; preds = %if.else7
  br label %if.end12

if.end12:                                         ; preds = %if.else, %entry, %if.then10, %if.else7
  %success.0 = phi i32 [ 0, %if.then10 ], [ 1, %if.else7 ], [ 0, %entry ], [ 0, %if.else ]
  ret i32 %success.0
}

declare void @getX(i32*, i8*) #0

declare void @getY(i32*, i8*) #0

declare void @getZ(i32*, i8*) #0

define internal float @f_hasfp(i32 %response, i8 zeroext %value1) #0 {
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

define internal float @f_nofp(i32 %response, i8 zeroext %value1) #1 {
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

declare float @fabsf(float) optsize minsize

declare float @llvm.pow.f32(float, float) optsize minsize

attributes #0 = { optsize }
attributes #1 = { optsize "use-soft-float"="true" }
