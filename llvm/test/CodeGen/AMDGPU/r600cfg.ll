;RUN: llc < %s -mtriple=r600 -mcpu=redwood

define amdgpu_vs void @main(<4 x float> inreg %reg0, <4 x float> inreg %reg1) {
main_body:
  %0 = extractelement <4 x float> %reg1, i32 0
  %1 = extractelement <4 x float> %reg1, i32 1
  %2 = extractelement <4 x float> %reg1, i32 2
  %3 = extractelement <4 x float> %reg1, i32 3
  %4 = bitcast float %0 to i32
  %5 = icmp eq i32 %4, 0
  %6 = sext i1 %5 to i32
  %7 = bitcast i32 %6 to float
  %8 = bitcast float %7 to i32
  %9 = icmp ne i32 %8, 0
  %. = select i1 %9, float 0x36A0000000000000, float %0
  br label %LOOP

LOOP:                                             ; preds = %LOOP47, %main_body
  %temp12.0 = phi float [ 0x36A0000000000000, %main_body ], [ %temp12.1, %LOOP47 ]
  %temp8.0 = phi float [ 0.000000e+00, %main_body ], [ %38, %LOOP47 ]
  %temp4.1 = phi float [ %., %main_body ], [ %52, %LOOP47 ]
  %10 = bitcast float %temp4.1 to i32
  %11 = icmp eq i32 %10, 1
  %12 = sext i1 %11 to i32
  %13 = bitcast i32 %12 to float
  %14 = bitcast float %13 to i32
  %15 = icmp ne i32 %14, 0
  br i1 %15, label %IF41, label %ENDIF40

IF41:                                             ; preds = %LOOP
  %16 = insertelement <4 x float> undef, float %0, i32 0
  %17 = insertelement <4 x float> %16, float %temp8.0, i32 1
  %18 = insertelement <4 x float> %17, float %temp12.0, i32 2
  %19 = insertelement <4 x float> %18, float 0.000000e+00, i32 3
  call void @llvm.r600.store.stream.output(<4 x float> %19, i32 0, i32 0, i32 1)
  %20 = insertelement <4 x float> undef, float %0, i32 0
  %21 = insertelement <4 x float> %20, float %temp8.0, i32 1
  %22 = insertelement <4 x float> %21, float %temp12.0, i32 2
  %23 = insertelement <4 x float> %22, float 0.000000e+00, i32 3
  call void @llvm.r600.store.stream.output(<4 x float> %23, i32 0, i32 0, i32 2)
  %24 = insertelement <4 x float> undef, float %0, i32 0
  %25 = insertelement <4 x float> %24, float %temp8.0, i32 1
  %26 = insertelement <4 x float> %25, float %temp12.0, i32 2
  %27 = insertelement <4 x float> %26, float 0.000000e+00, i32 3
  call void @llvm.r600.store.stream.output(<4 x float> %27, i32 0, i32 0, i32 4)
  %28 = insertelement <4 x float> undef, float 0.000000e+00, i32 0
  %29 = insertelement <4 x float> %28, float 0.000000e+00, i32 1
  %30 = insertelement <4 x float> %29, float 0.000000e+00, i32 2
  %31 = insertelement <4 x float> %30, float 0.000000e+00, i32 3
  call void @llvm.r600.store.swizzle(<4 x float> %31, i32 60, i32 1)
  %32 = insertelement <4 x float> undef, float %0, i32 0
  %33 = insertelement <4 x float> %32, float %temp8.0, i32 1
  %34 = insertelement <4 x float> %33, float %temp12.0, i32 2
  %35 = insertelement <4 x float> %34, float 0.000000e+00, i32 3
  call void @llvm.r600.store.swizzle(<4 x float> %35, i32 0, i32 2)
  ret void

ENDIF40:                                          ; preds = %LOOP
  %36 = bitcast float %temp8.0 to i32
  %37 = add i32 %36, 1
  %38 = bitcast i32 %37 to float
  %39 = bitcast float %temp4.1 to i32
  %40 = urem i32 %39, 2
  %41 = bitcast i32 %40 to float
  %42 = bitcast float %41 to i32
  %43 = icmp eq i32 %42, 0
  %44 = sext i1 %43 to i32
  %45 = bitcast i32 %44 to float
  %46 = bitcast float %45 to i32
  %47 = icmp ne i32 %46, 0
  %48 = bitcast float %temp4.1 to i32
  br i1 %47, label %IF44, label %ELSE45

IF44:                                             ; preds = %ENDIF40
  %49 = udiv i32 %48, 2
  br label %ENDIF43

ELSE45:                                           ; preds = %ENDIF40
  %50 = mul i32 3, %48
  %51 = add i32 %50, 1
  br label %ENDIF43

ENDIF43:                                          ; preds = %ELSE45, %IF44
  %.sink = phi i32 [ %49, %IF44 ], [ %51, %ELSE45 ]
  %52 = bitcast i32 %.sink to float
  %53 = load <4 x float>, ptr addrspace(8) null
  %54 = extractelement <4 x float> %53, i32 0
  %55 = bitcast float %54 to i32
  br label %LOOP47

LOOP47:                                           ; preds = %ENDIF48, %ENDIF43
  %temp12.1 = phi float [ %temp12.0, %ENDIF43 ], [ %67, %ENDIF48 ]
  %temp28.0 = phi float [ 0.000000e+00, %ENDIF43 ], [ %70, %ENDIF48 ]
  %56 = bitcast float %temp28.0 to i32
  %57 = icmp uge i32 %56, %55
  %58 = sext i1 %57 to i32
  %59 = bitcast i32 %58 to float
  %60 = bitcast float %59 to i32
  %61 = icmp ne i32 %60, 0
  br i1 %61, label %LOOP, label %ENDIF48

ENDIF48:                                          ; preds = %LOOP47
  %62 = bitcast float %temp12.1 to i32
  %63 = mul i32 %62, 2
  %64 = bitcast i32 %63 to float
  %65 = bitcast float %64 to i32
  %66 = urem i32 %65, 2147483647
  %67 = bitcast i32 %66 to float
  %68 = bitcast float %temp28.0 to i32
  %69 = add i32 %68, 1
  %70 = bitcast i32 %69 to float
  br label %LOOP47
}

declare void @llvm.r600.store.stream.output(<4 x float>, i32, i32, i32)

declare void @llvm.r600.store.swizzle(<4 x float>, i32, i32)
