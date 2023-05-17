;RUN: llc < %s -march=r600 -mcpu=cayman

define amdgpu_vs void @main(<4 x float> inreg %reg0, <4 x float> inreg %reg1, <4 x float> inreg %reg2, <4 x float> inreg %reg3) {
main_body:
  %0 = extractelement <4 x float> %reg1, i32 0
  %1 = extractelement <4 x float> %reg1, i32 1
  %2 = extractelement <4 x float> %reg1, i32 2
  %3 = extractelement <4 x float> %reg1, i32 3
  %4 = extractelement <4 x float> %reg2, i32 0
  %5 = extractelement <4 x float> %reg2, i32 1
  %6 = extractelement <4 x float> %reg2, i32 2
  %7 = extractelement <4 x float> %reg2, i32 3
  %8 = extractelement <4 x float> %reg3, i32 0
  %9 = extractelement <4 x float> %reg3, i32 1
  %10 = extractelement <4 x float> %reg3, i32 2
  %11 = extractelement <4 x float> %reg3, i32 3
  %12 = load <4 x float>, ptr addrspace(8) null
  %13 = extractelement <4 x float> %12, i32 0
  %14 = fmul float %0, %13
  %15 = load <4 x float>, ptr addrspace(8) null
  %16 = extractelement <4 x float> %15, i32 1
  %17 = fmul float %0, %16
  %18 = load <4 x float>, ptr addrspace(8) null
  %19 = extractelement <4 x float> %18, i32 2
  %20 = fmul float %0, %19
  %21 = load <4 x float>, ptr addrspace(8) null
  %22 = extractelement <4 x float> %21, i32 3
  %23 = fmul float %0, %22
  %24 = load <4 x float>, ptr addrspace(8) getelementptr ([1024 x <4 x float>], ptr addrspace(8) null, i64 0, i32 1)
  %25 = extractelement <4 x float> %24, i32 0
  %26 = fmul float %1, %25
  %27 = fadd float %26, %14
  %28 = load <4 x float>, ptr addrspace(8) getelementptr ([1024 x <4 x float>], ptr addrspace(8) null, i64 0, i32 1)
  %29 = extractelement <4 x float> %28, i32 1
  %30 = fmul float %1, %29
  %31 = fadd float %30, %17
  %32 = load <4 x float>, ptr addrspace(8) getelementptr ([1024 x <4 x float>], ptr addrspace(8) null, i64 0, i32 1)
  %33 = extractelement <4 x float> %32, i32 2
  %34 = fmul float %1, %33
  %35 = fadd float %34, %20
  %36 = load <4 x float>, ptr addrspace(8) getelementptr ([1024 x <4 x float>], ptr addrspace(8) null, i64 0, i32 1)
  %37 = extractelement <4 x float> %36, i32 3
  %38 = fmul float %1, %37
  %39 = fadd float %38, %23
  %40 = load <4 x float>, ptr addrspace(8) getelementptr ([1024 x <4 x float>], ptr addrspace(8) null, i64 0, i32 2)
  %41 = extractelement <4 x float> %40, i32 0
  %42 = fmul float %2, %41
  %43 = fadd float %42, %27
  %44 = load <4 x float>, ptr addrspace(8) getelementptr ([1024 x <4 x float>], ptr addrspace(8) null, i64 0, i32 2)
  %45 = extractelement <4 x float> %44, i32 1
  %46 = fmul float %2, %45
  %47 = fadd float %46, %31
  %48 = load <4 x float>, ptr addrspace(8) getelementptr ([1024 x <4 x float>], ptr addrspace(8) null, i64 0, i32 2)
  %49 = extractelement <4 x float> %48, i32 2
  %50 = fmul float %2, %49
  %51 = fadd float %50, %35
  %52 = load <4 x float>, ptr addrspace(8) getelementptr ([1024 x <4 x float>], ptr addrspace(8) null, i64 0, i32 2)
  %53 = extractelement <4 x float> %52, i32 3
  %54 = fmul float %2, %53
  %55 = fadd float %54, %39
  %56 = load <4 x float>, ptr addrspace(8) getelementptr ([1024 x <4 x float>], ptr addrspace(8) null, i64 0, i32 3)
  %57 = extractelement <4 x float> %56, i32 0
  %58 = fmul float %3, %57
  %59 = fadd float %58, %43
  %60 = load <4 x float>, ptr addrspace(8) getelementptr ([1024 x <4 x float>], ptr addrspace(8) null, i64 0, i32 3)
  %61 = extractelement <4 x float> %60, i32 1
  %62 = fmul float %3, %61
  %63 = fadd float %62, %47
  %64 = load <4 x float>, ptr addrspace(8) getelementptr ([1024 x <4 x float>], ptr addrspace(8) null, i64 0, i32 3)
  %65 = extractelement <4 x float> %64, i32 2
  %66 = fmul float %3, %65
  %67 = fadd float %66, %51
  %68 = load <4 x float>, ptr addrspace(8) getelementptr ([1024 x <4 x float>], ptr addrspace(8) null, i64 0, i32 3)
  %69 = extractelement <4 x float> %68, i32 3
  %70 = fmul float %3, %69
  %71 = fadd float %70, %55
  %72 = load <4 x float>, ptr addrspace(8) getelementptr ([1024 x <4 x float>], ptr addrspace(8) null, i64 0, i32 4)
  %73 = extractelement <4 x float> %72, i32 0
  %74 = load <4 x float>, ptr addrspace(8) getelementptr ([1024 x <4 x float>], ptr addrspace(8) null, i64 0, i32 4)
  %75 = extractelement <4 x float> %74, i32 1
  %76 = load <4 x float>, ptr addrspace(8) getelementptr ([1024 x <4 x float>], ptr addrspace(8) null, i64 0, i32 4)
  %77 = extractelement <4 x float> %76, i32 2
  %78 = insertelement <4 x float> undef, float %4, i32 0
  %79 = insertelement <4 x float> %78, float %5, i32 1
  %80 = insertelement <4 x float> %79, float %6, i32 2
  %81 = insertelement <4 x float> %80, float 0.000000e+00, i32 3
  %82 = insertelement <4 x float> undef, float %73, i32 0
  %83 = insertelement <4 x float> %82, float %75, i32 1
  %84 = insertelement <4 x float> %83, float %77, i32 2
  %85 = insertelement <4 x float> %84, float 0.000000e+00, i32 3
  %86 = call float @llvm.r600.dot4(<4 x float> %81, <4 x float> %85)
  %87 = insertelement <4 x float> undef, float %86, i32 0
  call void @llvm.r600.store.swizzle(<4 x float> %87, i32 2, i32 2)
  ret void
}

; Function Attrs: readnone
declare float @llvm.r600.dot4(<4 x float>, <4 x float>) #1

; Function Attrs: nounwind readonly
declare float @llvm.pow.f32(float, float) #3

declare void @llvm.r600.store.swizzle(<4 x float>, i32, i32)

attributes #1 = { readnone }
attributes #2 = { readonly }
attributes #3 = { nounwind readonly }
