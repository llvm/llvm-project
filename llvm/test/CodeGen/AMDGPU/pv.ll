; RUN: llc < %s -march=r600 | FileCheck %s

; CHECK: DOT4 * T{{[0-9]\.W}} (MASKED)
; CHECK: MAX T{{[0-9].[XYZW]}}, 0.0, PV.X

define amdgpu_vs void @main(<4 x float> inreg %reg0, <4 x float> inreg %reg1, <4 x float> inreg %reg2, <4 x float> inreg %reg3, <4 x float> inreg %reg4, <4 x float> inreg %reg5, <4 x float> inreg %reg6, <4 x float> inreg %reg7) {
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
  %12 = extractelement <4 x float> %reg4, i32 0
  %13 = extractelement <4 x float> %reg4, i32 1
  %14 = extractelement <4 x float> %reg4, i32 2
  %15 = extractelement <4 x float> %reg4, i32 3
  %16 = extractelement <4 x float> %reg5, i32 0
  %17 = extractelement <4 x float> %reg5, i32 1
  %18 = extractelement <4 x float> %reg5, i32 2
  %19 = extractelement <4 x float> %reg5, i32 3
  %20 = extractelement <4 x float> %reg6, i32 0
  %21 = extractelement <4 x float> %reg6, i32 1
  %22 = extractelement <4 x float> %reg6, i32 2
  %23 = extractelement <4 x float> %reg6, i32 3
  %24 = extractelement <4 x float> %reg7, i32 0
  %25 = extractelement <4 x float> %reg7, i32 1
  %26 = extractelement <4 x float> %reg7, i32 2
  %27 = extractelement <4 x float> %reg7, i32 3
  %28 = load <4 x float>, <4 x float> addrspace(8)* null
  %29 = extractelement <4 x float> %28, i32 0
  %30 = fmul float %0, %29
  %31 = load <4 x float>, <4 x float> addrspace(8)* null
  %32 = extractelement <4 x float> %31, i32 1
  %33 = fmul float %0, %32
  %34 = load <4 x float>, <4 x float> addrspace(8)* null
  %35 = extractelement <4 x float> %34, i32 2
  %36 = fmul float %0, %35
  %37 = load <4 x float>, <4 x float> addrspace(8)* null
  %38 = extractelement <4 x float> %37, i32 3
  %39 = fmul float %0, %38
  %40 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 1)
  %41 = extractelement <4 x float> %40, i32 0
  %42 = fmul float %1, %41
  %43 = fadd float %42, %30
  %44 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 1)
  %45 = extractelement <4 x float> %44, i32 1
  %46 = fmul float %1, %45
  %47 = fadd float %46, %33
  %48 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 1)
  %49 = extractelement <4 x float> %48, i32 2
  %50 = fmul float %1, %49
  %51 = fadd float %50, %36
  %52 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 1)
  %53 = extractelement <4 x float> %52, i32 3
  %54 = fmul float %1, %53
  %55 = fadd float %54, %39
  %56 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 2)
  %57 = extractelement <4 x float> %56, i32 0
  %58 = fmul float %2, %57
  %59 = fadd float %58, %43
  %60 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 2)
  %61 = extractelement <4 x float> %60, i32 1
  %62 = fmul float %2, %61
  %63 = fadd float %62, %47
  %64 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 2)
  %65 = extractelement <4 x float> %64, i32 2
  %66 = fmul float %2, %65
  %67 = fadd float %66, %51
  %68 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 2)
  %69 = extractelement <4 x float> %68, i32 3
  %70 = fmul float %2, %69
  %71 = fadd float %70, %55
  %72 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 3)
  %73 = extractelement <4 x float> %72, i32 0
  %74 = fmul float %3, %73
  %75 = fadd float %74, %59
  %76 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 3)
  %77 = extractelement <4 x float> %76, i32 1
  %78 = fmul float %3, %77
  %79 = fadd float %78, %63
  %80 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 3)
  %81 = extractelement <4 x float> %80, i32 2
  %82 = fmul float %3, %81
  %83 = fadd float %82, %67
  %84 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 3)
  %85 = extractelement <4 x float> %84, i32 3
  %86 = fmul float %3, %85
  %87 = fadd float %86, %71
  %88 = insertelement <4 x float> undef, float %4, i32 0
  %89 = insertelement <4 x float> %88, float %5, i32 1
  %90 = insertelement <4 x float> %89, float %6, i32 2
  %91 = insertelement <4 x float> %90, float 0.000000e+00, i32 3
  %92 = insertelement <4 x float> undef, float %4, i32 0
  %93 = insertelement <4 x float> %92, float %5, i32 1
  %94 = insertelement <4 x float> %93, float %6, i32 2
  %95 = insertelement <4 x float> %94, float 0.000000e+00, i32 3
  %96 = call float @llvm.AMDGPU.dp4(<4 x float> %91, <4 x float> %95)
  %97 = call float @fabs(float %96)
  %98 = call float @llvm.AMDGPU.rsq.clamped.f32(float %97)
  %99 = fmul float %4, %98
  %100 = fmul float %5, %98
  %101 = fmul float %6, %98
  %102 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 4)
  %103 = extractelement <4 x float> %102, i32 0
  %104 = fmul float %103, %8
  %105 = fadd float %104, %20
  %106 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 4)
  %107 = extractelement <4 x float> %106, i32 1
  %108 = fmul float %107, %9
  %109 = fadd float %108, %21
  %110 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 4)
  %111 = extractelement <4 x float> %110, i32 2
  %112 = fmul float %111, %10
  %113 = fadd float %112, %22
  %114 = call float @llvm.AMDGPU.clamp.f32(float %105, float 0.000000e+00, float 1.000000e+00)
  %115 = call float @llvm.AMDGPU.clamp.f32(float %109, float 0.000000e+00, float 1.000000e+00)
  %116 = call float @llvm.AMDGPU.clamp.f32(float %113, float 0.000000e+00, float 1.000000e+00)
  %117 = call float @llvm.AMDGPU.clamp.f32(float %15, float 0.000000e+00, float 1.000000e+00)
  %118 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 5)
  %119 = extractelement <4 x float> %118, i32 0
  %120 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 5)
  %121 = extractelement <4 x float> %120, i32 1
  %122 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 5)
  %123 = extractelement <4 x float> %122, i32 2
  %124 = insertelement <4 x float> undef, float %99, i32 0
  %125 = insertelement <4 x float> %124, float %100, i32 1
  %126 = insertelement <4 x float> %125, float %101, i32 2
  %127 = insertelement <4 x float> %126, float 0.000000e+00, i32 3
  %128 = insertelement <4 x float> undef, float %119, i32 0
  %129 = insertelement <4 x float> %128, float %121, i32 1
  %130 = insertelement <4 x float> %129, float %123, i32 2
  %131 = insertelement <4 x float> %130, float 0.000000e+00, i32 3
  %132 = call float @llvm.AMDGPU.dp4(<4 x float> %127, <4 x float> %131)
  %133 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 7)
  %134 = extractelement <4 x float> %133, i32 0
  %135 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 7)
  %136 = extractelement <4 x float> %135, i32 1
  %137 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 7)
  %138 = extractelement <4 x float> %137, i32 2
  %139 = insertelement <4 x float> undef, float %99, i32 0
  %140 = insertelement <4 x float> %139, float %100, i32 1
  %141 = insertelement <4 x float> %140, float %101, i32 2
  %142 = insertelement <4 x float> %141, float 0.000000e+00, i32 3
  %143 = insertelement <4 x float> undef, float %134, i32 0
  %144 = insertelement <4 x float> %143, float %136, i32 1
  %145 = insertelement <4 x float> %144, float %138, i32 2
  %146 = insertelement <4 x float> %145, float 0.000000e+00, i32 3
  %147 = call float @llvm.AMDGPU.dp4(<4 x float> %142, <4 x float> %146)
  %148 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 8)
  %149 = extractelement <4 x float> %148, i32 0
  %150 = fmul float %149, %8
  %151 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 8)
  %152 = extractelement <4 x float> %151, i32 1
  %153 = fmul float %152, %9
  %154 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 8)
  %155 = extractelement <4 x float> %154, i32 2
  %156 = fmul float %155, %10
  %157 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 9)
  %158 = extractelement <4 x float> %157, i32 0
  %159 = fmul float %158, %12
  %160 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 9)
  %161 = extractelement <4 x float> %160, i32 1
  %162 = fmul float %161, %13
  %163 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 9)
  %164 = extractelement <4 x float> %163, i32 2
  %165 = fmul float %164, %14
  %166 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 10)
  %167 = extractelement <4 x float> %166, i32 0
  %168 = fmul float %167, %16
  %169 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 10)
  %170 = extractelement <4 x float> %169, i32 1
  %171 = fmul float %170, %17
  %172 = load <4 x float>, <4 x float> addrspace(8)* getelementptr ([1024 x <4 x float>], [1024 x <4 x float>] addrspace(8)* null, i64 0, i32 10)
  %173 = extractelement <4 x float> %172, i32 2
  %174 = fmul float %173, %18
  %175 = fcmp uge float %132, 0.000000e+00
  %176 = select i1 %175, float %132, float 0.000000e+00
  %177 = fcmp uge float %147, 0.000000e+00
  %178 = select i1 %177, float %147, float 0.000000e+00
  %179 = call float @llvm.pow.f32(float %178, float %24)
  %180 = fcmp ult float %132, 0.000000e+00
  %181 = select i1 %180, float 0.000000e+00, float %179
  %182 = fadd float %150, %105
  %183 = fadd float %153, %109
  %184 = fadd float %156, %113
  %185 = fmul float %176, %159
  %186 = fadd float %185, %182
  %187 = fmul float %176, %162
  %188 = fadd float %187, %183
  %189 = fmul float %176, %165
  %190 = fadd float %189, %184
  %191 = fmul float %181, %168
  %192 = fadd float %191, %186
  %193 = fmul float %181, %171
  %194 = fadd float %193, %188
  %195 = fmul float %181, %174
  %196 = fadd float %195, %190
  %197 = call float @llvm.AMDGPU.clamp.f32(float %192, float 0.000000e+00, float 1.000000e+00)
  %198 = call float @llvm.AMDGPU.clamp.f32(float %194, float 0.000000e+00, float 1.000000e+00)
  %199 = call float @llvm.AMDGPU.clamp.f32(float %196, float 0.000000e+00, float 1.000000e+00)
  %200 = insertelement <4 x float> undef, float %75, i32 0
  %201 = insertelement <4 x float> %200, float %79, i32 1
  %202 = insertelement <4 x float> %201, float %83, i32 2
  %203 = insertelement <4 x float> %202, float %87, i32 3
  call void @llvm.R600.store.swizzle(<4 x float> %203, i32 60, i32 1)
  %204 = insertelement <4 x float> undef, float %197, i32 0
  %205 = insertelement <4 x float> %204, float %198, i32 1
  %206 = insertelement <4 x float> %205, float %199, i32 2
  %207 = insertelement <4 x float> %206, float %117, i32 3
  call void @llvm.R600.store.swizzle(<4 x float> %207, i32 0, i32 2)
  ret void
}

; Function Attrs: readnone
declare float @llvm.AMDGPU.dp4(<4 x float>, <4 x float>) #1

; Function Attrs: readonly
declare float @fabs(float) #2

; Function Attrs: readnone
declare float @llvm.AMDGPU.rsq.clamped.f32(float) #1

; Function Attrs: readnone
declare float @llvm.AMDGPU.clamp.f32(float, float, float) #1

; Function Attrs: nounwind readonly
declare float @llvm.pow.f32(float, float) #3

declare void @llvm.R600.store.swizzle(<4 x float>, i32, i32)

attributes #1 = { readnone }
attributes #2 = { readonly }
attributes #3 = { nounwind readonly }
