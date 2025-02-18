; RUN: not llc -mtriple=amdgcn -mcpu=gfx942 -O3 -global-isel=false -o - %s 2>&1 | FileCheck -check-prefix=ERR %s

source_filename = "llvm.amdgcn.init.exec.wave32.ll"

@G = global i32 -2147483648
@G.1 = global <32 x i32> splat (i32 1)
@G.2 = global <16 x i64> splat (i64 1)
@G.3 = global <8 x i1> zeroinitializer

define amdgpu_ps float @test_init_exec(float %a, float %b) {
main_body:
  %s = fadd float %a, %b
  call void @llvm.amdgcn.init.exec(i64 74565)
  ret float %s
}

define amdgpu_ps float @test_init_exec_from_input(i32 inreg %0, i32 inreg %1, i32 inreg %2, i32 inreg %count, float %a, float %b) {
main_body:
  %LGV2 = load <16 x i64>, ptr @G.2, align 128
  %LGV1 = load <32 x i32>, ptr @G.1, align 128
  %LGV = load i32, ptr @G, align 4
  %C = call <8 x i1> @f(<32 x i32> %LGV1, <16 x i64> %LGV2, <2 x half> splat (half 0xH5140))
  %B = or i32 0, %LGV
  %s = fadd float %a, %b
  call void @llvm.amdgcn.init.exec.from.input(i32 %B, i32 8)
  store <8 x i1> %C, ptr @G.3, align 1
  ret float %s
}

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.init.exec(i64 immarg) #0

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.init.exec.from.input(i32, i32 immarg) #0

declare <8 x i1> @f(<32 x i32>, <16 x i64>, <2 x half>)

attributes #0 = { convergent nocallback nofree nounwind willreturn }

ERR: error: <unknown>:0:0: in function test_init_exec_from_input float (i32, i32, i32, i32, float, float): EXEC must be initialized using function argument
