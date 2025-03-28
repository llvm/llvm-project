; RUN: llc -mtriple=amdgcn -mcpu=gfx90a -O1 -global-isel=true < %s

; Function Attrs: nounwind
define amdgpu_kernel void @test_iglp_opt() #0 {
entry:
  call void @llvm.amdgcn.iglp.opt(i32 0) #4
  ret void
}

; Function Attrs: nounwind
define amdgpu_kernel void @test_iglp_opt_mfma_gemm(ptr addrspace(3) noalias %in, ptr addrspace(3) noalias %out) #0 {
entry:
  call void @llvm.amdgcn.iglp.opt(i32 0)
  %idx = call i32 @llvm.amdgcn.workitem.id.x()
  %load.0.addr = getelementptr <32 x float>, ptr addrspace(3) %in, i32 %idx
  %load.0 = load <32 x float>, ptr addrspace(3) %load.0.addr, align 128
  %load.1.addr = getelementptr <32 x float>, ptr addrspace(3) %load.0.addr, i32 64
  %load.1 = load <32 x float>, ptr addrspace(3) %load.1.addr, align 128
  %load.2.addr = getelementptr <32 x float>, ptr addrspace(3) %load.1.addr, i32 128
  %load.2 = load <32 x float>, ptr addrspace(3) %load.2.addr, align 128
  %load.3.addr = getelementptr <32 x float>, ptr addrspace(3) %load.2.addr, i32 192
  %load.3 = load <32 x float>, ptr addrspace(3) %load.3.addr, align 128
  %load.4.addr = getelementptr <32 x float>, ptr addrspace(3) %load.3.addr, i32 256
  %load.4 = load <32 x float>, ptr addrspace(3) %load.4.addr, align 128
  %mai.0 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.000000e+00, float 2.000000e+00, <32 x float> %load.0, i32 0, i32 0, i32 0)
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.000000e+00, float 2.000000e+00, <32 x float> %load.1, i32 0, i32 0, i32 0)
  %mai.2 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.000000e+00, float 2.000000e+00, <32 x float> %load.2, i32 0, i32 0, i32 0)
  %mai.3 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.000000e+00, float 2.000000e+00, <32 x float> %load.3, i32 0, i32 0, i32 0)
  %mai.4 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.000000e+00, float 2.000000e+00, <32 x float> %load.4, i32 0, i32 0, i32 0)
  %store.0.addr = getelementptr <32 x float>, ptr addrspace(3) %out, i32 %idx
  store <32 x float> %mai.0, ptr addrspace(3) %store.0.addr, align 128
  %store.1.addr = getelementptr <32 x float>, ptr addrspace(3) %out, i32 64
  store <32 x float> %mai.1, ptr addrspace(3) %store.1.addr, align 128
  %store.2.addr = getelementptr <32 x float>, ptr addrspace(3) %out, i32 128
  store <32 x float> %mai.2, ptr addrspace(3) %store.2.addr, align 128
  %store.3.addr = getelementptr <32 x float>, ptr addrspace(3) %out, i32 192
  store <32 x float> %mai.3, ptr addrspace(3) %store.3.addr, align 128
  %store.4.addr = getelementptr <32 x float>, ptr addrspace(3) %out, i32 256
  store <32 x float> %mai.4, ptr addrspace(3) %store.4.addr, align 128
  ret void
}

; Function Attrs: nounwind
define amdgpu_kernel void @test_iglp_opt_rev_mfma_gemm(ptr addrspace(3) noalias %in, ptr addrspace(3) noalias %out) #0 {
entry:
  call void @llvm.amdgcn.iglp.opt(i32 1)
  %idx = call i32 @llvm.amdgcn.workitem.id.x()
  %load.0.addr = getelementptr <32 x float>, ptr addrspace(3) %in, i32 %idx
  %load.0 = load <32 x float>, ptr addrspace(3) %load.0.addr, align 128
  %load.1.addr = getelementptr <32 x float>, ptr addrspace(3) %load.0.addr, i32 64
  %load.1 = load <32 x float>, ptr addrspace(3) %load.1.addr, align 128
  %load.2.addr = getelementptr <32 x float>, ptr addrspace(3) %load.1.addr, i32 128
  %load.2 = load <32 x float>, ptr addrspace(3) %load.2.addr, align 128
  %load.3.addr = getelementptr <32 x float>, ptr addrspace(3) %load.2.addr, i32 192
  %L = load <1 x i64>, ptr addrspace(3) %load.3.addr, align 8
  %load.3 = load <32 x float>, ptr addrspace(3) %load.3.addr, align 128
  %load.4.addr = getelementptr <32 x float>, ptr addrspace(3) %load.3.addr, i32 256
  %L1 = load <1 x i64>, ptr addrspace(3) %load.4.addr, align 8
  %load.4 = load <32 x float>, ptr addrspace(3) %load.4.addr, align 128
  %B = urem <1 x i64> %L, %L1
  %mai.0 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.000000e+00, float 2.000000e+00, <32 x float> %load.0, i32 0, i32 0, i32 0)
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.000000e+00, float 2.000000e+00, <32 x float> %load.1, i32 0, i32 0, i32 0)
  %mai.2 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.000000e+00, float 2.000000e+00, <32 x float> %load.2, i32 0, i32 0, i32 0)
  %mai.3 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.000000e+00, float 2.000000e+00, <32 x float> %load.3, i32 0, i32 0, i32 0)
  %mai.4 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.000000e+00, float 2.000000e+00, <32 x float> %load.4, i32 0, i32 0, i32 0)
  %store.0.addr = getelementptr <32 x float>, ptr addrspace(3) %out, i32 %idx
  store <32 x float> %mai.0, ptr addrspace(3) %store.0.addr, align 128
  %store.1.addr = getelementptr <32 x float>, ptr addrspace(3) %out, i32 64
  store <32 x float> %mai.1, ptr addrspace(3) %store.1.addr, align 128
  %store.2.addr = getelementptr <32 x float>, ptr addrspace(3) %out, i32 128
  store <32 x float> %mai.2, ptr addrspace(3) %store.2.addr, align 128
  %store.3.addr = getelementptr <32 x float>, ptr addrspace(3) %out, i32 192
  store <32 x float> %mai.3, ptr addrspace(3) %store.3.addr, align 128
  %store.4.addr = getelementptr <32 x float>, ptr addrspace(3) %out, i32 256
  store <32 x float> %mai.4, ptr addrspace(3) %store.4.addr, align 128
  store <1 x i64> %B, ptr addrspace(3) %store.3.addr, align 8
  ret void
}

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.iglp.opt(i32 immarg) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.amdgcn.workitem.id.x() #2

; Function Attrs: convergent nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float, float, <32 x float>, i32 immarg, i32 immarg, i32 immarg) #3

attributes #0 = { nounwind "amdgpu-flat-work-group-size"="1,256" }
attributes #1 = { convergent nocallback nofree nounwind willreturn }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }
attributes #4 = { convergent nounwind }
