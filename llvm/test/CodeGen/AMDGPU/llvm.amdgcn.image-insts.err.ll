; RUN: not llc -global-isel=0 -mtriple=amdgpu9.42-amd-amdhsa -filetype=null %s 2>&1 | FileCheck --implicit-check-not="LLVM ERROR" %s
; RUN: not llc -global-isel=1 -mtriple=amdgpu9.42-amd-amdhsa -filetype=null %s 2>&1 | FileCheck --implicit-check-not="LLVM ERROR" %s
; RUN: not llc -global-isel=0 -mtriple=amdgpu12.50-amd-amdhsa -filetype=null %s 2>&1 | FileCheck --implicit-check-not="LLVM ERROR" %s
; RUN: not llc -global-isel=1 -mtriple=amdgpu12.50-amd-amdhsa -filetype=null %s 2>&1 | FileCheck --implicit-check-not="LLVM ERROR" %s

; CHECK: error: {{.*}}in function @store_2d {{.*}}: llvm.amdgcn.image.store.2d requires target feature 'image-insts'
define void @store_2d(<8 x i32> inreg %rsrc, i32 %x, i32 %y, <4 x float> %data) {
  call void @llvm.amdgcn.image.store.2d.v4f32.i32(<4 x float> %data, i32 15, i32 %x, i32 %y, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; CHECK: error: {{.*}}in function @load_2d {{.*}}: llvm.amdgcn.image.load.2d requires target feature 'image-insts'
define <4 x float> @load_2d(<8 x i32> inreg %rsrc, i32 %x, i32 %y) {
  %v = call <4 x float> @llvm.amdgcn.image.load.2d.v4f32.i32(i32 15, i32 %x, i32 %y, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; A zero dmask makes the load a no-op, but it is still rejected.
; CHECK: error: {{.*}}in function @load_2d_dmask0 {{.*}}: llvm.amdgcn.image.load.2d requires target feature 'image-insts'
define <4 x float> @load_2d_dmask0(<8 x i32> inreg %rsrc, i32 %x, i32 %y) {
  %v = call <4 x float> @llvm.amdgcn.image.load.2d.v4f32.i32(i32 0, i32 %x, i32 %y, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

; CHECK: error: {{.*}}in function @sample_2d {{.*}}: llvm.amdgcn.image.sample.2d requires target feature 'image-insts'
define <4 x float> @sample_2d(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, float %s, float %t) {
  %v = call <4 x float> @llvm.amdgcn.image.sample.2d.v4f32.f32(i32 15, float %s, float %t, <8 x i32> %rsrc, <4 x i32> %samp, i1 false, i32 0, i32 0)
  ret <4 x float> %v
}

; CHECK: error: {{.*}}in function @atomic_add_1d {{.*}}: llvm.amdgcn.image.atomic.add.1d requires target feature 'image-insts'
define i32 @atomic_add_1d(<8 x i32> inreg %rsrc, i32 %x, i32 %data) {
  %old = call i32 @llvm.amdgcn.image.atomic.add.1d.i32.i32(i32 %data, i32 %x, <8 x i32> %rsrc, i32 0, i32 0)
  ret i32 %old
}

; CHECK: error: {{.*}}in function @getresinfo_1d {{.*}}: llvm.amdgcn.image.getresinfo.1d requires target feature 'image-insts'
define <4 x float> @getresinfo_1d(<8 x i32> inreg %rsrc, i32 %mip) {
  %v = call <4 x float> @llvm.amdgcn.image.getresinfo.1d.v4f32.i32(i32 15, i32 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %v
}

declare void @llvm.amdgcn.image.store.2d.v4f32.i32(<4 x float>, i32 immarg, i32, i32, <8 x i32>, i32 immarg, i32 immarg)
declare <4 x float> @llvm.amdgcn.image.load.2d.v4f32.i32(i32 immarg, i32, i32, <8 x i32>, i32 immarg, i32 immarg)
declare <4 x float> @llvm.amdgcn.image.sample.2d.v4f32.f32(i32 immarg, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg)
declare i32 @llvm.amdgcn.image.atomic.add.1d.i32.i32(i32, i32, <8 x i32>, i32 immarg, i32 immarg)
declare <4 x float> @llvm.amdgcn.image.getresinfo.1d.v4f32.i32(i32 immarg, i32, <8 x i32>, i32 immarg, i32 immarg)
