; RUN: not llc -mtriple=amdgpu12.50 -global-isel=0 -stop-after=finalize-isel -verify-machineinstrs -o /dev/null %s 2>&1 | FileCheck %s
; RUN: not llc -mtriple=amdgpu12.51 -global-isel=0 -stop-after=finalize-isel -verify-machineinstrs -o /dev/null %s 2>&1 | FileCheck %s
; RUN: not llc -mtriple=amdgpu12.50 -global-isel=1 -global-isel-abort=1 -stop-after=legalizer -verify-machineinstrs -o /dev/null %s 2>&1 | FileCheck %s
; RUN: not llc -mtriple=amdgpu12.51 -global-isel=1 -global-isel-abort=1 -stop-after=legalizer -verify-machineinstrs -o /dev/null %s 2>&1 | FileCheck %s
; RUN: llc -mtriple=amdgpu11.00 -mattr=-nsa-encoding -global-isel=0 -stop-after=finalize-isel -verify-machineinstrs -o /dev/null %s
; RUN: llc -mtriple=amdgpu12.00 -global-isel=0 -stop-after=finalize-isel -verify-machineinstrs -o /dev/null %s
; RUN: llc -mtriple=amdgpu13.10 -global-isel=0 -stop-after=finalize-isel -verify-machineinstrs -o /dev/null %s
; RUN: llc -mtriple=amdgpu12.00 -global-isel=1 -global-isel-abort=1 -stop-after=legalizer -verify-machineinstrs -o /dev/null %s
; RUN: llvm-extract --rfunc='^noop_' -S %s -o %t.noop.ll
; RUN: llc -mtriple=amdgpu12.50 -global-isel=0 -stop-after=finalize-isel -verify-machineinstrs %t.noop.ll -o - | FileCheck --check-prefix=NOOP %s
; RUN: llc -mtriple=amdgpu12.51 -global-isel=0 -stop-after=finalize-isel -verify-machineinstrs %t.noop.ll -o - | FileCheck --check-prefix=NOOP %s
; RUN: llc -mtriple=amdgpu12.50 -global-isel=1 -global-isel-abort=1 -stop-after=legalizer -verify-machineinstrs %t.noop.ll -o - | FileCheck --check-prefix=NOOP %s
; RUN: llc -mtriple=amdgpu12.51 -global-isel=1 -global-isel-abort=1 -stop-after=legalizer -verify-machineinstrs %t.noop.ll -o - | FileCheck --check-prefix=NOOP %s

; CHECK: error: {{.*}}in function store_2d {{.*}}requested image instruction is not supported on this GPU
define void @store_2d(<8 x i32> %rsrc, i32 %x, i32 %y, <4 x float> %data) {
  call void @llvm.amdgcn.image.store.2d.v4f32.i32(<4 x float> %data, i32 15, i32 %x, i32 %y, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; CHECK: error: {{.*}}in function store_2d_image_insts {{.*}}requested image instruction is not supported on this GPU
define void @store_2d_image_insts(<8 x i32> %rsrc, i32 %x, i32 %y, <4 x float> %data) #0 {
  call void @llvm.amdgcn.image.store.2d.v4f32.i32(<4 x float> %data, i32 15, i32 %x, i32 %y, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; CHECK: error: {{.*}}in function load_2d {{.*}}requested image instruction is not supported on this GPU
define <4 x float> @load_2d(<8 x i32> %rsrc, i32 %x, i32 %y) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.load.2d.v4f32.i32(i32 15, i32 %x, i32 %y, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %data
}

; CHECK: error: {{.*}}in function getresinfo {{.*}}requested image instruction is not supported on this GPU
define <4 x float> @getresinfo(<8 x i32> %rsrc, i32 %mip) {
  %data = call <4 x float> @llvm.amdgcn.image.getresinfo.1d.v4f32.i32(i32 15, i32 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %data
}

; CHECK: error: {{.*}}in function atomic_add {{.*}}requested image instruction is not supported on this GPU
define i32 @atomic_add(<8 x i32> %rsrc, i32 %x, i32 %data) #0 {
  %old = call i32 @llvm.amdgcn.image.atomic.add.1d.i32.i32(i32 %data, i32 %x, <8 x i32> %rsrc, i32 0, i32 0)
  ret i32 %old
}

; CHECK: error: {{.*}}in function atomic_add_noret {{.*}}requested image instruction is not supported on this GPU
define void @atomic_add_noret(<8 x i32> %rsrc, i32 %x, i32 %data) #0 {
  %old = call i32 @llvm.amdgcn.image.atomic.add.1d.i32.i32(i32 %data, i32 %x, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; CHECK: error: {{.*}}in function atomic_cmpswap_64 {{.*}}requested image instruction is not supported on this GPU
define i64 @atomic_cmpswap_64(<8 x i32> %rsrc, i32 %x, i64 %compare, i64 %replacement) #0 {
  %old = call i64 @llvm.amdgcn.image.atomic.cmpswap.1d.i64.i32(i64 %compare, i64 %replacement, i32 %x, <8 x i32> %rsrc, i32 0, i32 0)
  ret i64 %old
}

; CHECK: error: {{.*}}in function store_zero_dmask {{.*}}requested image instruction is not supported on this GPU
define void @store_zero_dmask(<8 x i32> %rsrc, i32 %x, i32 %y, <4 x float> %data) #0 {
  call void @llvm.amdgcn.image.store.2d.v4f32.i32(<4 x float> %data, i32 0, i32 %x, i32 %y, <8 x i32> %rsrc, i32 0, i32 0)
  ret void
}

; CHECK: error: {{.*}}in function load_zero_dmask_tfe {{.*}}requested image instruction is not supported on this GPU
define i32 @load_zero_dmask_tfe(<8 x i32> %rsrc, i32 %x) #0 {
  %result = call { <4 x float>, i32 } @llvm.amdgcn.image.load.1d.v4f32i32.i32(i32 0, i32 %x, <8 x i32> %rsrc, i32 1, i32 0)
  %status = extractvalue { <4 x float>, i32 } %result, 1
  ret i32 %status
}

; CHECK: error: {{.*}}in function load_zero_dmask_lwe {{.*}}requested image instruction is not supported on this GPU
define i32 @load_zero_dmask_lwe(<8 x i32> %rsrc, i32 %x) #0 {
  %result = call { <4 x float>, i32 } @llvm.amdgcn.image.load.1d.v4f32i32.i32(i32 0, i32 %x, <8 x i32> %rsrc, i32 2, i32 0)
  %status = extractvalue { <4 x float>, i32 } %result, 1
  ret i32 %status
}

; CHECK: error: {{.*}}in function sample_zero_dmask_tfe {{.*}}requested image instruction is not supported on this GPU
define i32 @sample_zero_dmask_tfe(<8 x i32> %rsrc, <4 x i32> %sampler, float %x) #0 {
  %result = call { <4 x float>, i32 } @llvm.amdgcn.image.sample.1d.v4f32i32.f32(i32 0, float %x, <8 x i32> %rsrc, <4 x i32> %sampler, i1 false, i32 1, i32 0)
  %status = extractvalue { <4 x float>, i32 } %result, 1
  ret i32 %status
}

; NOOP-LABEL: name: noop_load
; NOOP: body:
; NOOP-NOT: {{[Ii][Mm][Aa][Gg][Ee]}}
; NOOP: ...
define <4 x float> @noop_load(<8 x i32> %rsrc, i32 %x) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.load.1d.v4f32.i32(i32 0, i32 %x, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %data
}

; NOOP-LABEL: name: noop_sample
; NOOP: body:
; NOOP-NOT: {{[Ii][Mm][Aa][Gg][Ee]}}
; NOOP: ...
define <4 x float> @noop_sample(<8 x i32> %rsrc, <4 x i32> %sampler, float %x) #0 {
  %data = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 0, float %x, <8 x i32> %rsrc, <4 x i32> %sampler, i1 false, i32 0, i32 0)
  ret <4 x float> %data
}

; NOOP-LABEL: name: noop_getresinfo
; NOOP: body:
; NOOP-NOT: {{[Ii][Mm][Aa][Gg][Ee]}}
; NOOP: ...
define <4 x float> @noop_getresinfo(<8 x i32> %rsrc, i32 %mip) {
  %data = call <4 x float> @llvm.amdgcn.image.getresinfo.1d.v4f32.i32(i32 0, i32 %mip, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %data
}

declare void @llvm.amdgcn.image.store.2d.v4f32.i32(<4 x float>, i32 immarg, i32, i32, <8 x i32>, i32 immarg, i32 immarg)
declare <4 x float> @llvm.amdgcn.image.load.2d.v4f32.i32(i32 immarg, i32, i32, <8 x i32>, i32 immarg, i32 immarg)
declare i32 @llvm.amdgcn.image.atomic.add.1d.i32.i32(i32, i32, <8 x i32>, i32 immarg, i32 immarg)
declare i64 @llvm.amdgcn.image.atomic.cmpswap.1d.i64.i32(i64, i64, i32, <8 x i32>, i32 immarg, i32 immarg)
declare <4 x float> @llvm.amdgcn.image.load.1d.v4f32.i32(i32 immarg, i32, <8 x i32>, i32 immarg, i32 immarg)
declare { <4 x float>, i32 } @llvm.amdgcn.image.load.1d.v4f32i32.i32(i32 immarg, i32, <8 x i32>, i32 immarg, i32 immarg)
declare <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 immarg, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg)
declare { <4 x float>, i32 } @llvm.amdgcn.image.sample.1d.v4f32i32.f32(i32 immarg, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg)
declare <4 x float> @llvm.amdgcn.image.getresinfo.1d.v4f32.i32(i32 immarg, i32, <8 x i32>, i32 immarg, i32 immarg)

attributes #0 = { "target-features"="+image-insts" }