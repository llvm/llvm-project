; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -start-before=amdgpu-isel -stop-after=amdgpu-isel -verify-machineinstrs < %s | FileCheck %s --check-prefix=GCN

; We're really just checking for no crashes
; The feature we're testing for in AdjustWriteMask leaves the image_load as an instruction just post amdgpu-isel
; In reality, it's hard to get an image intrinsic into AdjustWriteMask with no uses as it will usually get removed
; first, but it can happen, hence the fix associated with this test

; GCN-LABEL: name: _amdgpu_cs_main
; GCN-LABEL: bb.0
; GCN: IMAGE_LOAD_V4_V2
define amdgpu_cs void @_amdgpu_cs_main(i32 %dummy) local_unnamed_addr #0 {
.entry:
  %unused.result = tail call <4 x float> @llvm.amdgcn.image.load.2d.v4f32.i32(i32 15, i32 undef, i32 undef, <8 x i32> undef, i32 0, i32 0) #3
  call void asm sideeffect ";", "" () #0
  ret void
}

; Function Attrs: nounwind readonly
declare <4 x float> @llvm.amdgcn.image.load.2d.v4f32.i32(i32, i32, i32, <8 x i32>, i32, i32) #1
  
attributes #0 = { nounwind }
attributes #1 = { nounwind readonly  }
