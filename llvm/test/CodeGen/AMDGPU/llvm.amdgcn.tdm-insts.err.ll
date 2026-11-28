; RUN: not llc -global-isel=0 -mtriple=amdgpu9.42 -filetype=null %s 2>&1 | FileCheck %s
; RUN: not llc -global-isel=1 -mtriple=amdgpu9.42 -filetype=null %s 2>&1 | FileCheck %s
; RUN: not llc -global-isel=0 -mtriple=amdgpu12.00 -filetype=null %s 2>&1 | FileCheck %s
; RUN: not llc -global-isel=1 -mtriple=amdgpu12.00 -filetype=null %s 2>&1 | FileCheck %s
; RUN: not llc -global-isel=0 -mtriple=amdgpu13.10 -filetype=null %s 2>&1 | FileCheck %s
; RUN: not llc -global-isel=1 -mtriple=amdgpu13.10 -filetype=null %s 2>&1 | FileCheck %s

; CHECK: error: {{.*}}in function @tensor_load_to_lds {{.*}}: llvm.amdgcn.tensor.load.to.lds requires target feature 'tdm-insts'
define amdgpu_ps void @tensor_load_to_lds(<4 x i32> inreg %D0, <8 x i32> inreg %D1, <4 x i32> inreg %D2, <4 x i32> inreg %D3) {
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %D0, <8 x i32> %D1, <4 x i32> %D2, <4 x i32> %D3, <8 x i32> zeroinitializer, i32 0)
  ret void
}

; CHECK: error: {{.*}}in function @tensor_store_from_lds {{.*}}: llvm.amdgcn.tensor.store.from.lds requires target feature 'tdm-insts'
define amdgpu_ps void @tensor_store_from_lds(<4 x i32> inreg %D0, <8 x i32> inreg %D1, <4 x i32> inreg %D2, <4 x i32> inreg %D3) {
  call void @llvm.amdgcn.tensor.store.from.lds(<4 x i32> %D0, <8 x i32> %D1, <4 x i32> %D2, <4 x i32> %D3, <8 x i32> zeroinitializer, i32 0)
  ret void
}

declare void @llvm.amdgcn.tensor.load.to.lds(<4 x i32>, <8 x i32>, <4 x i32>, <4 x i32>, <8 x i32>, i32 immarg)
declare void @llvm.amdgcn.tensor.store.from.lds(<4 x i32>, <8 x i32>, <4 x i32>, <4 x i32>, <8 x i32>, i32 immarg)
