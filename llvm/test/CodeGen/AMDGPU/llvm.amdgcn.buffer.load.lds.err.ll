; RUN: split-file %s %t
;
; RUN: not llc -global-isel=0 -mtriple=amdgpu12.50 -filetype=null < %t/struct.ll 2>&1 | FileCheck --check-prefix=STRUCT %s
; RUN: not llc -global-isel=1 -mtriple=amdgpu12.50 -filetype=null < %t/struct.ll 2>&1 | FileCheck --check-prefix=STRUCT %s
; RUN: not llc -global-isel=0 -mtriple=amdgpu12.50 -filetype=null < %t/struct.async.ll 2>&1 | FileCheck --check-prefix=STRUCT-ASYNC %s
; RUN: not llc -global-isel=1 -mtriple=amdgpu12.50 -filetype=null < %t/struct.async.ll 2>&1 | FileCheck --check-prefix=STRUCT-ASYNC %s
; RUN: not llc -global-isel=0 -mtriple=amdgpu12.50 -filetype=null < %t/struct.ptr.ll 2>&1 | FileCheck --check-prefix=STRUCT-PTR %s
; RUN: not llc -global-isel=1 -mtriple=amdgpu12.50 -filetype=null < %t/struct.ptr.ll 2>&1 | FileCheck --check-prefix=STRUCT-PTR %s
; RUN: not llc -global-isel=0 -mtriple=amdgpu12.50 -filetype=null < %t/struct.ptr.async.ll 2>&1 | FileCheck --check-prefix=STRUCT-PTR-ASYNC %s
; RUN: not llc -global-isel=1 -mtriple=amdgpu12.50 -filetype=null < %t/struct.ptr.async.ll 2>&1 | FileCheck --check-prefix=STRUCT-PTR-ASYNC %s
; RUN: not llc -global-isel=0 -mtriple=amdgpu12.50 -filetype=null < %t/raw.ll 2>&1 | FileCheck --check-prefix=RAW %s
; RUN: not llc -global-isel=1 -mtriple=amdgpu12.50 -filetype=null < %t/raw.ll 2>&1 | FileCheck --check-prefix=RAW %s
; RUN: not llc -global-isel=0 -mtriple=amdgpu12.50 -filetype=null < %t/raw.async.ll 2>&1 | FileCheck --check-prefix=RAW-ASYNC %s
; RUN: not llc -global-isel=1 -mtriple=amdgpu12.50 -filetype=null < %t/raw.async.ll 2>&1 | FileCheck --check-prefix=RAW-ASYNC %s
; RUN: not llc -global-isel=0 -mtriple=amdgpu12.50 -filetype=null < %t/raw.ptr.ll 2>&1 | FileCheck --check-prefix=RAW-PTR %s
; RUN: not llc -global-isel=1 -mtriple=amdgpu12.50 -filetype=null < %t/raw.ptr.ll 2>&1 | FileCheck --check-prefix=RAW-PTR %s
; RUN: not llc -global-isel=0 -mtriple=amdgpu12.50 -filetype=null < %t/raw.ptr.async.ll 2>&1 | FileCheck --check-prefix=RAW-PTR-ASYNC %s
; RUN: not llc -global-isel=1 -mtriple=amdgpu12.50 -filetype=null < %t/raw.ptr.async.ll 2>&1 | FileCheck --check-prefix=RAW-PTR-ASYNC %s
;
; STRUCT: llvm.amdgcn.struct.buffer.load.lds requires target feature 'vmem-to-lds-load-insts'
; STRUCT-ASYNC: llvm.amdgcn.struct.buffer.load.async.lds requires target feature 'vmem-to-lds-load-insts'
; STRUCT-PTR: llvm.amdgcn.struct.ptr.buffer.load.lds requires target feature 'vmem-to-lds-load-insts'
; STRUCT-PTR-ASYNC: llvm.amdgcn.struct.ptr.buffer.load.async.lds requires target feature 'vmem-to-lds-load-insts'
; RAW: llvm.amdgcn.raw.buffer.load.lds requires target feature 'vmem-to-lds-load-insts'
; RAW-ASYNC: llvm.amdgcn.raw.buffer.load.async.lds requires target feature 'vmem-to-lds-load-insts'
; RAW-PTR: llvm.amdgcn.raw.ptr.buffer.load.lds requires target feature 'vmem-to-lds-load-insts'
; RAW-PTR-ASYNC: llvm.amdgcn.raw.ptr.buffer.load.async.lds requires target feature 'vmem-to-lds-load-insts'

;--- struct.ll
define amdgpu_ps void @buffer_load_lds(<4 x i32> inreg %rsrc, ptr addrspace(3) inreg %lds) {
  call void @llvm.amdgcn.struct.buffer.load.lds(<4 x i32> %rsrc, ptr addrspace(3) %lds, i32 4, i32 0, i32 0, i32 0, i32 0, i32 0)
  ret void
}

;--- struct.async.ll
define amdgpu_ps void @buffer_load_lds(<4 x i32> inreg %rsrc, ptr addrspace(3) inreg %lds) {
  call void @llvm.amdgcn.struct.buffer.load.async.lds(<4 x i32> %rsrc, ptr addrspace(3) %lds, i32 4, i32 0, i32 0, i32 0, i32 0, i32 0)
  ret void
}

;--- struct.ptr.ll
define amdgpu_ps void @buffer_load_lds(ptr addrspace(8) inreg %rsrc, ptr addrspace(3) inreg %lds) {
  call void @llvm.amdgcn.struct.ptr.buffer.load.lds(ptr addrspace(8) %rsrc, ptr addrspace(3) %lds, i32 4, i32 0, i32 0, i32 0, i32 0, i32 0)
  ret void
}

;--- struct.ptr.async.ll
define amdgpu_ps void @buffer_load_lds(ptr addrspace(8) inreg %rsrc, ptr addrspace(3) inreg %lds) {
  call void @llvm.amdgcn.struct.ptr.buffer.load.async.lds(ptr addrspace(8) %rsrc, ptr addrspace(3) %lds, i32 4, i32 0, i32 0, i32 0, i32 0, i32 0)
  ret void
}

;--- raw.ll
define amdgpu_ps void @buffer_load_lds(<4 x i32> inreg %rsrc, ptr addrspace(3) inreg %lds) {
  call void @llvm.amdgcn.raw.buffer.load.lds(<4 x i32> %rsrc, ptr addrspace(3) %lds, i32 4, i32 0, i32 0, i32 0, i32 0)
  ret void
}

;--- raw.async.ll
define amdgpu_ps void @buffer_load_lds(<4 x i32> inreg %rsrc, ptr addrspace(3) inreg %lds) {
  call void @llvm.amdgcn.raw.buffer.load.async.lds(<4 x i32> %rsrc, ptr addrspace(3) %lds, i32 4, i32 0, i32 0, i32 0, i32 0)
  ret void
}

;--- raw.ptr.ll
define amdgpu_ps void @buffer_load_lds(ptr addrspace(8) inreg %rsrc, ptr addrspace(3) inreg %lds) {
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %rsrc, ptr addrspace(3) %lds, i32 4, i32 0, i32 0, i32 0, i32 0)
  ret void
}

;--- raw.ptr.async.ll
define amdgpu_ps void @buffer_load_lds(ptr addrspace(8) inreg %rsrc, ptr addrspace(3) inreg %lds) {
  call void @llvm.amdgcn.raw.ptr.buffer.load.async.lds(ptr addrspace(8) %rsrc, ptr addrspace(3) %lds, i32 4, i32 0, i32 0, i32 0, i32 0)
  ret void
}
