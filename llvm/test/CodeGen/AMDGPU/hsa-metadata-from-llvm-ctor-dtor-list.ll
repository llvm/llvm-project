; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 --amdhsa-code-object-version=3 -amdgpu-dump-hsa-metadata -amdgpu-verify-hsa-metadata -filetype=obj -o - < %s 2>&1 | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx802 --amdhsa-code-object-version=3 -amdgpu-dump-hsa-metadata -amdgpu-verify-hsa-metadata -filetype=obj -o - < %s 2>&1 | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 --amdhsa-code-object-version=3 -amdgpu-dump-hsa-metadata -amdgpu-verify-hsa-metadata -filetype=obj -o - < %s 2>&1 | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 --amdhsa-code-object-version=3 -amdgpu-dump-hsa-metadata -amdgpu-verify-hsa-metadata -filetype=obj -o - < %s 2>&1 |   FileCheck --check-prefix=PARSER %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx802 --amdhsa-code-object-version=3 -amdgpu-dump-hsa-metadata -amdgpu-verify-hsa-metadata -filetype=obj -o - < %s 2>&1 |   FileCheck --check-prefix=PARSER %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 --amdhsa-code-object-version=3 -amdgpu-dump-hsa-metadata -amdgpu-verify-hsa-metadata -filetype=obj -o - < %s 2>&1 |   FileCheck --check-prefix=PARSER %s

@llvm.global_ctors = appending addrspace(1) global [2 x { i32, void ()*, i8*  }] [{ i32, void ()*, i8*  } { i32 1, void ()* @foo, i8* null  }, { i32, void ()*, i8*  } { i32 1, void ()* @foo.5, i8* null  }]

define internal void @foo() {
      ret void
      
}

define internal void @foo.5() {
      ret void
      
}

; CHECK: ---
; CHECK: .kind: init
; CHECK: .name: amdgcn.device.init

@llvm.global_dtors = appending addrspace(1) global [2 x { i32, void ()*, i8*  }] [{ i32, void ()*, i8*  } { i32 1, void ()* @bar, i8* null  }, { i32, void ()*, i8*  } { i32 1, void ()* @bar.5, i8* null  }]

define internal void @bar() {
      ret void
      
}

define internal void @bar.5() {
      ret void
      
}

; CHECK: .kind: fini
; CHECK: .name: amdgcn.device.fini

; PARSER: AMDGPU HSA Metadata Parser Test: PASS
