; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 -amdgpu-dump-hsa-metadata -amdgpu-verify-hsa-metadata -filetype=obj -o - < %s 2>&1 | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx802 -amdgpu-dump-hsa-metadata -amdgpu-verify-hsa-metadata -filetype=obj -o - < %s 2>&1 | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -amdgpu-dump-hsa-metadata -amdgpu-verify-hsa-metadata -filetype=obj -o - < %s 2>&1 | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 -amdgpu-dump-hsa-metadata -amdgpu-verify-hsa-metadata -filetype=obj -o - < %s 2>&1 |   FileCheck --check-prefix=PARSER %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx802 -amdgpu-dump-hsa-metadata -amdgpu-verify-hsa-metadata -filetype=obj -o - < %s 2>&1 |   FileCheck --check-prefix=PARSER %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -amdgpu-dump-hsa-metadata -amdgpu-verify-hsa-metadata -filetype=obj -o - < %s 2>&1 |   FileCheck --check-prefix=PARSER %s

@llvm.global_ctors = appending addrspace(1) global [2 x { i32, ptr, ptr  }] [{ i32, ptr, ptr  } { i32 1, ptr @foo, ptr null  }, { i32, ptr, ptr  } { i32 1, ptr @foo.5, ptr null  }]

define internal void @foo() {
      ret void

}

define internal void @foo.5() {
      ret void

}

; CHECK: ---
; CHECK: .kind: init
; CHECK: .name: amdgcn.device.init

@llvm.global_dtors = appending addrspace(1) global [2 x { i32, ptr, ptr  }] [{ i32, ptr, ptr  } { i32 1, ptr @bar, ptr null  }, { i32, ptr, ptr  } { i32 1, ptr @bar.5, ptr null  }]

define internal void @bar() {
      ret void

}

define internal void @bar.5() {
      ret void

}

; CHECK: .kind: fini
; CHECK: .name: amdgcn.device.fini

; PARSER: AMDGPU HSA Metadata Parser Test: PASS

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdgpu_code_object_version", i32 400}
