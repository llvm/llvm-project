; RUN: not --crash llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx600 -filetype=obj -filetype=null %s 2>&1 | FileCheck -check-prefix=ERROR %s
; RUN: not --crash llc -mtriple=amdgcn-mesa-mesa3d -mcpu=gfx600 -filetype=obj -filetype=null %s 2>&1 | FileCheck -check-prefix=ERROR %s

; RUN: llc -mtriple=amdgcn-amd-amdhsa -o - %s | FileCheck -check-prefix=HSA-DEFAULT %s
; RUN: not --crash llc -mtriple=amdgcn-mesa-mesa3d -mcpu=gfx600 -filetype=obj -filetype=null %s 2>&1 | FileCheck -check-prefix=ERROR %s

; Flat instructions should not select if the target device doesn't
; support them. The default device should be able to select for HSA.

; ERROR: LLVM ERROR: Cannot select: {{0x[0-9,a-f]+|t[0-9]+}}: i32,ch = load<(volatile load (s32) from %ir.flat.ptr.load)>
; HSA-DEFAULT: flat_load_dword
define amdgpu_kernel void @load_flat_i32(ptr %flat.ptr) {
  %load = load volatile i32, ptr %flat.ptr, align 4
  ret void
}
