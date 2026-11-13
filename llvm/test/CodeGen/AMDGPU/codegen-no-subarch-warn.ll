; A target triple with no subarch still codegens, but is deprecated. Check that
; the warning fires and names the replacement triple derived from -mcpu.

; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -filetype=null %s 2>&1 | FileCheck %s
; RUN: llc -mtriple=amdgpu -mcpu=gfx900 -filetype=null %s 2>&1 | FileCheck %s

; Without -mcpu there is no subarch to suggest, so the warning omits a triple.
; RUN: llc -mtriple=amdgcn -filetype=null %s 2>&1 | FileCheck -check-prefix=NOCPU %s

; CHECK: warning: codegen with no subarch in the target triple is deprecated and will become an error; use the target triple 'amdgpu9.00--' instead{{$}}

; NOCPU: warning: codegen with no subarch in the target triple is deprecated and will become an error{{$}}

define amdgpu_kernel void @kernel() {
  ret void
}
