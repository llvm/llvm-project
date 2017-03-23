; RUN: llc -debug-only=misched -march=amdgcn -verify-machineinstrs %s -o - 2>&1| FileCheck -check-prefix=SI-NOHSA -check-prefix=FUNC -check-prefix=DEBUG %s
; REQUIRES: asserts

; Verify that the extload generated from %eval has the default
; alignment size (2) corresponding to the underlying memory size (i16)
; size and not 4 corresponding to the sign-extended size (i32).

; DEBUG: {{^}}# Machine code for function extload_align:
; DEBUG: mem:LD2[<unknown>]{{[^(]}}
; DEBUG: {{^}}# End machine code for function extload_align.

define amdgpu_kernel void @extload_align(i32* %out, i32 %index) #0 {
  %v0 = alloca [4 x i16]
  %a1 = getelementptr inbounds [4 x i16], [4 x i16]* %v0, i32 0, i32 0
  %a2 = getelementptr inbounds [4 x i16], [4 x i16]* %v0, i32 0, i32 1
  store i16 0, i16* %a1
  store i16 1, i16* %a2
  %a = getelementptr inbounds [4 x i16], [4 x i16]* %v0, i32 0, i32 %index
  %val = load i16, i16* %a
  %eval = sext i16 %val to i32
  store i32 %eval, i32* %out
  ret void
}