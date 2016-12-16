; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; This should end with an no-op sequence of exec mask manipulations
; Mask should be in original state after executed unreachable block

; GCN-LABEL: {{^}}main:
; GCN: s_cbranch_scc1 [[RET_BB:BB[0-9]+_[0-9]+]]

; GCN: s_and_saveexec_b64 [[SAVE_EXEC:s\[[0-9]+:[0-9]+\]]], vcc
; GCN-NEXT: s_xor_b64 [[XOR_EXEC:s\[[0-9]+:[0-9]+\]]], exec, [[SAVE_EXEC]]
; GCN-NEXT: ; mask branch [[UNREACHABLE_BB:BB[0-9]+_[0-9]+]]

; GCN: [[RET_BB]]:
; GCN-NEXT: s_branch [[FINAL_BB:BB[0-9]+_[0-9]+]]

; GCN-NEXT: [[UNREACHABLE_BB]]:
; GCN-NEXT: s_or_b64 exec, exec, [[XOR_EXEC]]
; GCN-NEXT: [[FINAL_BB]]:
; GCN-NEXT: .Lfunc_end0
define amdgpu_ps <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, float, float, float, float, float, float, float, float, float, float, float, float, float, float }> @main([9 x <16 x i8>] addrspace(2)* byval, [17 x <16 x i8>] addrspace(2)* byval, [17 x <8 x i32>] addrspace(2)* byval, i32 addrspace(2)* byval, float inreg, i32 inreg, <2 x i32>, <2 x i32>, <2 x i32>, <3 x i32>, <2 x i32>, <2 x i32>, <2 x i32>, float, float, float, float, float, i32, i32, float, i32) #0 {
main_body:
  %p83 = call float @llvm.SI.fs.interp(i32 1, i32 0, i32 %5, <2 x i32> %7)
  %p87 = fmul float undef, %p83
  %p88 = fadd float %p87, undef
  %p93 = fadd float %p88, undef
  %p97 = fmul float %p93, undef
  %p102 = fsub float %p97, undef
  %p104 = fmul float %p102, undef
  %p106 = fadd float 0.000000e+00, %p104
  %p108 = fadd float undef, %p106
  br i1 undef, label %ENDIF69, label %ELSE

ELSE:                                             ; preds = %main_body
  %p124 = fmul float %p108, %p108
  %p125 = fsub float %p124, undef
  %p126 = fcmp olt float %p125, 0.000000e+00
  br i1 %p126, label %ENDIF69, label %ELSE41

ELSE41:                                           ; preds = %ELSE
  unreachable

ENDIF69:                                          ; preds = %ELSE, %main_body
  ret <{ i32, i32, i32, i32, i32, i32, i32, i32, i32, float, float, float, float, float, float, float, float, float, float, float, float, float, float }> undef
}

; Function Attrs: nounwind readnone
declare float @llvm.SI.load.const(<16 x i8>, i32) #1

; Function Attrs: nounwind readnone
declare float @llvm.SI.fs.interp(i32, i32, i32, <2 x i32>) #1

; Function Attrs: nounwind readnone
declare float @llvm.fabs.f32(float) #1

; Function Attrs: nounwind readnone
declare float @llvm.sqrt.f32(float) #1

; Function Attrs: nounwind readnone
declare float @llvm.floor.f32(float) #1

attributes #0 = { "InitialPSInputAddr"="36983" }
attributes #1 = { nounwind readnone }
