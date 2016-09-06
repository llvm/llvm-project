; RUN: llc < %s -march=amdgcn -mcpu=SI -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck %s

@lds = external addrspace(3) global [64 x float]

; CHECK-LABEL: {{^}}main:
; CHECK-NOT: v_readlane_b32 m0
define amdgpu_ps void @main(<16 x i8> addrspace(2)* inreg, <16 x i8> addrspace(2)* inreg, <32 x i8> addrspace(2)* inreg, i32 inreg) {
main_body:
  %4 = call float @llvm.SI.fs.constant(i32 0, i32 0, i32 %3)
  %cmp = fcmp ueq float 0.0, %4
  br i1 %cmp, label %if, label %else

if:
  %lds_ptr = getelementptr [64 x float], [64 x float] addrspace(3)* @lds, i32 0, i32 0
  %lds_data = load float, float addrspace(3)* %lds_ptr
  br label %endif

else:
  %interp = call float @llvm.SI.fs.constant(i32 0, i32 0, i32 %3)
  br label %endif

endif:
  %export = phi float [%lds_data, %if], [%interp, %else]
  %5 = call i32 @llvm.SI.packf16(float %export, float %export)
  %6 = bitcast i32 %5 to float
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %6, float %6, float %6, float %6)
  ret void
}

declare float @llvm.SI.fs.constant(i32, i32, i32) readnone

declare i32 @llvm.SI.packf16(float, float) readnone

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)
