; RUN: llc -march=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}main:
; GCN: v_cvt_pkrtz_f16_f32
; GCN: v_cvt_pkrtz_f16_f32
; GCN-NOT: v_cvt_pkrtz_f16_f32

define amdgpu_ps void @main(float %src) {
main_body:
  %p1 = call i32 @llvm.SI.packf16(float undef, float %src)
  %p2 = call i32 @llvm.SI.packf16(float %src, float undef)
  %p3 = call i32 @llvm.SI.packf16(float undef, float undef)
  %f1 = bitcast i32 %p1 to float
  %f2 = bitcast i32 %p2 to float
  %f3 = bitcast i32 %p3 to float
  call void @llvm.SI.export(i32 15, i32 1, i32 0, i32 0, i32 1, float undef, float %f1, float undef, float %f1)
  call void @llvm.SI.export(i32 15, i32 1, i32 0, i32 0, i32 1, float undef, float %f2, float undef, float %f2)
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float undef, float %f3, float undef, float %f2)
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @llvm.SI.packf16(float, float) #0

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)

attributes #0 = { nounwind readnone }
