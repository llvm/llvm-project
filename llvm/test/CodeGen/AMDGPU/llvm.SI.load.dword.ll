; RUN: llc -march=amdgcn -mcpu=verde -show-mc-encoding -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -march=amdgcn -mcpu=tonga -show-mc-encoding -verify-machineinstrs < %s | FileCheck %s

; Example of a simple geometry shader loading vertex attributes from the
; ESGS ring buffer

; FIXME: Out of bounds immediate offset crashes

; CHECK-LABEL: {{^}}main:
; CHECK: buffer_load_dword {{v[0-9]+}}, off, {{s\[[0-9]+:[0-9]+\]}}, 0 glc slc
; CHECK: buffer_load_dword {{v[0-9]+}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, 0 offen glc slc
; CHECK: buffer_load_dword {{v[0-9]+}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, 0 idxen glc slc
; CHECK: buffer_load_dword {{v[0-9]+}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 idxen offen glc slc
; CHECK: s_movk_i32 [[K:s[0-9]+]], 0x4d2 ; encoding
; CHECK: buffer_load_dword {{v[0-9]+}}, {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, [[K]] idxen offen offset:65535 glc slc

define amdgpu_vs void @main([17 x <16 x i8>] addrspace(2)* byval %arg, [32 x <16 x i8>] addrspace(2)* byval %arg1, [16 x <32 x i8>] addrspace(2)* byval %arg2, [2 x <16 x i8>] addrspace(2)* byval %arg3, [17 x <16 x i8>] addrspace(2)* inreg %arg4, [17 x <16 x i8>] addrspace(2)* inreg %arg5, i32 %arg6, i32 %arg7, i32 %arg8, i32 %arg9) {
main_body:
  %tmp = getelementptr [2 x <16 x i8>], [2 x <16 x i8>] addrspace(2)* %arg3, i64 0, i32 1
  %tmp10 = load <16 x i8>, <16 x i8> addrspace(2)* %tmp, !tbaa !0
  %tmp11 = shl i32 %arg6, 2
  %tmp12 = call i32 @llvm.SI.buffer.load.dword.i32.i32(<16 x i8> %tmp10, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 0)
  %tmp13 = bitcast i32 %tmp12 to float
  %tmp14 = call i32 @llvm.SI.buffer.load.dword.i32.i32(<16 x i8> %tmp10, i32 %tmp11, i32 0, i32 0, i32 1, i32 0, i32 1, i32 1, i32 0)
  %tmp15 = bitcast i32 %tmp14 to float
  %tmp16 = call i32 @llvm.SI.buffer.load.dword.i32.i32(<16 x i8> %tmp10, i32 %tmp11, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 0)
  %tmp17 = bitcast i32 %tmp16 to float
  %tmp18 = call i32 @llvm.SI.buffer.load.dword.i32.v2i32(<16 x i8> %tmp10, <2 x i32> zeroinitializer, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1, i32 0)
  %tmp19 = bitcast i32 %tmp18 to float

  %tmp20 = call i32 @llvm.SI.buffer.load.dword.i32.v2i32(<16 x i8> %tmp10, <2 x i32> zeroinitializer, i32 0, i32 123, i32 1, i32 1, i32 1, i32 1, i32 0)
  %tmp21 = bitcast i32 %tmp20 to float

  %tmp22 = call i32 @llvm.SI.buffer.load.dword.i32.v2i32(<16 x i8> %tmp10, <2 x i32> zeroinitializer, i32 1234, i32 65535, i32 1, i32 1, i32 1, i32 1, i32 0)
  %tmp23 = bitcast i32 %tmp22 to float

  call void @llvm.amdgcn.exp.f32(i32 15, i32 12, float %tmp13, float %tmp15, float %tmp17, float %tmp19, i1 false, i1 false)
  call void @llvm.amdgcn.exp.f32(i32 15, i32 12, float %tmp21, float %tmp23, float %tmp23, float %tmp23, i1 true, i1 false)
  ret void
}

; Function Attrs: nounwind readonly
declare i32 @llvm.SI.buffer.load.dword.i32.i32(<16 x i8>, i32, i32, i32, i32, i32, i32, i32, i32) #0

; Function Attrs: nounwind readonly
declare i32 @llvm.SI.buffer.load.dword.i32.v2i32(<16 x i8>, <2 x i32>, i32, i32, i32, i32, i32, i32, i32) #0

declare void @llvm.amdgcn.exp.f32(i32, i32, float, float, float, float, i1, i1) #1

attributes #0 = { nounwind readonly }
attributes #1 = { nounwind inaccessiblememonly }

!0 = !{!"const", !1, i32 1}
!1 = !{!"tbaa root"}
