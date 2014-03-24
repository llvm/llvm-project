;RUN: llc < %s -march=r600 -mcpu=verde -verify-machineinstrs | FileCheck %s

; Example of a simple geometry shader loading vertex attributes from the
; ESGS ring buffer

; CHECK-LABEL: @main
; CHECK: BUFFER_LOAD_DWORD
; CHECK: BUFFER_LOAD_DWORD
; CHECK: BUFFER_LOAD_DWORD
; CHECK: BUFFER_LOAD_DWORD

define void @main([17 x <16 x i8>] addrspace(2)* byval, [32 x <16 x i8>] addrspace(2)* byval, [16 x <32 x i8>] addrspace(2)* byval, [2 x <16 x i8>] addrspace(2)* byval, [17 x <16 x i8>] addrspace(2)* inreg, [17 x <16 x i8>] addrspace(2)* inreg, i32, i32, i32, i32) #0 {
main_body:
  %10 = getelementptr [2 x <16 x i8>] addrspace(2)* %3, i64 0, i32 1
  %11 = load <16 x i8> addrspace(2)* %10, !tbaa !0
  %12 = shl i32 %6, 2
  %13 = call i32 @llvm.SI.buffer.load.dword.i32.i32(<16 x i8> %11, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 0)
  %14 = bitcast i32 %13 to float
  %15 = call i32 @llvm.SI.buffer.load.dword.i32.i32(<16 x i8> %11, i32 %12, i32 0, i32 0, i32 1, i32 0, i32 1, i32 1, i32 0)
  %16 = bitcast i32 %15 to float
  %17 = call i32 @llvm.SI.buffer.load.dword.i32.i32(<16 x i8> %11, i32 %12, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 0)
  %18 = bitcast i32 %17 to float
  %19 = call i32 @llvm.SI.buffer.load.dword.i32.v2i32(<16 x i8> %11, <2 x i32> <i32 0, i32 0>, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1, i32 0)
  %20 = bitcast i32 %19 to float
  call void @llvm.SI.export(i32 15, i32 0, i32 1, i32 12, i32 0, float %14, float %16, float %18, float %20)
  ret void
}

; Function Attrs: nounwind readonly
declare i32 @llvm.SI.buffer.load.dword.i32.i32(<16 x i8>, i32, i32, i32, i32, i32, i32, i32, i32) #1

; Function Attrs: nounwind readonly
declare i32 @llvm.SI.buffer.load.dword.i32.v2i32(<16 x i8>, <2 x i32>, i32, i32, i32, i32, i32, i32, i32) #1

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)

attributes #0 = { "ShaderType"="1" }
attributes #1 = { nounwind readonly }

!0 = metadata !{metadata !"const", null, i32 1}
