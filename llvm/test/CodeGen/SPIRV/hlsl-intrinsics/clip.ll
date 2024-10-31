; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}


define void @test_scalar_lowering(float noundef %Buf) {
entry:
  %Buf.addr = alloca float, align 4
  store float %Buf, ptr %Buf.addr, align 4
  %0 = load float, ptr %Buf.addr, align 4
  %1 = fcmp olt float %0, 0.000000e+00
  br i1 %1, label %lt0, label %end

lt0:                                              ; preds = %entry
  call void @llvm.spv.clip()
  br label %end

end:                                              ; preds = %lt0, %entry
  ret void
}

declare void @llvm.spv.clip()


define void @test_vector(<4 x float> noundef %Buf) {
entry:
  %Buf.addr = alloca <4 x float>, align 16
  store <4 x float> %Buf, ptr %Buf.addr, align 16
  %1 = load <4 x float>, ptr %Buf.addr, align 16
  %2 = fcmp olt <4 x float> %1, zeroinitializer
  %3 = call i1 @llvm.vector.reduce.or.v4i1(<4 x i1> %2)
  br i1 %3, label %lt0, label %end

lt0:                                              ; preds = %entry
  call void @llvm.spv.clip()
  br label %end

end:                                              ; preds = %lt0, %entry
  ret void
}

declare i1 @llvm.vector.reduce.or.v4i1(<4 x i1>) #3
