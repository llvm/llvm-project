; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Check saturation conversion is translated when there is forward declaration
; of SPIRV entry.

; CHECK: OpDecorate %[[#SAT:]] SaturatedConversion
; CHECK: %[[#SAT]] = OpConvertFToU %[[#]] %[[#]]

declare spir_func zeroext i8 @_Z30__spirv_ConvertFToU_Ruchar_satf(float)

define spir_func void @forward(float %val, i8 %initval, ptr addrspace(1) %dst) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %new_val.0 = phi i8 [ %initval, %entry ], [ %call1, %for.body ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %cmp = icmp ult i32 %i.0, 1
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %call1 = call spir_func zeroext i8 @_Z30__spirv_ConvertFToU_Ruchar_satf(float noundef %val)
  %inc = add i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  store i8 %new_val.0, ptr addrspace(1) %dst, align 1
  ret void
}
