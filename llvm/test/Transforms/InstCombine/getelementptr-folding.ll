; RUN: opt -passes=instcombine -S < %s | FileCheck %s

; We used to fold this by rewriting the indices to 0, 0, 2, 0.  This is
; invalid because there is a 4-byte padding after each <3 x float> field.

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

%struct.matrix_float3x3 = type { [3 x <3 x float>] }

@matrix_identity_float3x3 = external global %struct.matrix_float3x3, align 16
@bbb = global float* getelementptr inbounds (%struct.matrix_float3x3, %struct.matrix_float3x3* @matrix_identity_float3x3, i64 0, i32 0, i64 1, i64 3)
; CHECK: @bbb = global float* getelementptr inbounds (%struct.matrix_float3x3, %struct.matrix_float3x3* @matrix_identity_float3x3, i64 0, i32 0, i64 1, i64 3)
