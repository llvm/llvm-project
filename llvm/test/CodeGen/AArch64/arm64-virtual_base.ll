; RUN: llc < %s -O3 -mtriple=arm64-apple-ios -disable-post-ra | FileCheck %s
; <rdar://13463602>

%struct.Counter_Struct = type { i64, i64 }
%struct.Bicubic_Patch_Struct = type { %struct.Method_Struct*, i32, %struct.Object_Struct*, %struct.Texture_Struct*, %struct.Interior_Struct*, %struct.Object_Struct*, %struct.Object_Struct*, %struct.Bounding_Box_Struct, i64, i32, i32, i32, [4 x [4 x [3 x double]]], [3 x double], double, double, %struct.Bezier_Node_Struct* }
%struct.Method_Struct = type { i32 (%struct.Object_Struct*, %struct.Ray_Struct*, %struct.istack_struct*)*, i32 (double*, %struct.Object_Struct*)*, void (double*, %struct.Object_Struct*, %struct.istk_entry*)*, i8* (%struct.Object_Struct*)*, void (%struct.Object_Struct*, double*, %struct.Transform_Struct*)*, void (%struct.Object_Struct*, double*, %struct.Transform_Struct*)*, void (%struct.Object_Struct*, double*, %struct.Transform_Struct*)*, void (%struct.Object_Struct*, %struct.Transform_Struct*)*, void (%struct.Object_Struct*)*, void (%struct.Object_Struct*)* }
%struct.Object_Struct = type { %struct.Method_Struct*, i32, %struct.Object_Struct*, %struct.Texture_Struct*, %struct.Interior_Struct*, %struct.Object_Struct*, %struct.Object_Struct*, %struct.Bounding_Box_Struct, i64 }
%struct.Texture_Struct = type { i16, i16, i16, i32, float, float, float, %struct.Warps_Struct*, %struct.Pattern_Struct*, %struct.Blend_Map_Struct*, %union.anon.9, %struct.Texture_Struct*, %struct.Pigment_Struct*, %struct.Tnormal_Struct*, %struct.Finish_Struct*, %struct.Texture_Struct*, i32 }
%struct.Warps_Struct = type { i16, %struct.Warps_Struct* }
%struct.Pattern_Struct = type { i16, i16, i16, i32, float, float, float, %struct.Warps_Struct*, %struct.Pattern_Struct*, %struct.Blend_Map_Struct*, %union.anon.6 }
%struct.Blend_Map_Struct = type { i16, i16, i16, i64, %struct.Blend_Map_Entry* }
%struct.Blend_Map_Entry = type { float, i8, %union.anon }
%union.anon = type { [2 x double], [8 x i8] }
%union.anon.6 = type { %struct.anon.7 }
%struct.anon.7 = type { float, [3 x double] }
%union.anon.9 = type { %struct.anon.10 }
%struct.anon.10 = type { float, [3 x double] }
%struct.Pigment_Struct = type { i16, i16, i16, i32, float, float, float, %struct.Warps_Struct*, %struct.Pattern_Struct*, %struct.Blend_Map_Struct*, %union.anon.0, [5 x float] }
%union.anon.0 = type { %struct.anon }
%struct.anon = type { float, [3 x double] }
%struct.Tnormal_Struct = type { i16, i16, i16, i32, float, float, float, %struct.Warps_Struct*, %struct.Pattern_Struct*, %struct.Blend_Map_Struct*, %union.anon.3, float }
%union.anon.3 = type { %struct.anon.4 }
%struct.anon.4 = type { float, [3 x double] }
%struct.Finish_Struct = type { float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, [3 x float], [3 x float] }
%struct.Interior_Struct = type { i64, i32, float, float, float, float, float, %struct.Media_Struct* }
%struct.Media_Struct = type { i32, i32, i32, i32, i32, double, double, i32, i32, i32, i32, [5 x float], [5 x float], [5 x float], [5 x float], double, double, double, double*, %struct.Pigment_Struct*, %struct.Media_Struct* }
%struct.Bounding_Box_Struct = type { [3 x float], [3 x float] }
%struct.Ray_Struct = type { [3 x double], [3 x double], i32, [100 x %struct.Interior_Struct*] }
%struct.istack_struct = type { %struct.istack_struct*, %struct.istk_entry*, i32 }
%struct.istk_entry = type { double, [3 x double], [3 x double], %struct.Object_Struct*, i32, i32, double, double, i8* }
%struct.Transform_Struct = type { [4 x [4 x double]], [4 x [4 x double]] }
%struct.Bezier_Node_Struct = type { i32, [3 x double], double, i32, i8* }

define void @Precompute_Patch_Values(%struct.Bicubic_Patch_Struct* %Shape) {
; CHECK: Precompute_Patch_Values
; CHECK: ldr [[VAL2:q[0-9]+]], [x0, #272]
; CHECK-NEXT: ldr [[VAL:x[0-9]+]], [x0, #288]
; CHECK-NEXT: stur [[VAL2]], [sp, #216]
; CHECK-NEXT: str [[VAL]], [sp, #232]
entry:
  %Control_Points = alloca [16 x [3 x double]], align 8
  %arraydecay5.3.1 = getelementptr inbounds [16 x [3 x double]], [16 x [3 x double]]* %Control_Points, i64 0, i64 9, i64 0
  %tmp14 = bitcast double* %arraydecay5.3.1 to i8*
  %arraydecay11.3.1 = getelementptr inbounds %struct.Bicubic_Patch_Struct, %struct.Bicubic_Patch_Struct* %Shape, i64 0, i32 12, i64 1, i64 3, i64 0
  %tmp15 = bitcast double* %arraydecay11.3.1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %tmp14, i8* %tmp15, i64 24, i1 false)
  ret void
}

; Function Attrs: nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i1)
