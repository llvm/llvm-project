; RUN: opt -S --dxil-prepare %s | FileCheck %s

source_filename = "C:\\Users\\jbatista\\Desktop\\particle_life.hlsl"
target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxilv1.0-unknown-shadermodel6.0-compute"

%StructuredBuffer = type { <4 x i32> }
%StructuredBuffer.1 = type { <4 x float> }
%RWStructuredBuffer = type { <4 x float> }
%dx.types.Handle = type { ptr }
%dx.types.ResRet.i32 = type { i32, i32, i32, i32, i32 }
%dx.types.ResRet.f32 = type { float, float, float, float, i32 }

@0 = external constant %StructuredBuffer
@1 = external constant %StructuredBuffer.1
@2 = external constant %RWStructuredBuffer

; Function Attrs: noinline nounwind memory(readwrite, inaccessiblemem: none)
define void @main() local_unnamed_addr #0 {
entry:  
  %cmp.i1.not = icmp eq i32 1, 0
  br i1 %cmp.i1.not, label %_Z4mainDv3_j.exit, label %for.body.i

for.body.i:                                       ; preds = %entry
  %cmp.i = icmp ult i32 1, 2
  br i1 %cmp.i, label %for.body.i, label %_Z4mainDv3_j.exit, !llvm.loop !16

_Z4mainDv3_j.exit:                                ; preds = %for.body.i, %entry
  ret void

; uselistorder directives
  uselistorder label %for.body.i, { 1, 0 }
  }

declare %dx.types.Handle @dx.op.createHandle(i32, i8, i32, i32, i1)

declare i32 @dx.op.threadId.i32(i32, i32)

declare %dx.types.ResRet.i32 @dx.op.bufferLoad.i32(i32, %dx.types.Handle, i32, i32)

declare %dx.types.ResRet.f32 @dx.op.bufferLoad.f32(i32, %dx.types.Handle, i32, i32)

declare void @dx.op.bufferStore.f32(i32, %dx.types.Handle, i32, i32, float, float, float, float, i8)

; uselistorder directives

attributes #0 = { noinline nounwind memory(readwrite, inaccessiblemem: none) }
attributes #1 = { memory(read) }
attributes #2 = { memory(none) }

!dx.valver = !{!0}
!llvm.ident = !{!1}
!dx.shaderModel = !{!2}
!dx.version = !{!3}
!dx.resources = !{!4}
!dx.entryPoints = !{!11}
!llvm.module.flags = !{!14, !15}

!0 = !{i32 1, i32 8}
!1 = !{!"clang version 21.0.0git (https://github.com/llvm/llvm-project.git 9ed4c705ac1c5c8797f328694f6cd22fbcdae03b)"}
!2 = !{!"cs", i32 6, i32 0}
!3 = !{i32 1, i32 0}
!4 = !{!5, !9, null, null}
!5 = !{!6, !8}
!6 = !{i32 0, ptr @0, !"", i32 0, i32 0, i32 1, i32 12, i32 0, !7}
!7 = !{i32 1, i32 16}
!8 = !{i32 1, ptr @1, !"", i32 0, i32 1, i32 1, i32 12, i32 0, !7}
!9 = !{!10}
!10 = !{i32 0, ptr @2, !"", i32 0, i32 0, i32 1, i32 12, i1 false, i1 false, i1 false, !7}
!11 = !{ptr @main, !"main", null, !4, !12}
!12 = !{i32 0, i64 16, i32 4, !13}
!13 = !{i32 1, i32 1, i32 1}
!14 = !{i32 1, !"wchar_size", i32 4}
; CHECK: !15 = !{i32 2, !"frame-pointer", i32 2}
; this next line checks that nothing comes after the above check line. 
; No more metadata should be necessary after !15, the rest should be removed.
; CHECK-NOT: .
!15 = !{i32 2, !"frame-pointer", i32 2}
!16 = distinct !{!16, !17}
!17 = !{!"llvm.loop.mustprogress"}
