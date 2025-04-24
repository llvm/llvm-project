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
  %_ZL1X_h.i.i3 = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 0, i32 0, i32 0, i1 false) #1
  %0 = call i32 @dx.op.threadId.i32(i32 93, i32 0) #2
  %1 = call %dx.types.ResRet.i32 @dx.op.bufferLoad.i32(i32 68, %dx.types.Handle %_ZL1X_h.i.i3, i32 %0, i32 0) #1
  %2 = extractvalue %dx.types.ResRet.i32 %1, 0
  %cmp.i1.not = icmp eq i32 %2, 0
  br i1 %cmp.i1.not, label %_Z4mainDv3_j.exit, label %for.body.i.lr.ph

for.body.i.lr.ph:                                 ; preds = %entry
  %_ZL3Out_h.i.i5 = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 1, i32 0, i32 0, i1 false) #1
  %_ZL2In_h.i.i4 = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 0, i32 1, i32 1, i1 false) #1
  %3 = call %dx.types.ResRet.f32 @dx.op.bufferLoad.f32(i32 68, %dx.types.Handle %_ZL2In_h.i.i4, i32 %0, i32 0) #1
  %4 = extractvalue %dx.types.ResRet.f32 %3, 0
  %5 = extractvalue %dx.types.ResRet.f32 %3, 1
  %6 = extractvalue %dx.types.ResRet.f32 %3, 2
  %7 = extractvalue %dx.types.ResRet.f32 %3, 3
  br label %for.body.i

for.body.i:                                       ; preds = %for.body.i.lr.ph, %for.body.i
  %I.0.i2 = phi i32 [ 0, %for.body.i.lr.ph ], [ %inc.i, %for.body.i ]
  call void @dx.op.bufferStore.f32(i32 69, %dx.types.Handle %_ZL3Out_h.i.i5, i32 %0, i32 0, float %4, float %5, float %6, float %7, i8 15)
  %inc.i = add nuw i32 %I.0.i2, 1
  %8 = call %dx.types.ResRet.i32 @dx.op.bufferLoad.i32(i32 68, %dx.types.Handle %_ZL1X_h.i.i3, i32 %0, i32 0) #1
  %9 = extractvalue %dx.types.ResRet.i32 %8, 0
  ; CHECK: %cmp.i = icmp ult i32 %inc.i, %9
  ; CHECK-NEXT: br i1 %cmp.i, label %for.body.i, label %_Z4mainDv3_j.exit
  %cmp.i = icmp ult i32 %inc.i, %9
  br i1 %cmp.i, label %for.body.i, label %_Z4mainDv3_j.exit, !llvm.loop !16

_Z4mainDv3_j.exit:                                ; preds = %for.body.i, %entry
  ret void

; uselistorder directives
  uselistorder %dx.types.Handle %_ZL1X_h.i.i3, { 1, 0 }
  uselistorder i32 %0, { 3, 0, 1, 2 }
  uselistorder label %for.body.i, { 1, 0 }
  }

declare %dx.types.Handle @dx.op.createHandle(i32, i8, i32, i32, i1)

declare i32 @dx.op.threadId.i32(i32, i32)

declare %dx.types.ResRet.i32 @dx.op.bufferLoad.i32(i32, %dx.types.Handle, i32, i32)

declare %dx.types.ResRet.f32 @dx.op.bufferLoad.f32(i32, %dx.types.Handle, i32, i32)

declare void @dx.op.bufferStore.f32(i32, %dx.types.Handle, i32, i32, float, float, float, float, i8)

; uselistorder directives
uselistorder i32 57, { 1, 0, 2 }
uselistorder i32 0, { 3, 0, 10, 1, 5, 6, 9, 2, 4, 7, 8 }
uselistorder i1 false, { 1, 0, 2 }
uselistorder i32 68, { 2, 0, 1 }
uselistorder i32 1, { 2, 0, 1 }
uselistorder ptr @dx.op.createHandle, { 2, 0, 1 }

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
