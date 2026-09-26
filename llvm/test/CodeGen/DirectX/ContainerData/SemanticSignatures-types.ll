; RUN: opt -S -dxil-translate-metadata %s | FileCheck %s --check-prefix=MD
; RUN: llc -filetype=obj %s -o - | obj2yaml | FileCheck %s --check-prefix=PARTS

target triple = "dxil-pc-shadermodel6.8-pixel"

define void @main() #0 {
  %b = call i32 @llvm.dx.load.input.i32(i32 0, i32 0, i8 0, i32 poison)
  %i = call i32 @llvm.dx.load.input.i32(i32 1, i32 0, i8 0, i32 poison)
  %u = call i32 @llvm.dx.load.input.i32(i32 2, i32 0, i8 0, i32 poison)
  %p = call float @llvm.dx.load.input.f32(i32 3, i32 0, i8 2, i32 poison)
  %cond = icmp ne i32 %b, 0
  %v = select i1 %cond, i32 %i, i32 %u
  %f = sitofp i32 %v to float
  call void @llvm.dx.store.output.f32(i32 0, i32 0, i8 0, float %f)
  call void @llvm.dx.store.output.f32(i32 0, i32 0, i8 1, float %p)
  ret void
}
attributes #0 = { "hlsl.shader"="pixel" }
!dx.valver = !{!20}
!20 = !{i32 1, i32 8}
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{!3, !4, !5, !6}
!2 = !{!7}
!3 = !{i32 0, !"B", i32 1, i32 0, !8, i32 0, i32 1, i8 1, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!4 = !{i32 1, !"I", i32 4, i32 0, !8, i32 0, i32 1, i8 1, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!5 = !{i32 2, !"U", i32 5, i32 0, !8, i32 0, i32 1, i8 1, i32 -1, i8 -1, i8 0, i8 0, i32 0}
; Pixel input metadata may use the HLSL spelling; ISG1 is canonicalized.
!6 = !{i32 3, !"SV_POSITION", i32 9, i32 3, !8, i32 0, i32 1, i8 4, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!7 = !{i32 0, !"SV_Target", i32 9, i32 16, !8, i32 0, i32 1, i8 2, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!8 = !{i32 0}

; MD: !{i32 0, !"B", i8 1, i8 0, !{{[0-9]+}}, i8 1, i32 1, i8 1, i32 0, i8 0,
; MD: !{i32 1, !"I", i8 4, i8 0, !{{[0-9]+}}, i8 1, i32 1, i8 1, i32 0, i8 1,
; MD: !{i32 2, !"U", i8 5, i8 0, !{{[0-9]+}}, i8 1, i32 1, i8 1, i32 0, i8 2,
; MD: !{i32 3, !"SV_POSITION", i8 9, i8 3, !{{[0-9]+}}, i8 4, i32 1, i8 4, i32 1, i8 0,

; PARTS: Name: B
; PARTS: CompType: UInt32
; PARTS: Name: I
; PARTS: CompType: SInt32
; PARTS: Name: U
; PARTS: CompType: UInt32
; PARTS: Name: SV_Position
; PARTS: SystemValue: Position
; PARTS: Name: PSV0
; PARTS: SigInputElements:
; PARTS: Name: B
; PARTS: ComponentType: UInt32
; PARTS-NEXT: Interpolation: Constant
; PARTS: Name: I
; PARTS: ComponentType: SInt32
; PARTS: Name: U
; PARTS: ComponentType: UInt32
; PARTS: Kind: Position
; PARTS: Interpolation: LinearNoperspective
