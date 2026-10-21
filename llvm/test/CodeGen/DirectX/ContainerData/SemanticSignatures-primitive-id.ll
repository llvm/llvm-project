; RUN: opt -disable-output -passes='print<dxil-signature>' %s 2>&1 | FileCheck %s --check-prefix=ANALYSIS
; RUN: llc -filetype=obj %s -o %t.dxbc
; RUN: obj2yaml %t.dxbc | FileCheck %s --check-prefix=PARTS
; RUN: llvm-objcopy --dump-section=DXIL=%t.bc %t.dxbc
; RUN: llvm-dis %t.bc -o - | FileCheck %s --check-prefixes=MD,DXIL

; SV_PrimitiveID is a system-generated pixel input, read with loadInput.
target triple = "dxil-pc-shadermodel6.0-pixel"

define void @main() #0 {
  %a = call float @llvm.dx.load.input.f32(i32 0, i32 0, i8 0, i32 poison)
  %pid = call i32 @llvm.dx.load.input.i32(i32 1, i32 0, i8 0, i32 poison)
  %f = uitofp i32 %pid to float
  %result = fadd float %a, %f
  call void @llvm.dx.store.output.f32(i32 0, i32 0, i8 0, float %result)
  ret void
}
attributes #0 = { "hlsl.shader"="pixel" }

!dx.valver = !{!10}
!10 = !{i32 1, i32 8}
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{!3, !4}
!2 = !{!6}
!3 = !{i32 0, !"A", i32 9, i32 0, !7, i32 2, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
!4 = !{i32 1, !"SV_PrimitiveID", i32 5, i32 10, !7, i32 1, i32 1, i8 1, i32 1, i8 0, i8 0, i8 0, i32 0}
!6 = !{i32 0, !"SV_Target", i32 9, i32 16, !7, i32 0, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
!7 = !{i32 0}

; ANALYSIS:      Inputs: 2 elements, 2 vectors
; ANALYSIS-NEXT: 0: A rows=1 cols=1 at 0:0 usage=1 dynamic=0
; ANALYSIS-NEXT: 1: SV_PrimitiveID rows=1 cols=1 at 1:0 usage=1 dynamic=0

; DXIL: call i32 @dx.op.loadInput.i32(i32 4, i32 1, i32 0, i8 0, i32 {{.*}})
; MD-DAG: !{i32 1, !"SV_PrimitiveID", i8 5, i8 10, !{{[0-9]+}}, i8 1, i32 1, i8 1, i32 1, i8 0, !{{[0-9]+}}}

; PARTS: Name: ISG1
; PARTS:      Name: SV_PrimitiveID
; PARTS-NEXT: Index: 0
; PARTS-NEXT: SystemValue: PrimitiveID
; PARTS-NEXT: CompType: UInt32
; PARTS-NEXT: Register: 1
; PARTS-NEXT: Mask: 1
; PARTS-NEXT: ExclusiveMask: 1
; PARTS: Name: PSV0
; PARTS: SampleFrequency: 0
; PARTS: SigInputElements:
; PARTS:      Kind: Arbitrary
; PARTS:      Indices: [ 0 ]
; PARTS-NEXT: StartRow: 1
; PARTS-NEXT: Cols: 1
; PARTS-NEXT: StartCol: 0
; PARTS-NEXT: Allocated: true
; PARTS-NEXT: Kind: PrimitiveID
; PARTS-NEXT: ComponentType: UInt32
; PARTS-NEXT: Interpolation: Constant
