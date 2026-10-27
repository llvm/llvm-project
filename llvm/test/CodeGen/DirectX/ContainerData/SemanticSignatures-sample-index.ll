; RUN: opt -disable-output -passes='print<dxil-signature>' %s 2>&1 | FileCheck %s --check-prefix=ANALYSIS
; RUN: llc -filetype=obj %s -o %t.dxbc
; RUN: obj2yaml %t.dxbc | FileCheck %s --check-prefix=PARTS
; RUN: llvm-objcopy --dump-section=DXIL=%t.bc %t.dxbc
; RUN: llvm-dis %t.bc -o - | FileCheck %s --check-prefixes=MD,DXIL

; SV_SampleIndex is an unpacked element: it keeps a signature ID, but is read with
; dx.op.sampleIndex, does not consume a row and marks the pixel shader as
; sample-frequency.
target triple = "dxil-pc-shadermodel6.0-pixel"

define void @main() #0 {
  %p = call float @llvm.dx.load.input.f32(i32 0, i32 0, i8 0, i32 poison)
  %si = call i32 @llvm.dx.sample.index()
  %f = uitofp i32 %si to float
  %result = fadd float %p, %f
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
!4 = !{i32 1, !"SV_SampleIndex", i32 5, i32 12, !7, i32 1, i32 1, i8 1, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!6 = !{i32 0, !"SV_Target", i32 9, i32 16, !7, i32 0, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
!7 = !{i32 0}

; ANALYSIS:      Inputs: 2 elements, 1 vectors
; ANALYSIS-NEXT: 0: A rows=1 cols=1 at 0:0 usage=1 dynamic=0
; ANALYSIS-NEXT: 1: SV_SampleIndex rows=1 cols=1 at 4294967295:255 usage=0 dynamic=0
; ANALYSIS-NEXT: Outputs: 1 elements, 1 vectors

; DXIL: call float @dx.op.loadInput.f32(i32 4, i32 0, i32 0, i8 0, i32 {{.*}})
; DXIL: call i32 @dx.op.sampleIndex.i32(i32 90)
; MD-DAG: !{i32 1, !"SV_SampleIndex", i8 5, i8 12, !{{[0-9]+}}, i8 1, i32 1, i8 1, i32 -1, i8 -1, null}

; PARTS: Name: ISG1
; PARTS:      Name: SV_SampleIndex
; PARTS-NEXT: Index: 0
; PARTS-NEXT: SystemValue: SampleIndex
; PARTS-NEXT: CompType: UInt32
; PARTS-NEXT: Register: 4294967295
; PARTS-NEXT: Mask: 1
; PARTS-NEXT: ExclusiveMask: 0
; PARTS: Name: PSV0
; PARTS: SampleFrequency: 1
; PARTS:      SigInputVectors: 1
; PARTS: SigInputElements:
; PARTS:      Kind: Arbitrary
; PARTS:      Indices: [ 0 ]
; PARTS-NEXT: StartRow: 0
; PARTS-NEXT: Cols: 1
; PARTS-NEXT: StartCol: 0
; PARTS-NEXT: Allocated: false
; PARTS-NEXT: Kind: SampleIndex
; PARTS-NEXT: ComponentType: UInt32
; PARTS-NEXT: Interpolation: Constant
