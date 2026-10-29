; RUN: opt -disable-output -passes='print<dxil-signature>' %s 2>&1 | FileCheck %s --check-prefix=ANALYSIS
; RUN: llc -filetype=obj %s -o %t.dxbc
; RUN: obj2yaml %t.dxbc | FileCheck %s --check-prefix=PARTS
; RUN: llvm-objcopy --dump-section=DXIL=%t.bc %t.dxbc
; RUN: llvm-dis %t.bc -o - | FileCheck %s --check-prefixes=MD,DXIL

; Barycentrics keep signature IDs and ordinary loadInput accesses, but do not
; consume rows or participate in the packed input/output dependency table.
target triple = "dxil-pc-shadermodel6.1-pixel"

define void @main() #0 {
  %b = call <3 x float> @llvm.dx.load.input.v3f32(i32 0, i32 0, i8 0, i32 poison)
  %x = extractelement <3 x float> %b, i32 0
  %y = extractelement <3 x float> %b, i32 1
  %z = extractelement <3 x float> %b, i32 2
  %a = call float @llvm.dx.load.input.f32(i32 1, i32 0, i8 3, i32 poison)
  %xy = fadd float %x, %y
  %xyz = fadd float %xy, %z
  %result = fadd float %xyz, %a
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
!3 = !{i32 0, !"sv_barycentrics", i32 9, i32 28, !7, i32 7, i32 1, i8 3, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!4 = !{i32 1, !"A", i32 9, i32 0, !7, i32 2, i32 1, i8 4, i32 0, i8 0, i8 0, i8 0, i32 0}
!6 = !{i32 0, !"SV_Target", i32 9, i32 16, !7, i32 0, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
!7 = !{i32 0}

; ANALYSIS:      Inputs: 2 elements, 1 vectors
; ANALYSIS-NEXT: 0: sv_barycentrics rows=1 cols=3 at 4294967295:255 usage=7 dynamic=0
; ANALYSIS-NEXT: 1: A rows=1 cols=4 at 0:0 usage=8 dynamic=0
; ANALYSIS-NEXT: Outputs: 1 elements, 1 vectors

; DXIL: call float @dx.op.loadInput.f32(i32 4, i32 0, i32 0, i8 0, i32 {{.*}})
; DXIL: call float @dx.op.loadInput.f32(i32 4, i32 0, i32 0, i8 1, i32 {{.*}})
; DXIL: call float @dx.op.loadInput.f32(i32 4, i32 0, i32 0, i8 2, i32 {{.*}})
; DXIL: call float @dx.op.loadInput.f32(i32 4, i32 1, i32 0, i8 3, i32 {{.*}})
; MD: !dx.viewIdState = !{![[STATE:[0-9]+]]}
; MD: !dx.entryPoints =
; MD-DAG: ![[STATE]] = !{[6 x i32] [i32 4, i32 4, i32 0, i32 0, i32 0, i32 1]}
; MD-DAG: !{i32 0, !"sv_barycentrics", i8 9, i8 28, !{{[0-9]+}}, i8 7, i32 1, i8 3, i32 -1, i8 -1, ![[P:[0-9]+]]}
; MD-DAG: ![[P]] = !{i32 3, i32 7}
; MD-DAG: !{i32 0, i64 4194304}

; PARTS: Name: ISG1
; PARTS:      Name: SV_Barycentrics
; PARTS-NEXT: Index: 0
; PARTS-NEXT: SystemValue: Barycentrics
; PARTS-NEXT: CompType: Float32
; PARTS-NEXT: Register: 4294967295
; PARTS-NEXT: Mask: 7
; PARTS-NEXT: ExclusiveMask: 7
; PARTS: Name: PSV0
; PARTS: SampleFrequency: 1
; PARTS:      SigInputVectors: 1
; PARTS-NEXT: SigOutputVectors: [ 1, 0, 0, 0 ]
; PARTS: SigInputElements:
; PARTS:      Indices: [ 0 ]
; PARTS-NEXT: StartRow: 0
; PARTS-NEXT: Cols: 3
; PARTS-NEXT: StartCol: 0
; PARTS-NEXT: Allocated: false
; PARTS-NEXT: Kind: Barycentrics
; PARTS: Interpolation: LinearNoperspectiveSample
; PARTS:      InputOutputMap:
; PARTS-NEXT: - [ 0x0, 0x0, 0x0, 0x1 ]
