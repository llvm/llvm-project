; RUN: opt -S -dxil-translate-metadata %s | FileCheck %s --check-prefix=MD
; RUN: opt -disable-output -passes='print<dxil-signature>' %s 2>&1 | FileCheck %s --check-prefix=ANALYSIS
; RUN: llc -filetype=obj %s -o %t.dxbc
; RUN: obj2yaml %t.dxbc | FileCheck %s --check-prefix=PARTS
; RUN: obj2yaml %t.dxbc | yaml2obj | obj2yaml | FileCheck %s --check-prefix=PARTS
; RUN: llvm-objcopy --dump-section=DXIL=%t.bc %t.dxbc
; RUN: llvm-dis %t.bc -o - | FileCheck %s --check-prefix=MD

target triple = "dxil-pc-shadermodel6.8-pixel"

define void @main() #0 {
  %a = call float @llvm.dx.load.input.f32(i32 0, i32 0, i8 0, i32 poison)
  %i = fptoui float %a to i32
  %row = and i32 %i, 1
  %b = call float @llvm.dx.load.input.f32(i32 1, i32 %row, i8 1, i32 poison)
  %cond = fcmp ogt float %a, 0.0
  br i1 %cond, label %write, label %exit
write:
  call void @llvm.dx.store.output.f32(i32 0, i32 0, i8 0, float %b)
  br label %exit
exit:
  call void @llvm.dx.store.output.f32(i32 0, i32 0, i8 2, float %a)
  ret void
}
attributes #0 = { "hlsl.shader"="pixel" }

!dx.valver = !{!20}
!20 = !{i32 1, i32 8}
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{!3, !4}
!2 = !{!5}
!3 = !{i32 0, !"A", i32 9, i32 0, !10, i32 0, i32 1, i8 2, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!4 = !{i32 1, !"B", i32 9, i32 0, !11, i32 0, i32 2, i8 2, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!5 = !{i32 0, !"SV_Target", i32 9, i32 16, !12, i32 0, i32 1, i8 4, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!10 = !{i32 0}
!11 = !{i32 0, i32 1}
!12 = !{i32 3}

; ANALYSIS: Inputs: 2 elements, 2 vectors
; ANALYSIS-NEXT: 0: A rows=1 cols=2 at 0:0 usage=1 dynamic=0
; ANALYSIS-NEXT: 1: B rows=2 cols=2 at 0:2 usage=8 dynamic=2
; ANALYSIS-NEXT: Outputs: 1 elements, 4 vectors
; ANALYSIS-NEXT: 0: SV_Target rows=1 cols=4 at 3:0 usage=5 dynamic=0

; MD-NOT: !dx.semantic.signatures
; MD: !dx.viewIdState = !{![[STATE:[0-9]+]]}
; MD: !dx.entryPoints =
; MD-DAG: ![[STATE]] = !{[10 x i32] [i32 8, i32 16, i32 20480, i32 0, i32 0, i32 20480, i32 0, i32 0, i32 0, i32 20480]}
; MD-DAG: !{i32 1, !"B", i8 9, i8 0, !{{[0-9]+}}, i8 2, i32 2, i8 2, i32 0, i8 2, ![[PROPS:[0-9]+]]}
; MD-DAG: ![[PROPS]] = !{i32 2, i32 2, i32 3, i32 2}
; MD-DAG: !{i32 0, !"SV_Target", i8 9, i8 16, !{{[0-9]+}}, i8 0, i32 1, i8 4, i32 3, i8 0, ![[WRITE:[0-9]+]]}
; MD-DAG: ![[WRITE]] = !{i32 3, i32 5}
; MD-NOT: !dx.semantic.signatures

; PARTS: Name: ISG1
; PARTS: Name: A
; PARTS: Mask: 3
; PARTS-NEXT: ExclusiveMask: 1
; PARTS: Name: B
; PARTS-NEXT: Index: 0
; PARTS: Register: 0
; PARTS-NEXT: Mask: 12
; PARTS-NEXT: ExclusiveMask: 8
; PARTS: Name: B
; PARTS-NEXT: Index: 1
; PARTS: Register: 1
; PARTS-NEXT: Mask: 12
; PARTS-NEXT: ExclusiveMask: 8
; PARTS: Name: OSG1
; PARTS: Name: SV_Target
; PARTS-NEXT: Index: 3
; PARTS-NEXT: SystemValue: Target
; PARTS: Register: 3
; PARTS-NEXT: Mask: 15
; PARTS-NEXT: ExclusiveMask: 10
; PARTS: Name: PSV0
; PARTS: SigInputVectors: 2
; PARTS-NEXT: SigOutputVectors: [ 4, 0, 0, 0 ]
; PARTS: SigInputElements:
; PARTS: Name: B
; PARTS-NEXT: Indices: [ 0, 1 ]
; PARTS-NEXT: StartRow: 0
; PARTS-NEXT: Cols: 2
; PARTS-NEXT: StartCol: 2
; PARTS-NEXT: Allocated: true
; PARTS: Interpolation: Linear
; PARTS-NEXT: DynamicMask: 0x2
; PARTS: SigOutputElements:
; PARTS: Indices: [ 3 ]
; PARTS-NEXT: StartRow: 3
; PARTS: Kind: Target
; PARTS: InputOutputMap:
; PARTS-NEXT: - [ 0x5000, 0x0, 0x0, 0x5000, 0x0, 0x0, 0x0, 0x5000 ]
