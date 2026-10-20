; RUN: llc -filetype=obj %s -o %t.dxbc
; RUN: obj2yaml %t.dxbc | FileCheck %s --check-prefix=PARTS
; RUN: llvm-objcopy --dump-section=DXIL=%t.bc %t.dxbc
; RUN: llvm-dis %t.bc -o - | FileCheck %s --check-prefixes=MD,DXIL

; SV_ViewID is not part of the signature. Using it marks the shader as using
; the view ID, and every written output component is conservatively treated as
; depending on it, both in dx.viewIdState and in PSV0.
target triple = "dxil-pc-shadermodel6.1-pixel"

define void @main() #0 {
  %a = call float @llvm.dx.load.input.f32(i32 0, i32 0, i8 0, i32 poison)
  %view = call i32 @llvm.dx.view.id()
  %f = uitofp i32 %view to float
  %result = fadd float %a, %f
  call void @llvm.dx.store.output.f32(i32 0, i32 0, i8 0, float %result)
  call void @llvm.dx.store.output.f32(i32 0, i32 0, i8 2, float %result)
  ret void
}
attributes #0 = { "hlsl.shader"="pixel" }

!dx.valver = !{!10}
!10 = !{i32 1, i32 8}
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{!3}
!2 = !{!6}
!3 = !{i32 0, !"A", i32 9, i32 0, !7, i32 2, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
!6 = !{i32 0, !"SV_Target", i32 9, i32 16, !7, i32 0, i32 1, i8 4, i32 0, i8 0, i8 0, i8 0, i32 0}
!7 = !{i32 0}

; DXIL: call i32 @dx.op.viewID.i32(i32 138)

; The state is the input and output scalar counts, the outputs depending on
; the view ID (components 0 and 2), then one output mask per input scalar.
; MD-DAG: !dx.viewIdState = !{![[STATE:[0-9]+]]}
; MD-DAG: ![[STATE]] = !{[7 x i32] [i32 4, i32 4, i32 5, i32 5, i32 0, i32 0, i32 0]}
; MD-DAG: !{i32 0, i64 2097152}

; PARTS: Name: SFI0
; PARTS: ViewID: true
; PARTS: Name: PSV0
; PARTS: UsesViewID: 1
; PARTS:      OutputVectorMasks:
; PARTS-NEXT:   - [ 0x5 ]
; PARTS:      InputOutputMap:
; PARTS-NEXT:   - [ 0x5, 0x0, 0x0, 0x0 ]
