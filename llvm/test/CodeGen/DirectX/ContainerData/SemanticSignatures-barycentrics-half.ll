; RUN: split-file %s %t
; RUN: cat %t/common.ll %t/min.ll > %t/min-input.ll
; RUN: cat %t/common.ll %t/native.ll > %t/native-input.ll
; RUN: llc -filetype=obj %t/min-input.ll -o %t/min.dxbc
; RUN: llc -filetype=obj %t/native-input.ll -o %t/native.dxbc
; RUN: obj2yaml %t/min.dxbc | FileCheck %s --check-prefixes=PARTS,MIN
; RUN: obj2yaml %t/native.dxbc | FileCheck %s --check-prefixes=PARTS,NATIVE
; RUN: llvm-objcopy --dump-section=DXIL=%t/min.bc %t/min.dxbc
; RUN: llvm-objcopy --dump-section=DXIL=%t/native.bc %t/native.dxbc
; RUN: llvm-dis %t/min.bc -o - | FileCheck %s --check-prefixes=DXIL,MIN-MD
; RUN: llvm-dis %t/native.bc -o - | FileCheck %s --check-prefixes=DXIL,NATIVE-MD

; Each interpolation is tested in its own valid single-declaration signature.
; F16 remains an ordinary loadInput overload in both precision modes. Its
; unallocated location must not change component-relative masks or consume rows.
; MIN: MinimumPrecision: true
; MIN: NativeLowPrecision: false
; NATIVE: MinimumPrecision: false
; NATIVE: NativeLowPrecision: true
; PARTS: Name: ISG1
; PARTS: Name: SV_Barycentrics
; PARTS-NEXT: Index: 0
; PARTS-NEXT: SystemValue: Barycentrics
; PARTS-NEXT: CompType: Float16
; PARTS-NEXT: Register: 4294967295
; PARTS-NEXT: Mask: 7
; PARTS-NEXT: ExclusiveMask: 4
; MIN-NEXT: MinPrecision: Float16
; NATIVE-NEXT: MinPrecision: Default
; PARTS: Name: PSV0
; MIN: SampleFrequency: 1
; NATIVE: SampleFrequency: 0
; PARTS: SigInputVectors: 0
; PARTS: SigOutputVectors: [ 1, 0, 0, 0 ]
; PARTS: SigInputElements:
; PARTS: Indices: [ 0 ]
; PARTS-NEXT: StartRow: 0
; PARTS-NEXT: Cols: 3
; PARTS-NEXT: StartCol: 0
; PARTS-NEXT: Allocated: false
; PARTS-NEXT: Kind: Barycentrics
; MIN: Interpolation: LinearSample
; NATIVE: Interpolation: LinearNoperspectiveCentroid
; PARTS: InputOutputMap:
; PARTS-NEXT: - [  ]

; DXIL: call half @dx.op.loadInput.f16(i32 4, i32 0, i32 0, i8 2, i32 {{.*}})
; MIN-MD: !{i32 0, !"SV_Barycentrics", i8 8, i8 28, !{{[0-9]+}}, i8 6, i32 1, i8 3, i32 -1, i8 -1, ![[MASK:[0-9]+]]}
; NATIVE-MD: !{i32 0, !"SV_Barycentrics", i8 8, i8 28, !{{[0-9]+}}, i8 5, i32 1, i8 3, i32 -1, i8 -1, ![[MASK:[0-9]+]]}
; DXIL: ![[MASK]] = !{i32 3, i32 4}

;--- common.ll
target triple = "dxil-pc-shadermodel6.2-pixel"
define void @main() #0 {
  %b = call half @llvm.dx.load.input.f16(i32 0, i32 0, i8 2, i32 poison)
  %f = fpext half %b to float
  call void @llvm.dx.store.output.f32(i32 0, i32 0, i8 0, float %f)
  ret void
}
attributes #0 = { "hlsl.shader"="pixel" }
!dx.valver = !{!9}
!9 = !{i32 1, i32 8}
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{!3}
!2 = !{!4}
!4 = !{i32 0, !"SV_Target", i32 9, i32 16, !5, i32 0, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
!5 = !{i32 0}

;--- min.ll
!llvm.module.flags = !{!6}
!6 = !{i32 1, !"dx.nativelowprec", i32 0}
!3 = !{i32 0, !"SV_Barycentrics", i32 8, i32 28, !5, i32 6, i32 1, i8 3, i32 -1, i8 -1, i8 0, i8 0, i32 0}

;--- native.ll
!llvm.module.flags = !{!6}
!6 = !{i32 1, !"dx.nativelowprec", i32 1}
!3 = !{i32 0, !"SV_Barycentrics", i32 8, i32 28, !5, i32 5, i32 1, i8 3, i32 -1, i8 -1, i8 0, i8 0, i32 0}
