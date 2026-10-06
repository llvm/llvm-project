; RUN: split-file %s %t
; RUN: opt -disable-output -passes='print<dxil-signature>' %t/min.ll 2>&1 | FileCheck %s --check-prefix=MIN
; RUN: opt -disable-output -passes='print<dxil-signature>' %t/native.ll 2>&1 | FileCheck %s --check-prefix=NATIVE
; RUN: llc -filetype=obj %t/min.ll -o - | obj2yaml | FileCheck %s --check-prefix=MIN-PARTS
; RUN: llc -filetype=obj %t/native.ll -o - | obj2yaml | FileCheck %s --check-prefix=NATIVE-PARTS

; The 16-bit elements are unused: flags must still describe the signature.
; MIN: Outputs: 2 elements, 1 vectors
; MIN-NEXT: 0: A rows=1 cols=1 at 0:0 usage=0 dynamic=0
; MIN-NEXT: 1: B rows=1 cols=1 at 0:1 usage=0 dynamic=0
; NATIVE: Outputs: 2 elements, 2 vectors
; NATIVE-NEXT: 0: A rows=1 cols=1 at 0:0 usage=0 dynamic=0
; NATIVE-NEXT: 1: B rows=1 cols=1 at 1:0 usage=0 dynamic=0

; MIN-PARTS: MinimumPrecision: true
; MIN-PARTS: Name: A
; MIN-PARTS: CompType: Float16
; MIN-PARTS-NEXT: Register: 0
; MIN-PARTS-NEXT: Mask: 1
; MIN-PARTS-NEXT: ExclusiveMask: 15
; MIN-PARTS-NEXT: MinPrecision: Float16
; MIN-PARTS: Name: B
; MIN-PARTS: CompType: Float32
; MIN-PARTS-NEXT: Register: 0
; MIN-PARTS-NEXT: Mask: 2
; MIN-PARTS: SigOutputVectors: [ 1, 0, 0, 0 ]

; NATIVE-PARTS: NativeLowPrecision: true
; NATIVE-PARTS: Name: A
; NATIVE-PARTS: CompType: Float16
; NATIVE-PARTS-NEXT: Register: 0
; NATIVE-PARTS-NEXT: Mask: 1
; NATIVE-PARTS-NEXT: ExclusiveMask: 15
; NATIVE-PARTS-NEXT: MinPrecision: Default
; NATIVE-PARTS: Name: B
; NATIVE-PARTS: CompType: Float32
; NATIVE-PARTS-NEXT: Register: 1
; NATIVE-PARTS-NEXT: Mask: 1
; NATIVE-PARTS: SigOutputVectors: [ 2, 0, 0, 0 ]

;--- min.ll
target triple = "dxil-pc-shadermodel6.8-vertex"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="vertex" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, null, !1}
!1 = !{!2, !3}
!2 = !{i32 0, !"A", i32 8, i32 0, !4, i32 0, i32 1, i8 1, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!3 = !{i32 1, !"B", i32 9, i32 0, !4, i32 0, i32 1, i8 1, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!4 = !{i32 0}

;--- native.ll
target triple = "dxil-pc-shadermodel6.8-vertex"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="vertex" }
!llvm.module.flags = !{!5}
!5 = !{i32 1, !"dx.nativelowprec", i32 1}
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, null, !1}
!1 = !{!2, !3}
!2 = !{i32 0, !"A", i32 8, i32 0, !4, i32 0, i32 1, i8 1, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!3 = !{i32 1, !"B", i32 9, i32 0, !4, i32 0, i32 1, i8 1, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!4 = !{i32 0}
