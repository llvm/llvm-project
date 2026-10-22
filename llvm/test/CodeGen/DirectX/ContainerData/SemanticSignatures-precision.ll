; RUN: split-file %s %t
; RUN: opt -disable-output -passes='print<dxil-signature>' %t/min.ll 2>&1 | FileCheck %s --check-prefix=MIN
; RUN: opt -disable-output -passes='print<dxil-signature>' %t/native.ll 2>&1 | FileCheck %s --check-prefix=NATIVE
; RUN: llc -filetype=obj %t/min.ll -o - | obj2yaml | FileCheck %s --check-prefix=MIN-PARTS
; RUN: llc -filetype=obj %t/native.ll -o - | obj2yaml | FileCheck %s --check-prefix=NATIVE-PARTS
; RUN: cat %t/system.ll %t/signed.ll %t/native-mode.ll > %t/signed-system.ll
; RUN: cat %t/system.ll %t/unsigned.ll %t/native-mode.ll > %t/unsigned-system.ll
; RUN: cat %t/system.ll %t/unsigned.ll %t/min-mode.ll > %t/min-system.ll
; RUN: llc -filetype=obj %t/signed-system.ll -o - | obj2yaml | FileCheck %s --check-prefixes=SYSTEM,SIGNED,NATIVE-SYSTEM
; RUN: llc -filetype=obj %t/unsigned-system.ll -o - | obj2yaml | FileCheck %s --check-prefixes=SYSTEM,UNSIGNED,NATIVE-SYSTEM
; RUN: llc -filetype=obj %t/min-system.ll -o - | obj2yaml | FileCheck %s --check-prefixes=SYSTEM,UNSIGNED,MIN-SYSTEM

; The frontend-provided layouts share a row in min-precision mode and use
; separate rows in native 16-bit mode. The 16-bit elements are unused: flags
; must still describe the signature.
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

; NATIVE-SYSTEM: NativeLowPrecision: true
; MIN-SYSTEM: MinimumPrecision: true
; SYSTEM: Name: ISG1
; SYSTEM: Name: SV_VertexID
; SIGNED: CompType: SInt16
; UNSIGNED: CompType: UInt16
; SYSTEM: Name: OSG1
; SYSTEM: Name: SV_Position
; SYSTEM: CompType: Float16
; SYSTEM: Mask: 15
; NATIVE-SYSTEM: MinPrecision: Default
; MIN-SYSTEM: MinPrecision: Float16

;--- system.ll
; Both system values accept 16-bit components in native and min-precision modes.
target triple = "dxil-pc-shadermodel6.8-vertex"
define void @main() #0 {
  %id = call i16 @llvm.dx.load.input.i16(i32 0, i32 0, i8 0, i32 poison)
  call void @llvm.dx.store.output.v4f16(i32 0, i32 0, i8 0, <4 x half> <half 0.0, half 0.0, half 0.0, half 1.0>)
  ret void
}
attributes #0 = { "hlsl.shader"="vertex" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{!3}
!2 = !{!4}
!4 = !{i32 0, !"SV_Position", i32 8, i32 3, !5, i32 0, i32 1, i8 4, i32 0, i8 0, i8 0, i8 0, i32 0}
!5 = !{i32 0}

;--- signed.ll
!3 = !{i32 0, !"SV_VertexID", i32 2, i32 1, !5, i32 0, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}

;--- unsigned.ll
!3 = !{i32 0, !"SV_VertexID", i32 3, i32 1, !5, i32 0, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}

;--- native-mode.ll
!llvm.module.flags = !{!6}
!6 = !{i32 1, !"dx.nativelowprec", i32 1}

;--- min-mode.ll
!llvm.module.flags = !{!6}
!6 = !{i32 1, !"dx.nativelowprec", i32 0}

;--- min.ll
target triple = "dxil-pc-shadermodel6.8-vertex"
define void @main() #0 { ret void }
attributes #0 = { "hlsl.shader"="vertex" }
!dx.semantic.signatures = !{!0}
!0 = !{ptr @main, null, !1}
!1 = !{!2, !3}
!2 = !{i32 0, !"A", i32 8, i32 0, !4, i32 0, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
!3 = !{i32 1, !"B", i32 9, i32 0, !4, i32 0, i32 1, i8 1, i32 0, i8 1, i8 0, i8 0, i32 0}
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
!2 = !{i32 0, !"A", i32 8, i32 0, !4, i32 0, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
!3 = !{i32 1, !"B", i32 9, i32 0, !4, i32 0, i32 1, i8 1, i32 1, i8 0, i8 0, i8 0, i32 0}
!4 = !{i32 0}
