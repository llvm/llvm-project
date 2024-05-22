; RUN: llc --filetype=obj %s -o - 2>&1 | dxil-dis -o - | FileCheck %s
target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxil-unknown-shadermodel6.7-library"

%"$Globals" = type { float }

@CBV = external constant %"$Globals"

define void @main() {
  ret void
}

!llvm.ident = !{!0}
!dx.version = !{!1}
!dx.valver = !{!2}
!dx.shaderModel = !{!3}
!dx.entryPoints = !{!8}

!0 = !{!"clang version 15.0.0"}
!1 = !{i32 1, i32 0}
!2 = !{i32 1, i32 7}
!3 = !{!"lib", i32 6, i32 7}
!4 = !{null, null, !5, null}
!5 = !{!6}
; CHECK-DAG:!{{[0-9]+}} = !{i32 0, %"$Globals"* @CBV
!6 = !{i32 0, ptr @CBV, !"", i32 0, i32 0, i32 1, i32 4, null}
!7 = !{[2 x i32] [i32 0, i32 1]}
; CHECK-DAG:!{{[0-9]+}} = !{void ()* @main
!8 = !{ptr @main, !"main", null, !4, null}
