; RUN: opt -S -dxil-translate-metadata %s | FileCheck %s
; RUN: opt -S -passes=dxil-translate-metadata %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.8-library"

define void @vs() #0 {
  call void @llvm.dx.store.output.f32(i32 0, i32 0, i8 0, float 1.0)
  ret void
}
define void @ps() #1 {
  call void @llvm.dx.store.output.f32(i32 0, i32 0, i8 0, float 2.0)
  ret void
}
define void @empty() #2 { ret void }
attributes #0 = { "hlsl.shader"="vertex" }
attributes #1 = { "hlsl.shader"="pixel" }
attributes #2 = { "hlsl.shader"="compute" "hlsl.numthreads"="1,1,1" }

!dx.semantic.signatures = !{!0, !1, !2}
!0 = !{ptr @vs, null, !3}
!1 = !{ptr @ps, !8, !4}
!2 = !{ptr @empty, null, null}
!3 = !{!5}
!4 = !{!6}
!5 = !{i32 0, !"A", i32 9, i32 0, !7, i32 0, i32 1, i8 1, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!6 = !{i32 0, !"SV_Target", i32 9, i32 16, !7, i32 0, i32 1, i8 1, i32 -1, i8 -1, i8 0, i8 0, i32 0}
!7 = !{i32 2}
!8 = !{}

; CHECK-NOT: !dx.semantic.signatures
; CHECK-NOT: !dx.viewIdState
; CHECK: !dx.entryPoints = !{![[LIB:[0-9]+]], ![[VS:[0-9]+]], ![[PS:[0-9]+]], ![[EMPTY:[0-9]+]]}
; CHECK-DAG: ![[LIB]] = !{null, !"", null, null, null}
; CHECK-DAG: ![[VS]] = !{ptr @vs, !"vs", ![[VSIG:[0-9]+]], null, !{{[0-9]+}}}
; CHECK-DAG: ![[PS]] = !{ptr @ps, !"ps", ![[PSIG:[0-9]+]], null, !{{[0-9]+}}}
; CHECK-DAG: ![[EMPTY]] = !{ptr @empty, !"empty", null, null, !{{[0-9]+}}}
; CHECK-DAG: ![[VSIG]] = !{null, ![[VOUT:[0-9]+]], null}
; CHECK-DAG: ![[PSIG]] = !{null, ![[POUT:[0-9]+]], null}
; CHECK-DAG: ![[VOUT]] = !{![[VE:[0-9]+]]}
; CHECK-DAG: ![[POUT]] = !{![[PE:[0-9]+]]}
; CHECK-DAG: ![[VE]] = !{i32 0, !"A", i8 9, i8 0, ![[INDEX:[0-9]+]], i8 0, i32 1, i8 1, i32 0, i8 0, !{{[0-9]+}}}
; CHECK-DAG: ![[PE]] = !{i32 0, !"SV_Target", i8 9, i8 16, ![[INDEX]], i8 0, i32 1, i8 1, i32 2, i8 0, !{{[0-9]+}}}
; CHECK-NOT: !dx.semantic.signatures
; CHECK-NOT: !dx.viewIdState
