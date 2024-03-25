; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

; CHECK-DAG: #[[R0:.*]] = #llvm.tbaa_root<id = "Simple C/C++ TBAA">
; CHECK-DAG: #[[D0:.*]] = #llvm.tbaa_type_desc<id = "scalar type", members = {<#[[R0]], 0>}>
; CHECK-DAG: #[[$T0:.*]] = #llvm.tbaa_tag<base_type = #[[D0]], access_type = #[[D0]], offset = 0>
; CHECK-DAG: #[[R1:.*]] = #llvm.tbaa_root<id = "Other language TBAA">
; CHECK-DAG: #[[D1:.*]] = #llvm.tbaa_type_desc<id = "other scalar type", members = {<#[[R1]], 0>}>
; CHECK-DAG: #[[$T1:.*]] = #llvm.tbaa_tag<base_type = #[[D1]], access_type = #[[D1]], offset = 0>

; CHECK-LABEL: llvm.func @tbaa1
; CHECK:         llvm.store %{{.*}}, %{{.*}} {
; CHECK-SAME:        tbaa = [#[[$T0]]]
; CHECK-SAME:    } : i8, !llvm.ptr
; CHECK:         llvm.store %{{.*}}, %{{.*}} {
; CHECK-SAME:        tbaa = [#[[$T1]]]
; CHECK-SAME:    } : i8, !llvm.ptr
define dso_local void @tbaa1(ptr %0, ptr %1) {
  store i8 1, ptr %0, align 4, !tbaa !0
  store i8 1, ptr %1, align 4, !tbaa !3
  ret void
}

!0 = !{!1, !1, i64 0}
!1 = !{!"scalar type", !2, i64 0}
!2 = !{!"Simple C/C++ TBAA"}

!3 = !{!4, !4, i64 0}
!4 = !{!"other scalar type", !5, i64 0}
!5 = !{!"Other language TBAA"}

; // -----

; CHECK-DAG: #[[R0:.*]] = #llvm.tbaa_root<id = "Simple C/C++ TBAA">
; CHECK-DAG: #[[$T0:.*]] = #llvm.tbaa_tag<base_type = #[[D2:.*]], access_type = #[[D1:.*]], offset = 8>
; CHECK-DAG: #[[D1]] = #llvm.tbaa_type_desc<id = "long long", members = {<#[[D0:.*]], 0>}>
; CHECK-DAG: #[[D0]] = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#[[R0]], 0>}>
; CHECK-DAG: #[[D2]] = #llvm.tbaa_type_desc<id = "agg2_t", members = {<#[[D1]], 0>, <#[[D1]], 8>}>
; CHECK-DAG: #[[$T1:.*]] = #llvm.tbaa_tag<base_type = #[[D4:.*]], access_type = #[[D3:.*]], offset = 0>
; CHECK-DAG: #[[D3]] = #llvm.tbaa_type_desc<id = "int", members = {<#[[D0]], 0>}>
; CHECK-DAG: #[[D4]] = #llvm.tbaa_type_desc<id = "agg1_t", members = {<#[[D3]], 0>, <#[[D3]], 4>}>

; CHECK-LABEL: llvm.func @tbaa2
; CHECK:         llvm.load %{{.*}} {
; CHECK-SAME:        tbaa = [#[[$T0]]]
; CHECK-SAME:    } : !llvm.ptr -> i64
; CHECK:         llvm.store %{{.*}}, %{{.*}} {
; CHECK-SAME:        tbaa = [#[[$T1]]]
; CHECK-SAME:    } : i32, !llvm.ptr
%struct.agg2_t = type { i64, i64 }
%struct.agg1_t = type { i32, i32 }

define dso_local void @tbaa2(ptr %0, ptr %1) {
  %3 = getelementptr inbounds %struct.agg2_t, ptr %1, i32 0, i32 1
  %4 = load i64, ptr %3, align 8, !tbaa !6
  %5 = trunc i64 %4 to i32
  %6 = getelementptr inbounds %struct.agg1_t, ptr %0, i32 0, i32 0
  store i32 %5, ptr %6, align 4, !tbaa !11
  ret void
}

!6 = !{!7, !8, i64 8}
!7 = !{!"agg2_t", !8, i64 0, !8, i64 8}
!8 = !{!"long long", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!12, !13, i64 0}
!12 = !{!"agg1_t", !13, i64 0, !13, i64 4}
!13 = !{!"int", !9, i64 0}

; // -----

; CHECK-LABEL: llvm.func @supported_ops
define void @supported_ops(ptr %arg1, float %arg2, i32 %arg3, i32 %arg4) {
  ; CHECK: llvm.load {{.*}}tbaa =
  %1 = load i32, ptr %arg1, !tbaa !0
  ; CHECK: llvm.store {{.*}}tbaa =
  store i32 %1, ptr %arg1, !tbaa !0
  ; CHECK: llvm.atomicrmw {{.*}}tbaa =
  %2 = atomicrmw fmax ptr %arg1, float %arg2 acquire, !tbaa !0
  ; CHECK: llvm.cmpxchg {{.*}}tbaa =
  %3 = cmpxchg ptr %arg1, i32 %arg3, i32 %arg4 monotonic seq_cst, !tbaa !0
  ; CHECK: "llvm.intr.memcpy"{{.*}}tbaa =
  call void @llvm.memcpy.p0.p0.i32(ptr %arg1, ptr %arg1, i32 4, i1 false), !tbaa !0
  ; CHECK: "llvm.intr.memset"{{.*}}tbaa =
  call void @llvm.memset.p0.i32(ptr %arg1, i8 42, i32 4, i1 false), !tbaa !0
  ; CHECK: llvm.call{{.*}}tbaa =
  call void @foo(ptr %arg1), !tbaa !0
  ret void
}

declare void @llvm.memcpy.p0.p0.i32(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i32, i1 immarg)
declare void @llvm.memset.p0.i32(ptr nocapture writeonly, i8, i32, i1 immarg)
declare void @foo(ptr %arg1)

!0 = !{!1, !1, i64 0}
!1 = !{!"scalar type", !2, i64 0}
!2 = !{!"Simple C/C++ TBAA"}

; // -----

; CHECK: #llvm.tbaa_root
; CHECK-NOT: <{{.*}}>
; CHECK: {{[[:space:]]}}

define void @nameless_root(ptr %arg1) {
  ; CHECK: llvm.load {{.*}}tbaa =
  %1 = load i32, ptr %arg1, !tbaa !0
  ret void
}

!0 = !{!1, !1, i64 0}
!1 = !{!"scalar type", !2, i64 0}
!2 = !{}

