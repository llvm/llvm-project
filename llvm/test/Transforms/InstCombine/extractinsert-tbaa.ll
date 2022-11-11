; RUN: opt -S -passes=instcombine %s -o - | FileCheck %s

%Complex = type { double, double }

; Check that instcombine preserves TBAA when narrowing loads
define double @teststructextract(ptr %val) {
; CHECK: load double, {{.*}}, !tbaa
; CHECK-NOT: load %Complex
    %loaded = load %Complex, ptr %val, !tbaa !1
    %real = extractvalue %Complex %loaded, 0
    ret double %real
}

define double @testarrayextract(ptr %val) {
; CHECK: load double, {{.*}}, !tbaa
; CHECK-NOT: load [2 x double]
    %loaded = load [2 x double], ptr %val, !tbaa !1
    %real = extractvalue [2 x double] %loaded, 0
    ret double %real
}

; Check that inscombine preserves TBAA when breaking up stores
define void @teststructinsert(ptr %loc, double %a, double %b) {
; CHECK: store double %a, {{.*}}, !tbaa
; CHECK: store double %b, {{.*}}, !tbaa
; CHECK-NOT: store %Complex
    %inserted  = insertvalue %Complex undef,      double %a, 0
    %inserted2 = insertvalue %Complex %inserted,  double %b, 1
    store %Complex %inserted2, ptr %loc, !tbaa !1
    ret void
}

define void @testarrayinsert(ptr %loc, double %a, double %b) {
; CHECK: store double %a, {{.*}}, !tbaa
; CHECK: store double %b, {{.*}}, !tbaa
; CHECK-NOT: store [2 x double]
    %inserted  = insertvalue [2 x double] undef,      double %a, 0
    %inserted2 = insertvalue [2 x double] %inserted,  double %b, 1
    store [2 x double] %inserted2, ptr %loc, !tbaa !1
    ret void
}

!0 = !{!"tbaa_root"}
!1 = !{!2, !2, i64 0}
!2 = !{!"Complex", !0, i64 0}
