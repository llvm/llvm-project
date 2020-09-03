; RUN: opt < %s -internalize -S | FileCheck %s

%struct.A = type { i32 (...)** }
%struct.B = type { i32 (...)** }
%struct.C = type { i32 (...)** }

; Class A has default visibility, so has no !vcall_visibility metadata before
; or after LTO.
; CHECK-NOT: @_ZTV1A = {{.*}}!vcall_visibility
@_ZTV1A = dso_local unnamed_addr constant { [3 x i8*] } { [3 x i8*] [i8* null, i8* null, i8* bitcast (void (%struct.A*)* @_ZN1A3fooEv to i8*)] }, align 8, !type !0, !type !1

; Class B has hidden visibility but public LTO visibility, so has no
; !vcall_visibility metadata before or after LTO.
; CHECK-NOT: @_ZTV1B = {{.*}}!vcall_visibility
@_ZTV1B = hidden unnamed_addr constant { [3 x i8*] } { [3 x i8*] [i8* null, i8* null, i8* bitcast (void (%struct.B*)* @_ZN1B3fooEv to i8*)] }, align 8, !type !2, !type !3

; Class C has hidden visibility, so the !vcall_visibility metadata is set to 1
; (linkage unit) before LTO, and 2 (translation unit) after LTO.
; CHECK: @_ZTV1C ={{.*}}!vcall_visibility [[MD_TU_VIS:![0-9]+]]
@_ZTV1C = hidden unnamed_addr constant { [3 x i8*] } { [3 x i8*] [i8* null, i8* null, i8* bitcast (void (%struct.C*)* @_ZN1C3fooEv to i8*)] }, align 8, !type !4, !type !5, !vcall_visibility !6

; Class D has translation unit visibility before LTO, and this is not changed
; by LTO.
; CHECK: @_ZTVN12_GLOBAL__N_11DE = {{.*}}!vcall_visibility [[MD_TU_VIS:![0-9]+]]
@_ZTVN12_GLOBAL__N_11DE = internal unnamed_addr constant { [3 x i8*] } zeroinitializer, align 8, !type !7, !type !9, !vcall_visibility !11

define dso_local void @_ZN1A3fooEv(%struct.A* nocapture %this) {
entry:
  ret void
}

define hidden void @_ZN1B3fooEv(%struct.B* nocapture %this) {
entry:
  ret void
}

define hidden void @_ZN1C3fooEv(%struct.C* nocapture %this) {
entry:
  ret void
}

define hidden noalias nonnull i8* @_Z6make_dv() {
entry:
  %call = tail call i8* @_Znwm(i64 8) #3
  %0 = bitcast i8* %call to i32 (...)***
  store i32 (...)** bitcast (i8** getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @_ZTVN12_GLOBAL__N_11DE, i64 0, inrange i32 0, i64 2) to i32 (...)**), i32 (...)*** %0, align 8
  ret i8* %call
}

declare dso_local noalias nonnull i8* @_Znwm(i64)

; CHECK: [[MD_TU_VIS]] = !{i64 2}
!0 = !{i64 16, !"_ZTS1A"}
!1 = !{i64 16, !"_ZTSM1AFvvE.virtual"}
!2 = !{i64 16, !"_ZTS1B"}
!3 = !{i64 16, !"_ZTSM1BFvvE.virtual"}
!4 = !{i64 16, !"_ZTS1C"}
!5 = !{i64 16, !"_ZTSM1CFvvE.virtual"}
!6 = !{i64 1}
!7 = !{i64 16, !8}
!8 = distinct !{}
!9 = !{i64 16, !10}
!10 = distinct !{}
!11 = !{i64 2}
