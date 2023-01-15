; RUN: opt < %s -passes=internalize -S | FileCheck %s

%struct.A = type { ptr }
%struct.B = type { ptr }
%struct.C = type { ptr }

; Class A has default visibility, so has no !vcall_visibility metadata before
; or after LTO.
; CHECK-NOT: @_ZTV1A = {{.*}}!vcall_visibility
@_ZTV1A = dso_local unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr null, ptr @_ZN1A3fooEv] }, align 8, !type !0, !type !1

; Class B has hidden visibility but public LTO visibility, so has no
; !vcall_visibility metadata before or after LTO.
; CHECK-NOT: @_ZTV1B = {{.*}}!vcall_visibility
@_ZTV1B = hidden unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr null, ptr @_ZN1B3fooEv] }, align 8, !type !2, !type !3

; Class C has hidden visibility, so the !vcall_visibility metadata is set to 1
; (linkage unit) before LTO, and 2 (translation unit) after LTO.
; CHECK: @_ZTV1C ={{.*}}!vcall_visibility [[MD_TU_VIS:![0-9]+]]
@_ZTV1C = hidden unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr null, ptr @_ZN1C3fooEv] }, align 8, !type !4, !type !5, !vcall_visibility !6

; Class D has translation unit visibility before LTO, and this is not changed
; by LTO.
; CHECK: @_ZTVN12_GLOBAL__N_11DE = {{.*}}!vcall_visibility [[MD_TU_VIS:![0-9]+]]
@_ZTVN12_GLOBAL__N_11DE = internal unnamed_addr constant { [3 x ptr] } zeroinitializer, align 8, !type !7, !type !9, !vcall_visibility !11

define dso_local void @_ZN1A3fooEv(ptr nocapture %this) {
entry:
  ret void
}

define hidden void @_ZN1B3fooEv(ptr nocapture %this) {
entry:
  ret void
}

define hidden void @_ZN1C3fooEv(ptr nocapture %this) {
entry:
  ret void
}

define hidden noalias nonnull ptr @_Z6make_dv() {
entry:
  %call = tail call ptr @_Znwm(i64 8) #3
  store ptr getelementptr inbounds ({ [3 x ptr] }, ptr @_ZTVN12_GLOBAL__N_11DE, i64 0, inrange i32 0, i64 2), ptr %call, align 8
  ret ptr %call
}

declare dso_local noalias nonnull ptr @_Znwm(i64)

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
