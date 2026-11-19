; RUN: opt -S -verify-each -passes=always-inline %s | FileCheck %s

declare i64 @getter(ptr)
declare void @opaque(ptr)
declare i32 @__gxx_personality_v0(...)

define internal i64 @map(ptr %source, ptr noalias %out) alwaysinline {
  %value = call i64 @getter(ptr %source)
  store i64 %value, ptr %out, align 8
  ret i64 %value
}

define i64 @ordinary_call(ptr %source, ptr %out) {
; CHECK-LABEL: define i64 @ordinary_call(
; CHECK: call i64 @getter(ptr %source), !noalias ![[ORDINARY:[0-9]+]]
; CHECK: store i64 {{.*}}, ptr %out, align 8, !alias.scope ![[ORDINARY]]
  %value = call i64 @map(ptr %source, ptr %out)
  ret i64 %value
}

define i64 @unwinding_call(ptr %source, ptr %out) personality ptr @__gxx_personality_v0 {
; CHECK-LABEL: define i64 @unwinding_call(
; CHECK: invoke i64 @getter(ptr %source)
; CHECK-NEXT: to label {{.*}} unwind label %cleanup, !noalias ![[UNWINDING:[0-9]+]]
; CHECK: store i64 {{.*}}, ptr %out, align 8, !alias.scope ![[UNWINDING]]
entry:
  %value = invoke i64 @map(ptr %source, ptr %out)
      to label %exit unwind label %cleanup
exit:
  ret i64 %value
cleanup:
  %exception = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } %exception
}

define internal void @explicit_metadata(ptr %source) alwaysinline {
  call void @opaque(ptr %source), !prof !10, !alias.scope !0, !noalias !3, !llvm.mem.parallel_loop_access !7, !llvm.access.group !6, !noalias.addrspace !9
  ret void
}

define void @unwinding_explicit_metadata(ptr %source) personality ptr @__gxx_personality_v0 {
; CHECK-LABEL: define void @unwinding_explicit_metadata(
; CHECK: invoke void @opaque(ptr %source)
; CHECK-NEXT: to label {{.*}} unwind label %cleanup, !prof ![[PROF:[0-9]+]], !alias.scope ![[ALIAS:[0-9]+]], !noalias ![[NOALIAS:[0-9]+]], !llvm.mem.parallel_loop_access ![[PARALLEL:[0-9]+]], !llvm.access.group ![[ACCESS:[0-9]+]]{{$}}
entry:
  invoke void @explicit_metadata(ptr %source)
      to label %exit unwind label %cleanup
exit:
  ret void
cleanup:
  %exception = landingpad { ptr, i32 } cleanup
  resume { ptr, i32 } %exception
}

; CHECK: ![[PROF]] = !{!"branch_weights", i32 17}
; CHECK: ![[ALIAS]] = !{![[ALIAS_SCOPE:[0-9]+]]}
; CHECK: ![[ALIAS_SCOPE]] = distinct !{![[ALIAS_SCOPE]], !{{[0-9]+}}, !"scope"}
; CHECK: ![[NOALIAS]] = !{![[NOALIAS_SCOPE:[0-9]+]]}
; CHECK: ![[NOALIAS_SCOPE]] = distinct !{![[NOALIAS_SCOPE]], !{{[0-9]+}}, !"noalias"}
; CHECK: ![[PARALLEL]] = !{![[LOOP:[0-9]+]]}
; CHECK: ![[LOOP]] = distinct !{![[LOOP]]}
; CHECK: ![[ACCESS]] = distinct !{}

!0 = !{!1}
!1 = distinct !{!1, !2, !"scope"}
!2 = distinct !{!2, i1 false, !"domain"}
!3 = !{!4}
!4 = distinct !{!4, !5, !"noalias"}
!5 = distinct !{!5, i1 false, !"noalias domain"}
!6 = distinct !{}
!7 = !{!8}
!8 = distinct !{!8}
!9 = !{i32 1, i32 2}
!10 = !{!"branch_weights", i32 17}
