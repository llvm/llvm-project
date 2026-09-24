; RUN: opt -S -passes='structurizecfg,verify' %s | FileCheck %s --implicit-check-not="!prof" --implicit-check-not="!block.uniformity.profile" --implicit-check-not="!branch.uniformity.profile"

; Reusing an empty prefix as a loop header changes its execution event. Keep
; its identity but invalidate the old count, without losing the entry anchor.
define void @if_else(i1 %enter, i1 %stop, i1 %again, ptr %out) !wave.profile !0 !uniformity.profile !9 {
; CHECK-LABEL: define void @if_else(
; CHECK-SAME: !wave.profile [[WAVE:![0-9]+]]
entry:
; CHECK: entry:
; CHECK: br label %prefix, !wave.profile.block [[ENTRY:![0-9]+]]{{$}}
  br label %prefix, !wave.profile.block !1
prefix:
; CHECK: prefix:
; CHECK: br {{[^!]*}}!wave.profile.block [[PREFIX:![0-9]+]]{{$}}
  br i1 %enter, label %a, label %exit.a, !prof !8, !block.uniformity.profile !9, !branch.uniformity.profile !9, !wave.profile.block !2
a:
; CHECK: a:
; CHECK: br {{[^!]*}}!wave.profile.block [[A:![0-9]+]]{{$}}
  store i32 3, ptr %out
  br i1 %stop, label %exit.a, label %b, !wave.profile.block !3
b:
; CHECK: b:
; CHECK: br {{[^!]*}}!wave.profile.block [[B:![0-9]+]]{{$}}
  store i32 4, ptr %out
  br i1 %again, label %a, label %exit.b, !wave.profile.block !4
exit.a:
  store i32 5, ptr %out
  br label %exit, !wave.profile.block !5
exit.b:
  store i32 6, ptr %out
  br label %exit, !wave.profile.block !6
exit:
; CHECK: exit:
; CHECK: ret void, !wave.profile.block [[EXIT:![0-9]+]]{{$}}
  ret void, !wave.profile.block !7
}
!0 = !{i64 2, i64 2685589004101179296, i64 10, i64 10, i64 30, i64 20, i64 10, i64 0, i64 10}
!1 = !{i64 2, i64 2685589004101179296, i64 0, i64 1, i64 1}
!2 = !{i64 2, i64 2685589004101179296, i64 1, i64 1, i64 2, i64 4}
!3 = !{i64 2, i64 2685589004101179296, i64 2, i64 1, i64 4, i64 3}
!4 = !{i64 2, i64 2685589004101179296, i64 3, i64 1, i64 2, i64 5}
!5 = !{i64 2, i64 2685589004101179296, i64 4, i64 1, i64 6}
!6 = !{i64 2, i64 2685589004101179296, i64 5, i64 1, i64 6}
!7 = !{i64 2, i64 2685589004101179296, i64 6, i64 1}
; CHECK: [[WAVE]] = distinct !{i64 2, i64 2685589004101179296, i64 10, i64 10, i64 30, i64 20, i64 10, i64 0, i64 10{{.*}}}
; CHECK: [[ENTRY]] = !{i64 2, i64 2685589004101179296, i64 0, i64 1{{.*}}}
; CHECK: [[PREFIX]] = !{i64 2, i64 2685589004101179296, i64 1, i64 0{{.*}}}
; CHECK: [[A]] = !{i64 2, i64 2685589004101179296, i64 2, i64 1{{.*}}}
; CHECK: [[B]] = !{i64 2, i64 2685589004101179296, i64 3, i64 1{{.*}}}
; CHECK: [[EXIT]] = !{i64 2, i64 2685589004101179296, i64 6, i64 1}

!8 = !{!"branch_weights", i32 90, i32 10}
!9 = !{}
