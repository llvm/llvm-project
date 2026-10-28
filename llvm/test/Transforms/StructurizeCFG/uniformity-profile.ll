; RUN: opt -S -passes=structurizecfg -structurizecfg-skip-uniform-regions=false -verify-each %s | FileCheck %s --implicit-check-not=block.uniformity.profile --implicit-check-not=branch.uniformity.profile
;
; Block profiles describe executions of the original block, not the branch
; condition. Replacing a terminator must keep the block hint, without copying
; the old branch hint onto a synthesized condition or annotating a new Flow.

; CHECK-LABEL: define void @diamond(
; CHECK: entry:
; CHECK: br i1 {{.*}}, label {{.*}}, label {{.*}}, !block.uniformity.profile ![[MD:[0-9]+]]{{$}}
; CHECK: then:
; CHECK-NEXT: store i32 1, ptr %out
; CHECK-NEXT: br label {{.*}}, !block.uniformity.profile ![[MD]]{{$}}
; CHECK: exit:
; CHECK-NEXT: ret void, !block.uniformity.profile ![[MD]]{{$}}
define void @diamond(i1 %cond, ptr %out) !uniformity.profile !0 {
entry:
  br i1 %cond, label %then, label %else, !block.uniformity.profile !0, !branch.uniformity.profile !0
then:
  store i32 1, ptr %out
  br label %exit, !block.uniformity.profile !0
else:
  store i32 2, ptr %out
  br label %exit
exit:
  ret void, !block.uniformity.profile !0
}

; A reused empty prefix gains loop-backedge executions and must lose its hint.
; CHECK-LABEL: define void @loop_prefix(
; CHECK: prefix:{{.*}}preds = %{{.*}}, %entry
; CHECK: a:
; CHECK-NEXT: store i32 3, ptr %out
; CHECK-NEXT: br i1 {{.*}}, label {{.*}}, label {{.*}}, !block.uniformity.profile ![[MD]]{{$}}
define void @loop_prefix(i1 %enter, i1 %stop, i1 %again, ptr %out) !uniformity.profile !0 {
entry:
  br label %prefix
prefix:
  br i1 %enter, label %a, label %exit.a, !block.uniformity.profile !0, !branch.uniformity.profile !0
a:
  store i32 3, ptr %out
  br i1 %stop, label %exit.a, label %b, !block.uniformity.profile !0
b:
  store i32 4, ptr %out
  br i1 %again, label %a, label %exit.b
exit.a:
  store i32 5, ptr %out
  br label %exit
exit.b:
  store i32 6, ptr %out
  br label %exit
exit:
  ret void
}

; A nonempty prefix keeps its original executions. needPrefix removes its
; terminator before changeExit removes it again; preserve the saved hint.
; CHECK-LABEL: define void @nonempty_loop_prefix(
; CHECK: prefix:{{.*}}preds = %entry
; CHECK-NEXT: store i32 2, ptr %out
; CHECK-NEXT: br label {{.*}}, !block.uniformity.profile ![[MD]]{{$}}
; CHECK: a:
; CHECK-NEXT: store i32 3, ptr %out
; CHECK-NEXT: br i1 {{.*}}, label {{.*}}, label {{.*}}, !block.uniformity.profile ![[MD]]{{$}}
define void @nonempty_loop_prefix(i1 %enter, i1 %stop, i1 %again, ptr %out) !uniformity.profile !0 {
entry:
  br label %prefix
prefix:
  store i32 2, ptr %out
  br i1 %enter, label %a, label %exit.a, !block.uniformity.profile !0, !branch.uniformity.profile !0
a:
  store i32 3, ptr %out
  br i1 %stop, label %exit.a, label %b, !block.uniformity.profile !0
b:
  store i32 4, ptr %out
  br i1 %again, label %a, label %exit.b
exit.a:
  store i32 5, ptr %out
  br label %exit
exit.b:
  store i32 6, ptr %out
  br label %exit
exit:
  ret void
}

; Preserve the inner block profiles when a containing region is rewritten.
; CHECK-LABEL: define void @nested(
; CHECK: outer.then:
; CHECK-NEXT: br i1 {{.*}}, label {{.*}}, label {{.*}}, !block.uniformity.profile ![[MD]]{{$}}
; CHECK: inner.then:
; CHECK-NEXT: store i32 5, ptr %out
; CHECK-NEXT: br label {{.*}}, !block.uniformity.profile ![[MD]]{{$}}
; CHECK: inner.exit:
; CHECK-NEXT: store i32 7, ptr %out
; CHECK-NEXT: br label {{.*}}, !block.uniformity.profile ![[MD]]{{$}}
define void @nested(i1 %outer, i1 %inner, ptr %out) !uniformity.profile !0 {
entry:
  br i1 %outer, label %outer.then, label %outer.else
outer.then:
  br i1 %inner, label %inner.then, label %inner.else, !block.uniformity.profile !0, !branch.uniformity.profile !0
inner.then:
  store i32 5, ptr %out
  br label %inner.exit, !block.uniformity.profile !0
inner.else:
  store i32 6, ptr %out
  br label %inner.exit
inner.exit:
  store i32 7, ptr %out
  br label %exit, !block.uniformity.profile !0
outer.else:
  store i32 8, ptr %out
  br label %exit
exit:
  ret void
}

!0 = !{}
