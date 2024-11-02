; RUN: opt -S --passes=ipsccp -specialize-functions < %s | FileCheck %s
define dso_local i32 @p0(i32 noundef %x) {
entry:
  %add = add nsw i32 %x, 1
  ret i32 %add
}

define dso_local i32 @p1(i32 noundef %x) {
entry:
  %sub = add nsw i32 %x, -1
  ret i32 %sub
}

; CHECK-LABEL: define dso_local i32 @f0
; CHECK:       tail call fastcc i32 @g.[[#A:]]({{.*}}@p0)
;
define dso_local i32 @f0(i32 noundef %x) {
entry:
  %call = tail call fastcc i32 @g(i32 noundef %x, ptr noundef nonnull @p0)
  ret i32 %call
}

; CHECK-LABEL: define dso_local i32 @f1
; CHECK:       tail call fastcc i32 @g.[[#B:]]({{.*}}@p1)
;
define dso_local i32 @f1(i32 noundef %x) {
entry:
  %call = tail call fastcc i32 @g(i32 noundef %x, ptr noundef nonnull @p1)
  ret i32 %call
}

; @g gets fully specialized
; CHECK-NOT: define internal fastcc i32 @g(

define internal fastcc i32 @g(i32 noundef %x, ptr nocapture noundef readonly %p) noinline  {
entry:
  %pcall = tail call i32 %p(i32 noundef %x)
  %fcall = tail call fastcc i32 @f(i32 noundef %pcall, ptr noundef nonnull %p)
  ret i32 %fcall
}

; CHECK-LABEL: define dso_local i32 @g0
; CHECK:       tail call fastcc i32 @f.[[#C:]]({{.*}}@p0)
;
define dso_local i32 @g0(i32 noundef %x) {
entry:
  %call = tail call fastcc i32 @f(i32 noundef %x, ptr noundef nonnull @p0)
  ret i32 %call
}

define internal fastcc i32 @f(i32 noundef %x, ptr nocapture noundef readonly %p) noinline  {
entry:
  %call = tail call i32 %p(i32 noundef %x)
  %add = add nsw i32 %call, %x
  ret i32 %add
}

; CHECK-LABEL: define dso_local i32 @g1
; CHECK:       tail call fastcc i32 @f.[[#D:]]({{.*}}@p1)
;
define dso_local i32 @g1(i32 noundef %x) {
entry:
  %call = tail call fastcc i32 @f(i32 noundef %x, ptr noundef nonnull @p1)
  ret i32 %call
}

define dso_local i32 @g2(i32 noundef %x) {
entry:
  %call = tail call fastcc i32 @f(i32 noundef %x, ptr poison)
  ret i32 %call
}

; Check that a single argument, that cannot be used for specialisation, does not
; prevent specialisation based on other arguments.
;
; Also check that for callsites which reside in the body of newly created
; (specialized) functions, the lattice value of the arguments is known.
;
; CHECK-DAG: define internal fastcc i32 @g.[[#A]]
; CHECK-DAG: define internal fastcc i32 @g.[[#B]]
; CHECK-DAG: define internal fastcc i32 @f.[[#C]]
; CHECK-DAG: define internal fastcc i32 @f.[[#D]]
