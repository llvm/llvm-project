; RUN: opt -S --passes=function-specialization < %s | FileCheck %s
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
; CHECK: @f.1
; CHECK: @f.2
