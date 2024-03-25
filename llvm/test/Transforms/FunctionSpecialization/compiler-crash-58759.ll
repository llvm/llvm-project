; RUN: opt -S --passes="default<O3>" < %s | FileCheck %s

define dso_local i32 @g0(i32 noundef %x) local_unnamed_addr {
entry:
  %call = tail call fastcc i32 @f(i32 noundef %x, ptr noundef nonnull @p0)
  ret i32 %call
}

define internal fastcc i32 @f(i32 noundef %x, ptr nocapture noundef readonly %p) noinline {
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

declare i32 @p0(i32 noundef)
declare i32 @p1(i32 noundef)

;; Tests that `f` has been fully specialize and it didn't cause compiler crash.
;; CHECK-DAG: f.specialized.1
;; CHECK-DAG: f.specialized.2
