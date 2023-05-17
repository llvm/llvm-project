; RUN: opt -S --passes="ipsccp<func-spec>" -funcspec-max-clones=1 < %s | FileCheck %s
define internal i32 @f(i32 noundef %x, ptr nocapture noundef readonly %p, ptr nocapture noundef readonly %q) noinline {
entry:
  %call = tail call i32 %p(i32 noundef %x)
  %call1 = tail call i32 %q(i32 noundef %x)
  %add = add nsw i32 %call1, %call
  ret i32 %add
}

define internal i32 @g(i32 noundef %x, ptr nocapture noundef readonly %p, ptr nocapture noundef readonly %q) noinline {
entry:
  %call = tail call i32 %p(i32 noundef %x)
  %call1 = tail call i32 %q(i32 noundef %x)
  %sub = sub nsw i32 %call, %call1
  ret i32 %sub
}

define i32 @h0(i32 noundef %x) {
entry:
  %call = tail call i32 @f(i32 noundef %x, ptr noundef nonnull @pp, ptr noundef nonnull @qq)
  ret i32 %call
}

define i32 @h1(i32 noundef %x) {
entry:
  %call = tail call i32 @f(i32 noundef %x, ptr noundef nonnull @qq, ptr noundef nonnull @pp)
  ret i32 %call
}

define i32 @h2(i32 noundef %x, ptr nocapture noundef readonly %p) {
entry:
  %call = tail call i32 @g(i32 noundef %x, ptr noundef %p, ptr noundef nonnull @pp)
  ret i32 %call
}

define i32 @h3(i32 noundef %x, ptr nocapture noundef readonly %p) {
entry:
  %call = tail call i32 @g(i32 noundef %x, ptr noundef %p, ptr noundef nonnull @qq)
  ret i32 %call
}

declare i32 @pp(i32 noundef)
declare i32 @qq(i32 noundef)


; Check that the global ranking causes two specialisations of
; `f` to be chosen, whereas the old algorithm would choose
; one specialsation of `f` and one of `g`.

; CHECK-DAG: define internal i32 @f.1
; CHECK-DAG: define internal i32 @f.2
