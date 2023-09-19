; RUN: opt -S --passes="ipsccp<func-spec>" -force-specialization < %s | FileCheck %s

; Tests that `bar` has been specialized and that the compiler did not crash
; while attempting to promote the alloca in `entry`.
; CHECK: bar.specialized.1

@block = internal constant [8 x i8] zeroinitializer, align 1

define dso_local void @entry() {
  %1 = alloca i32, align 4
  call void @foo(ptr nonnull %1)
  ret void
}

define internal void @foo(ptr nocapture readnone %0) {
  %2 = alloca i32, align 4
  call void @bar(ptr nonnull %2, ptr nonnull @block)
  call void @bar(ptr nonnull %2, ptr nonnull getelementptr inbounds ([8 x i8], ptr @block, i64 0, i64 4))
  ret void
}

define internal void @bar(ptr nocapture readonly %0, ptr nocapture readonly %1) {
  %3 = load i32, ptr %0, align 4
  %4 = load i32, ptr %1, align 4
  ret void
}

