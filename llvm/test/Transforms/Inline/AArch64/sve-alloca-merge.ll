; RUN: opt -mtriple=aarch64--linux-gnu -mattr=+sve < %s -passes=inline -S | FileCheck %s

define void @bar(ptr %a) {
entry:
  %b = alloca <vscale x 2 x i64>, align 16
  store <vscale x 2 x i64> zeroinitializer, ptr %b, align 16
  %c = load <vscale x 2 x i64>, ptr %a, align 16
  %d = load <vscale x 2 x i64>, ptr %b, align 16
  %e = add <vscale x 2 x i64> %c, %d
  %f = add <vscale x 2 x i64> %e, %c
  store <vscale x 2 x i64> %f, ptr %a, align 16
  ret void
}

define i64 @foo() {
; CHECK-LABEL: @foo(
; CHECK: call void @llvm.lifetime.start.p0(i64 -1, ptr %{{.*}})
; CHECK: call void @llvm.lifetime.end.p0(i64 -1, ptr %{{.*}})
entry:
  %a = alloca <vscale x 2 x i64>, align 16
  store <vscale x 2 x i64> zeroinitializer, ptr %a, align 16
  store i64 1, ptr %a, align 8
  call void @bar(ptr %a)
  %el = load i64, ptr %a
  ret i64 %el
}
