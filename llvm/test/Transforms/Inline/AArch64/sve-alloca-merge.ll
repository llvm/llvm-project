; RUN: opt -mtriple=aarch64--linux-gnu -mattr=+sve < %s -passes=inline -S | FileCheck %s

define void @bar(<vscale x 2 x i64>* %a) {
entry:
  %b = alloca <vscale x 2 x i64>, align 16
  store <vscale x 2 x i64> zeroinitializer, <vscale x 2 x i64>* %b, align 16
  %c = load <vscale x 2 x i64>, <vscale x 2 x i64>* %a, align 16
  %d = load <vscale x 2 x i64>, <vscale x 2 x i64>* %b, align 16
  %e = add <vscale x 2 x i64> %c, %d
  %f = add <vscale x 2 x i64> %e, %c
  store <vscale x 2 x i64> %f, <vscale x 2 x i64>* %a, align 16
  ret void
}

define i64 @foo() {
; CHECK-LABEL: @foo(
; CHECK: %0 = bitcast <vscale x 2 x i64>* %{{.*}} to i8*
; CHECK-NEXT: call void @llvm.lifetime.start.p0i8(i64 -1, i8* %0)
; CHECK: %1 = bitcast <vscale x 2 x i64>* %{{.*}} to i8*
; CHECK-NEXT: call void @llvm.lifetime.end.p0i8(i64 -1, i8* %1)
entry:
  %a = alloca <vscale x 2 x i64>, align 16
  store <vscale x 2 x i64> zeroinitializer, <vscale x 2 x i64>* %a, align 16
  %a1 = bitcast <vscale x 2 x i64>* %a to i64*
  store i64 1, i64* %a1, align 8
  call void @bar(<vscale x 2 x i64>* %a)
  %el = load i64, i64* %a1
  ret i64 %el
}
