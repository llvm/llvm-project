; RUN: llc -march=hexagon < %s | FileCheck %s

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

; CHECK-LABEL: test0
; CHECK: memw(r29+#{{[0-9]+}}) += #1
define void @test0() #0 {
entry:
  %x = alloca i32, align 4
  %0 = bitcast i32* %x to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %0) #3
  call void @foo(i32* nonnull %x) #3
  %1 = load i32, i32* %x, align 4, !tbaa !1
  %inc = add nsw i32 %1, 1
  store i32 %inc, i32* %x, align 4, !tbaa !1
  call void @foo(i32* nonnull %x) #3
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %0) #3
  ret void
}

; CHECK-LABEL: test1
; CHECK: memw(r29+#{{[0-9]+}}) -= #1
define void @test1() #0 {
entry:
  %x = alloca i32, align 4
  %0 = bitcast i32* %x to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %0) #3
  call void @foo(i32* nonnull %x) #3
  %1 = load i32, i32* %x, align 4, !tbaa !1
  %inc = sub nsw i32 %1, 1
  store i32 %inc, i32* %x, align 4, !tbaa !1
  call void @foo(i32* nonnull %x) #3
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %0) #3
  ret void
}

; CHECK-LABEL: test2
; CHECK: memw(r29+#{{[0-9]+}}) = setbit(#0)
define void @test2() #0 {
entry:
  %x = alloca i32, align 4
  %0 = bitcast i32* %x to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %0) #3
  call void @foo(i32* nonnull %x) #3
  %1 = load i32, i32* %x, align 4, !tbaa !1
  %inc = or i32 %1, 1
  store i32 %inc, i32* %x, align 4, !tbaa !1
  call void @foo(i32* nonnull %x) #3
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %0) #3
  ret void
}

; CHECK-LABEL: test3
; CHECK: memw(r29+#{{[0-9]+}}) = clrbit(#0)
define void @test3() #0 {
entry:
  %x = alloca i32, align 4
  %0 = bitcast i32* %x to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %0) #3
  call void @foo(i32* nonnull %x) #3
  %1 = load i32, i32* %x, align 4, !tbaa !1
  %inc = and i32 %1, -2
  store i32 %inc, i32* %x, align 4, !tbaa !1
  call void @foo(i32* nonnull %x) #3
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %0) #3
  ret void
}

; CHECK-LABEL: test4
; CHECK: memw(r29+#{{[0-9]+}}) += r
define void @test4(i32 %a) #0 {
entry:
  %x = alloca i32, align 4
  %0 = bitcast i32* %x to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %0) #3
  call void @foo(i32* nonnull %x) #3
  %1 = load i32, i32* %x, align 4, !tbaa !1
  %inc = add nsw i32 %1, %a
  store i32 %inc, i32* %x, align 4, !tbaa !1
  call void @foo(i32* nonnull %x) #3
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %0) #3
  ret void
}

; CHECK-LABEL: test5
; CHECK: memw(r29+#{{[0-9]+}}) -= r
define void @test5(i32 %a) #0 {
entry:
  %x = alloca i32, align 4
  %0 = bitcast i32* %x to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %0) #3
  call void @foo(i32* nonnull %x) #3
  %1 = load i32, i32* %x, align 4, !tbaa !1
  %inc = sub nsw i32 %1, %a
  store i32 %inc, i32* %x, align 4, !tbaa !1
  call void @foo(i32* nonnull %x) #3
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %0) #3
  ret void
}

; CHECK-LABEL: test6
; CHECK: memw(r29+#{{[0-9]+}}) |= r
define void @test6(i32 %a) #0 {
entry:
  %x = alloca i32, align 4
  %0 = bitcast i32* %x to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %0) #3
  call void @foo(i32* nonnull %x) #3
  %1 = load i32, i32* %x, align 4, !tbaa !1
  %inc = or i32 %1, %a
  store i32 %inc, i32* %x, align 4, !tbaa !1
  call void @foo(i32* nonnull %x) #3
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %0) #3
  ret void
}

; CHECK-LABEL: test7
; CHECK: memw(r29+#{{[0-9]+}}) &= r
define void @test7(i32 %a) #0 {
entry:
  %x = alloca i32, align 4
  %0 = bitcast i32* %x to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %0) #3
  call void @foo(i32* nonnull %x) #3
  %1 = load i32, i32* %x, align 4, !tbaa !1
  %inc = and i32 %1, %a
  store i32 %inc, i32* %x, align 4, !tbaa !1
  call void @foo(i32* nonnull %x) #3
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %0) #3
  ret void
}


declare void @foo(i32*) #2
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1

attributes #0 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length64b" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length64b" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }

!1 = !{!2, !2, i64 0}
!2 = !{!"int", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
