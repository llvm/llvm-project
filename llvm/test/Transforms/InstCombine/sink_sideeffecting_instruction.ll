; RUN: opt -instcombine -S < %s | FileCheck %s

; Function Attrs: noinline uwtable
define dso_local i32 @foo(i32* nocapture writeonly %arg) local_unnamed_addr #0 {
bb:
  %tmp = call i32 @baz()
  store i32 %tmp, i32* %arg, align 4, !tbaa !2
  %tmp1 = call i32 @baz()
  ret i32 %tmp1
}
declare dso_local i32 @baz() local_unnamed_addr

; Function Attrs: uwtable
; This is an equivalent IR for a c-style example with a large function (foo)
; with out-params which are unused in the caller(test8). Note that foo is
; marked noinline to prevent IPO transforms.
; int foo();
; 
; extern int foo(int *out) __attribute__((noinline));
; int foo(int *out) {
;   *out = baz();
;   return baz();
; }
; 
; int test() {
; 
;   int notdead;
;   if (foo(&notdead))
;     return 0;
; 
;   int dead;
;   int tmp = foo(&dead);
;   if (notdead)
;     return tmp;
;   return bar();
; }

; TODO: We should be able to sink the second call @foo at bb5 down to bb_crit_edge 
define dso_local i32 @test() local_unnamed_addr #2 {
; CHECK-LABEL: test
; CHECK: bb5:
; CHECK: %tmp7 = call i32 @foo(i32* nonnull writeonly %tmp1) 
; CHECK-NEXT: br i1 %tmp9, label %bb10, label %bb_crit_edge
bb:
  %tmp = alloca i32, align 4
  %tmp1 = alloca i32, align 4
  %tmp2 = bitcast i32* %tmp to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %tmp2) #4
  %tmp3 = call i32 @foo(i32* nonnull writeonly %tmp)
  %tmp4 = icmp eq i32 %tmp3, 0
  br i1 %tmp4, label %bb5, label %bb14

bb5:                                              ; preds = %bb
  %tmp6 = bitcast i32* %tmp1 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %tmp6) #4
  %tmp8 = load i32, i32* %tmp, align 4, !tbaa !2
  %tmp9 = icmp eq i32 %tmp8, 0
  %tmp7 = call i32 @foo(i32* nonnull writeonly %tmp1)
  br i1 %tmp9, label %bb10, label %bb_crit_edge

bb10:                                             ; preds = %bb5
  %tmp11 = call i32 @bar()
  br label %bb12

bb_crit_edge:
  br label %bb12

bb12:                                             ; preds = %bb10, %bb5
  %tmp13 = phi i32 [ %tmp11, %bb10 ], [ %tmp7, %bb_crit_edge ]
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %tmp6) #4
  br label %bb14

bb14:                                             ; preds = %bb12, %bb
  %tmp15 = phi i32 [ %tmp13, %bb12 ], [ 0, %bb ]
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %tmp2)
  ret i32 %tmp15
}

declare i32 @bar()
; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #3

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #3

attributes #0 = { noinline uwtable argmemonly nounwind willreturn writeonly }
attributes #3 = { argmemonly nofree nosync nounwind willreturn }
!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 10.0.0-4ubuntu1 "}
!2 = !{!3, !3, i64 0}
!3 = !{!"int", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}
