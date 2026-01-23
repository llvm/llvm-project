; RUN: llc -mtriple=hexagon -disable-const64=1 < %s | FileCheck %s
; CHECK: combine(##4917,#88)

target triple = "hexagon"

%s.1 = type { %s.2 }
%s.2 = type { i32, ptr }

@g0 = internal constant [61 x i8] c"............................................................\00", align 4
@g1 = internal constant %s.1 { %s.2 { i32 8, ptr @g0 } }, align 4

define void @f0(i32 %a0) local_unnamed_addr #0 {
b0:
  %v0 = alloca ptr, align 4
  store ptr null, ptr %v0, align 4, !tbaa !0
  call void @f1(i32 88, i16 zeroext 4917, ptr nonnull %v0) #0
  %v1 = load ptr, ptr %v0, align 4, !tbaa !0
  %v2 = icmp eq ptr %v1, null
  br i1 %v2, label %b1, label %b2

b1:                                               ; preds = %b0
  call void @f2(ptr nonnull @g1) #0
  br label %b3

b2:                                               ; preds = %b0
  %v3 = call i32 @f3(i8 zeroext 22, ptr null, ptr nonnull %v1, i16 zeroext 88) #0
  %v4 = load ptr, ptr %v0, align 4, !tbaa !0
  call void @f4(ptr %v4, i32 88) #0
  br label %b3

b3:                                               ; preds = %b2, %b1
  ret void
}

declare void @f1(i32, i16 zeroext, ptr) local_unnamed_addr

declare void @f2(ptr) local_unnamed_addr

declare i32 @f3(i8 zeroext, ptr, ptr, i16 zeroext) local_unnamed_addr

declare void @f4(ptr, i32) local_unnamed_addr

attributes #0 = { nounwind optsize }

!0 = !{!1, !1, i64 0}
!1 = !{!"any pointer", !2}
!2 = !{!"omnipotent char", !3}
!3 = !{!"Simple C/C++ TBAA"}
