; RUN: llc -mtriple=arm-eabi -mattr=+neon < %s -o - | FileCheck %s

@c = global [4 x i32] [i32 3, i32 3, i32 3, i32 3], align 4
@d = common global i32 0, align 4

define void @foo() local_unnamed_addr nounwind norecurse {
entry:
  %0 = load <4 x i32>, ptr @c, align 4
  %constexpr = getelementptr inbounds [4 x i32], ptr @c, i32 0, i32 1
  %constexpr1 = icmp ne ptr %constexpr, @d
  %constexpr2 = zext i1 %constexpr1 to i32
  %constexpr3 = getelementptr inbounds [4 x i32], ptr @c, i32 0, i32 2
  %constexpr4 = icmp ne ptr %constexpr3, @d
  %constexpr5 = zext i1 %constexpr4 to i32
  %constexpr6 = getelementptr inbounds [4 x i32], ptr @c, i32 0, i32 3
  %constexpr7 = icmp ne ptr %constexpr6, @d
  %constexpr8 = zext i1 %constexpr7 to i32
  %constexpr.ins = insertelement <4 x i32> poison, i32 1, i32 0
  %constexpr.ins9 = insertelement <4 x i32> %constexpr.ins, i32 %constexpr2, i32 1
  %constexpr.ins10 = insertelement <4 x i32> %constexpr.ins9, i32 %constexpr5, i32 2
  %constexpr.ins11 = insertelement <4 x i32> %constexpr.ins10, i32 %constexpr8, i32 3
  %1 = and <4 x i32> %0, %constexpr.ins11
  store <4 x i32> %1, ptr @c, align 4
  ret void
; CHECK-NOT: mvnne
; CHECK: movne r{{[0-9]+}}, #1
; CHECK-NOT: mvnne
; CHECK: movne r{{[0-9]+}}, #1
; CHECK-NOT: mvnne
; CHECK: movne r{{[0-9]+}}, #1
; CHECK-NOT: mvnne
}
