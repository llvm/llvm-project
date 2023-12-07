; RUN: llc -mtriple=arm-eabi -mattr=+neon < %s -o - | FileCheck %s

@c = global [4 x i32] [i32 3, i32 3, i32 3, i32 3], align 4
@d = common global i32 0, align 4

define void @foo() local_unnamed_addr nounwind norecurse {
entry:
  %0 = load <4 x i32>, ptr @c, align 4
  %1 = and <4 x i32> %0,
           <i32 1,
            i32 zext (i1 icmp ne (ptr getelementptr inbounds ([4 x i32], ptr @c, i32 0, i32 1), ptr @d) to i32),
            i32 zext (i1 icmp ne (ptr getelementptr inbounds ([4 x i32], ptr @c, i32 0, i32 2), ptr @d) to i32),
            i32 zext (i1 icmp ne (ptr getelementptr inbounds ([4 x i32], ptr @c, i32 0, i32 3), ptr @d) to i32)>
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
