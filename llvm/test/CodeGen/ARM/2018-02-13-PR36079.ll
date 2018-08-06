; RUN: llc -mtriple=arm-eabi -mattr=+neon < %s -o - | FileCheck %s

@c = global [4 x i32] [i32 3, i32 3, i32 3, i32 3], align 4
@d = common global i32 0, align 4

define void @foo() local_unnamed_addr nounwind norecurse {
entry:
  %0 = load <4 x i32>, <4 x i32>* bitcast ([4 x i32]* @c to <4 x i32>*), align 4
  %1 = and <4 x i32> %0,
           <i32 1,
            i32 zext (i1 icmp ne (i32* getelementptr inbounds ([4 x i32], [4 x i32]* @c, i32 0, i32 1), i32* @d) to i32),
            i32 zext (i1 icmp ne (i32* getelementptr inbounds ([4 x i32], [4 x i32]* @c, i32 0, i32 2), i32* @d) to i32),
            i32 zext (i1 icmp ne (i32* getelementptr inbounds ([4 x i32], [4 x i32]* @c, i32 0, i32 3), i32* @d) to i32)>
  store <4 x i32> %1, <4 x i32>* bitcast ([4 x i32]* @c to <4 x i32>*), align 4
  ret void
; CHECK-NOT: mvnne
; CHECK: movne r{{[0-9]+}}, #1
; CHECK-NOT: mvnne
; CHECK: movne r{{[0-9]+}}, #1
; CHECK-NOT: mvnne
; CHECK: movne r{{[0-9]+}}, #1
; CHECK-NOT: mvnne
}
