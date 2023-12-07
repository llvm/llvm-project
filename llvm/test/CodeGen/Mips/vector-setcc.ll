; RUN: llc -march=mipsel < %s

@a = common global <4 x i32> zeroinitializer, align 16
@b = common global <4 x i32> zeroinitializer, align 16
@g0 = common global <4 x i32> zeroinitializer, align 16

define void @foo0() nounwind {
entry:
  %0 = load <4 x i32>, ptr @a, align 16
  %1 = load <4 x i32>, ptr @b, align 16
  %cmp = icmp slt <4 x i32> %0, %1
  %sext = sext <4 x i1> %cmp to <4 x i32>
  store <4 x i32> %sext, ptr @g0, align 16
  ret void
}

