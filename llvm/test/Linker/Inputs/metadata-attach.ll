@g1 = external global i32, !attach !0

@g2 = global i32 1, !attach !0

@g3 = global i32 2, !attach !0

declare !attach !0 void @f1()

define void @f2() !attach !0 {
  call void @f1()
  store i32 0, i32* @g1
  ret void
}

define void @f3() !attach !0 {
  ret void
}

!0 = !{i32 1}
