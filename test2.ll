define void @test(ptr %p, i32 %idx.base) {
  %idx = and i32 %idx.base, 1
  %v = load <2 x i32>, ptr %p
  %e1 = extractelement <2 x i32> %v, i32 %idx
  %e2 = extractelement <2 x i32> %v, i32 %idx
  ret void
}
