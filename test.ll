define void @test(ptr %p, i1 %idx) {
  %v = load <2 x i32>, ptr %p
  %e = extractelement <2 x i32> %v, i1 %idx
  ret void
}
