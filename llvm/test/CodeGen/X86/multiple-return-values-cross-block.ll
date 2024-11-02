; RUN: llc < %s -mtriple=i686--

declare {x86_fp80, x86_fp80} @test()

define void @call2(ptr%P1, ptr%P2) {
  %a = call {x86_fp80,x86_fp80} @test()
  %b = extractvalue {x86_fp80,x86_fp80} %a, 1
  store x86_fp80 %b, ptr %P1
br label %L

L:
  %c = extractvalue {x86_fp80,x86_fp80} %a, 0
  store x86_fp80 %c, ptr %P2
  ret void
}
