; RUN: llc < %s -march=x86

declare {x86_fp80, x86_fp80} @test()

define void @call2(x86_fp80 *%P1, x86_fp80 *%P2) {
  %a = call {x86_fp80,x86_fp80} @test()
  %b = extractvalue {x86_fp80,x86_fp80} %a, 1
  store x86_fp80 %b, x86_fp80* %P1
br label %L

L:
  %c = extractvalue {x86_fp80,x86_fp80} %a, 0
  store x86_fp80 %c, x86_fp80* %P2
  ret void
}
