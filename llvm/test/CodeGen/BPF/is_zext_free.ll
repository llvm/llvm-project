; RUN: llc -mtriple=bpfel -mattr=+alu32 < %s | FileCheck %s
; Source:
;   unsigned test(unsigned long x, unsigned long y) {
;     return x & y;
;   }
; Compilation flag:
;   clang -target bpf -O2 -emit-llvm -S test.c

; Function Attrs: norecurse nounwind readnone
define dso_local i32 @test(i64 %x, i64 %y) local_unnamed_addr #0 {
entry:
  %and = and i64 %y, %x
  %conv = trunc i64 %and to i32
  ret i32 %conv
}

; CHECK: r[[REG1:[0-9]+]] = r{{[0-9]+}}
; CHECK: w[[REG1]] &= w{{[0-9]+}}

attributes #0 = { norecurse nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 11.0.0 (https://github.com/llvm/llvm-project.git b3ab5b2e7ffe9964ddf75a92fd7a444fe5aaa426)"}
