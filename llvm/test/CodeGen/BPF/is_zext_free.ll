; RUN: llc -mtriple=bpfel -mattr=+alu32 < %s | FileCheck %s
; Source:
;   unsigned test(unsigned long x, unsigned long y) {
;     return x & y;
;   }
; Compilation flag:
;   clang -target bpf -O2 -emit-llvm -S test.c

; Function Attrs: norecurse nounwind readnone
define dso_local i32 @test(i64 %x, i64 %y) local_unnamed_addr {
entry:
  %and = and i64 %y, %x
  %conv = trunc i64 %and to i32
  ret i32 %conv
}

; CHECK: r[[REG1:[0-9]+]] = r{{[0-9]+}}
; CHECK: w[[REG1]] &= w{{[0-9]+}}

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 11.0.0 (https://github.com/llvm/llvm-project.git b3ab5b2e7ffe9964ddf75a92fd7a444fe5aaa426)"}
