; RUN: llc < %s -mtriple=bpfel | FileCheck %s
; source:
;   int test(int (*f)(void)) { return f(); }

; Function Attrs: nounwind
define dso_local i32 @test(ptr nocapture %f) local_unnamed_addr {
entry:
  %call = tail call i32 %f()
; CHECK: callx r{{[0-9]+}}
  ret i32 %call
}

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 10.0.0 (https://github.com/llvm/llvm-project.git 7015a5c54b53d8d2297a3aa38bc32aab167bdcfc)"}
