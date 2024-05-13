; RUN: llc < %s -march=bpfel | FileCheck %s
; source:
;   int test(int (*f)(void)) { return f(); }

; Function Attrs: nounwind
define dso_local i32 @test(ptr nocapture %f) local_unnamed_addr #0 {
entry:
  %call = tail call i32 %f() #1
; CHECK: callx r{{[0-9]+}}
  ret i32 %call
}

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 10.0.0 (https://github.com/llvm/llvm-project.git 7015a5c54b53d8d2297a3aa38bc32aab167bdcfc)"}
