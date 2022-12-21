; RUN: llc -march=hexagon < %s | FileCheck %s

%struct.test_struct = type { i32, i8, i64 }

; CHECK: r1 = #45
define void @foo(ptr noalias nocapture sret(%struct.test_struct) %agg.result, i32 %a) #0 {
entry:
  call void @bar(ptr sret(%struct.test_struct) %agg.result, i32 45) #0
  ret void
}

declare void @bar(ptr sret(%struct.test_struct), i32) #0

attributes #0 = { nounwind }
