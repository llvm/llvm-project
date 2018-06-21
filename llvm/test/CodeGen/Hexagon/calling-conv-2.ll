; RUN: llc -march=hexagon -mcpu=hexagonv5 <%s | \
; RUN:   FileCheck %s --check-prefix=CHECK-ONE

%struct.test_struct = type { i32, i8, i64 }

; CHECK-ONE:    r1 = #45
define void @foo(%struct.test_struct* noalias nocapture sret %agg.result, i32 %a) #0 {
entry:
  call void @bar(%struct.test_struct* sret %agg.result, i32 45) #2
  ret void
}

declare void @bar(%struct.test_struct* sret, i32) #1
