;RUN: llc -mtriple=sparc < %s | FileCheck %s

%struct.foo_t = type { i32, i32, i32 }

@s = internal unnamed_addr global %struct.foo_t { i32 10, i32 20, i32 30 }

define i32 @test() nounwind {
entry:
;CHECK-LABEL:     test:
;CHECK:     st
;CHECK:     st
;CHECK:     st
;CHECK:     bar
  %0 = tail call i32 @bar(ptr byval(%struct.foo_t) @s) nounwind
  ret i32 %0
}

declare i32 @bar(ptr byval(%struct.foo_t))
