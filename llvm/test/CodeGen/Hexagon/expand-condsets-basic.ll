; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: if{{.*}}add
; CHECK: if{{.*}}sub

define i32 @foo (i1 %a, i32 %b, i32 %c, i32 %d) nounwind {
  %1 = add i32 %b, %d
  %2 = sub i32 %c, %d
  %3 = select i1 %a, i32 %1, i32 %2
  ret i32 %3
}

