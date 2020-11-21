; RUN: llc -fast-isel-sink-local-values < %s -mtriple=i686-pc-linux -mcpu=corei7 | FileCheck --check-prefix=DAG %s
; RUN: llc -fast-isel-sink-local-values < %s -mtriple=i686-pc-linux -mcpu=corei7 -O0 | FileCheck --check-prefix=FAST %s

%struct.s1 = type { double, float }

define void @g1() nounwind {
entry:
  %tmp = alloca %struct.s1, align 4
  call void @f(%struct.s1* inreg sret(%struct.s1) %tmp, i32 inreg 41, i32 inreg 42, i32 43)
  ret void
  ; DAG-LABEL: g1:
  ; DAG: subl $[[AMT:.*]], %esp
  ; DAG-NEXT: $43, (%esp)
  ; DAG-NEXT: leal    16(%esp), %eax
  ; DAG-NEXT: movl    $41, %edx
  ; DAG-NEXT: movl    $42, %ecx
  ; DAG-NEXT: calll   f
  ; DAG-NEXT: addl $[[AMT]], %esp
  ; DAG-NEXT: ret

  ; FAST-LABEL: g1:
  ; FAST: subl $[[AMT:.*]], %esp
  ; FAST-NEXT: leal    16(%esp), %eax
  ; FAST-NEXT: movl    $41, %edx
  ; FAST-NEXT: movl    $42, %ecx
  ; FAST: $43, (%esp)
  ; FAST: calll   f
  ; FAST-NEXT: addl $[[AMT]], %esp
  ; FAST: ret
}

declare void @f(%struct.s1* inreg sret(%struct.s1), i32 inreg, i32 inreg, i32)

%struct.s2 = type {}

define void @g2(%struct.s2* inreg sret(%struct.s2) %agg.result) nounwind {
entry:
  ret void
  ; DAG: g2
  ; DAG-NOT: ret $4
  ; DAG: .size g2

  ; FAST: g2
  ; FAST-NOT: ret $4
  ; FAST: .size g2
}
