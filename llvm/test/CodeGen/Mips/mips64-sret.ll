; RUN: llc -march=mips64el -mcpu=mips64r2 -target-abi=n64 < %s | FileCheck %s

define void @foo(i32* noalias sret(i32) %agg.result) nounwind {
entry:
; CHECK-LABEL: foo:
; CHECK: sw {{.*}}, 0($4)
; CHECK: jr $ra
; CHECK-NEXT: move $2, $4

  store i32 42, i32* %agg.result
  ret void
}

define void @bar(i32 signext %v, i32* noalias sret(i32) %agg.result) nounwind {
entry:
; CHECK-LABEL: bar:
; CHECK: sw $4, 0($5)
; CHECK: jr $ra
; CHECK-NEXT: move $2, $5

  store i32 %v, i32* %agg.result
  ret void
}
