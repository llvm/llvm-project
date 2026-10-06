; RUN: llc --mtriple=loongarch32 --frame-pointer=all < %s | FileCheck %s
; RUN: llc --mtriple=loongarch64 --frame-pointer=all < %s | FileCheck %s

; CHECK-LABEL: foo:
; CHECK: addi.{{w|d}}	[[REG1:\$[a-z0-9]+]], [[REG1]], 1
; CHECK: st.w	[[REG1]], {{\$[a-z0-9]+}}, {{[0-9]+}}
; CHECK: ld.{{w|d}}	$fp, $sp, {{[0-9]+}}
define void @foo() nounwind sanitize_address {
entry:
  %1 = load ptr, ptr inttoptr (i32 64 to ptr), align 64
  %2 = load i32, ptr %1, align 8
  %3 = add nsw i32 %2, 1
  store i32 %3, ptr %1, align 8
  ret void
}
