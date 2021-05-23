; RUN: split-file %s %t
; RUN: not llvm-link %t/a.ll %t/b.ll 2>&1 | FileCheck --check-prefix=CHECK-KIND %s
; RUN: not llvm-link %t/c.ll %t/d.ll 2>&1 | FileCheck --check-prefix=CHECK-REG %s
; RUN: not llvm-link %t/e.ll %t/f.ll 2>&1 | FileCheck --check-prefix=CHECK-OFFSET %s
; RUN: llvm-link %t/g.ll %t/h.ll

; CHECK-KIND: error: linking module flags 'stack-protector-guard': IDs have conflicting values
; CHECK-REG: error: linking module flags 'stack-protector-guard-reg': IDs have conflicting values
; CHECK-OFFSET: error: linking module flags 'stack-protector-guard-offset': IDs have conflicting values

;--- a.ll
; Test that different values of stack-protector-guard fail.
define void @foo() sspstrong {
  ret void
}
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"stack-protector-guard", !"sysreg"}
;--- b.ll
declare void @foo() sspstrong
define void @bar() sspstrong {
  call void @foo()
  ret void
}
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"stack-protector-guard", !"global"}

;--- c.ll
; Test that different values of stack-protector-guard-reg fail.
define void @foo() sspstrong {
  ret void
}
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"stack-protector-guard-reg", !"sp_el0"}
;--- d.ll
declare void @foo() sspstrong
define void @bar() sspstrong {
  call void @foo()
  ret void
}
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"stack-protector-guard-reg", !"sp_el1"}

;--- e.ll
; Test that different values of stack-protector-guard-offset fail.
define void @foo() sspstrong {
  ret void
}
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"stack-protector-guard-offset", i32 257}
;--- f.ll
declare void @foo() sspstrong
define void @bar() sspstrong {
  call void @foo()
  ret void
}
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"stack-protector-guard-offset", i32 256}

;--- g.ll
; Test that the same values for the three module attributes succeed.
define void @foo() sspstrong {
  ret void
}
!llvm.module.flags = !{!0, !1, !2}
!0 = !{i32 1, !"stack-protector-guard", !"sysreg"}
!1 = !{i32 1, !"stack-protector-guard-reg", !"sp_el0"}
!2 = !{i32 1, !"stack-protector-guard-offset", i32 257}
;--- h.ll
declare void @foo() sspstrong
define void @bar() sspstrong {
  call void @foo()
  ret void
}
!llvm.module.flags = !{!0, !1, !2}
!0 = !{i32 1, !"stack-protector-guard", !"sysreg"}
!1 = !{i32 1, !"stack-protector-guard-reg", !"sp_el0"}
!2 = !{i32 1, !"stack-protector-guard-offset", i32 257}
