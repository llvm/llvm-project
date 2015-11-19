; RUN: opt < %s -S -functionattrs | FileCheck %s --check-prefix=CHECK-CONTROL
; RUN: opt < %s -S -functionattrs -force-attribute foo:noinline | FileCheck %s --check-prefix=CHECK-FOO

; CHECK-CONTROL: define void @foo() #0 {
; CHECK-FOO: define void @foo() #0 {
define void @foo() {
  ret void
}


; CHECK-CONTROL: attributes #0 = { norecurse readnone }
; CHECK-FOO: attributes #0 = { noinline norecurse readnone }
