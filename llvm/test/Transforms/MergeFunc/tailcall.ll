; RUN: opt -mergefunc -S < %s | FileCheck %s

declare void @dummy()

; CHECK-LABEL: define{{.*}}@foo
; CHECK: call {{.*}}@dummy
; CHECK: musttail {{.*}}@dummy
define void @foo() {
  call void @dummy()
  musttail call void @dummy()
  ret void
}

; CHECK-LABEL: define{{.*}}@bar
; CHECK: call {{.*}}@dummy
; CHECK: call {{.*}}@dummy
define void @bar() {
  call void @dummy()
  call void @dummy()
  ret void
}
