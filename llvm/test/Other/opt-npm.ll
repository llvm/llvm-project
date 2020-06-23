; RUN: opt -dce -enable-new-pm -disable-output -debug-pass-manager %s 2>&1 | FileCheck %s

; CHECK: DCEPass
define void @foo() {
    ret void
}
