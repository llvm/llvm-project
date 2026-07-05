; RUN: llc -mtriple=arm64-apple-ios -mcpu=apple-latest -stop-before=finalize-isel -o - 2>&1 < %s | FileCheck %s

; CHECK-LABEL: @dummy
; CHECK: "target-cpu"="apple-latest"
define void @dummy() {
  ret void
}
