; RUN: not llubi < %s 2>&1 | FileCheck %s

target datalayout = "e-p:16:16"

@g = global [32768 x i8] zeroinitializer

define void @main() {
  ret void
}
; CHECK: error: Failed to initialize global values
