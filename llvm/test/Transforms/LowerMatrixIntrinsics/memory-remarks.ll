; RUN: opt -enable-matrix=true -passes=lower-matrix-intrinsics -pass-remarks-analysis=lower-matrix-intrinsics %s -o /dev/null 2>&1 | FileCheck %s

; CHECK-LABEL: remark: <unknown>:0:0: Store.
; CHECK: Store size: 128 bytes.
define void @multiply(ptr %A, ptr %B, ptr %C) {
  %A.matrix = load <16 x double>, ptr %A
  %B.matrix = load <16 x double>, ptr %B
  %t = call <16 x double> @llvm.matrix.multiply(<16 x double> %A.matrix, <16 x double> %B.matrix, i32 4, i32 4, i32 4)
  store <16 x double> %t, ptr %C
  ret void
}
