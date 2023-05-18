; RUN: not opt -S -passes=verify < %s 2>&1 | FileCheck %s

%struct.test = type { <vscale x 1 x double>, <vscale x 1 x double> }

define void @gep(ptr %a) {
; CHECK: error: getelementptr cannot target structure that contains scalable vector type
  %a.addr = getelementptr %struct.test, ptr %a, i32 0
  ret void
}
