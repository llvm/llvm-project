; RUN: not --crash llc < %s -march=nvptx -mcpu=sm_20 2>&1 | FileCheck %s

%struct.A = type { [0 x float] }
%struct.B = type { i32, i32 }

; CHECK: ERROR: Empty parameter types are not supported
define void @kernel(%struct.A %a, %struct.B %b) {
entry:
  ret void
}
