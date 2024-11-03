; RUN: llvm-link -S -o - %p/pr22807.ll %p/Inputs/pr22807-1.ll %p/Inputs/pr22807-2.ll | FileCheck %s

; CHECK-NOT: type
; CHECK: %struct.B = type { %struct.A }
; CHECK-NEXT: %struct.A = type { %struct.B }
; CHECK-NOT: type

%struct.B = type { %struct.A }
%struct.A = type opaque

define i32 @baz(%struct.B %BB) {
  ret i32 0
}
