; RUN: not llubi --verbose < %s 2>&1 | FileCheck %s

@g = global i8 0
define void @main() {
  %v = load [4294967296 x [0 x i8]], ptr @g
  %e = extractvalue [4294967296 x [0 x i8]] %v, 0
  ret void
}

; CHECK: The number of elements of [4294967296 x [0 x i8]] is too large!
; CHECK-NEXT: error: -: input module cannot be executed!
