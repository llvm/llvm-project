; RUN: not llubi < %s 2>&1 | FileCheck %s

@value = global i32 0
@aggregate = global [1 x ptr] [ptr getelementptr (i32, ptr @value, i64 1)]

define void @main() {
  ret void
}

; CHECK: error: Failed to initialize global values
