; RUN: opt -passes=def-use-instrumentation -disable-output %s

define i32 @main() {
entry:
  ret i32 0
}