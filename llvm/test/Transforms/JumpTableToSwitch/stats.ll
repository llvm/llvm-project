; REQUIRES: asserts

; RUN: opt < %s -passes=jump-table-to-switch -stats -disable-output -jump-table-to-switch-function-size-threshold=3 2>&1 | FileCheck %s

; CHECK: 2 jump-table-to-switch - The number of jump tables seen by the pass that can be converted if deemed profitable.
; CHECK-NEXT: 1 jump-table-to-switch - The number of jump tables converted into switches.

@func_array1 = constant [2 x ptr] [ptr @func0, ptr @func1]
@func_array2 = constant [2 x ptr] [ptr @func1, ptr @func2]

define i32 @func0() {
  ret i32 0
}

define i32 @func1() {
  ret i32 1
}

define i32 @func2() {
  %a = add i32 0, 1
  %b = add i32 %a, 1
  %c = add i32 %b, 1
  %d = add i32 %c, 1
  ret i32 %d
}

define i32 @function_gets_converted(i32 %index) {
  %gep = getelementptr inbounds [2 x ptr], ptr @func_array1, i32 0, i32 %index
  %func_ptr = load ptr, ptr %gep
  %result = call i32 %func_ptr()
  ret i32 %result
}

define i32 @function_not_converted(i32 %index) {
  %gep = getelementptr inbounds [2 x ptr], ptr @func_array2, i32 0, i32 %index
  %func_ptr = load ptr, ptr %gep
  %result = call i32 %func_ptr()
  ret i32 %result
}
