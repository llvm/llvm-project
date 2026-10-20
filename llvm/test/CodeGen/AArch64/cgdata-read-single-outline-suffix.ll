; This test checks if a singleton instance (an instance that appears only once) can be outlined
; using codegen data from a previous codegen run.
; Unlike cgdata-read-single-outline.ll, this test also examines various suffixes that LLVM appends to names.
; Specifically, we aim to disregard the suffixes `.llvm.{number}` and `.__uniq.{number}` during the matching of call targets in hash computations.
; This approach helps in accurately identifying the original call target, especially when an LTO build may append additional suffixes for uniqueness.
; Conversely, we only consider the number from the suffix `.content.{number}`.
; This matching strategy is crucial for recursively finding outlining candidates when multiple outliner runs are enabled.

; RUN: split-file %s %t

; First, we generate the cgdata file from a local outline instance present in local-two.ll.
; RUN: llc -mtriple=arm64-apple-darwin -enable-machine-outliner -codegen-data-generate=true -filetype=obj %t/local-two.ll -o %t_write
; RUN: llvm-cgdata --merge %t_write -o %t_cgdata
; RUN: llvm-cgdata --show %t_cgdata | FileCheck %s --check-prefix=SHOW

; SHOW: Outlined hash tree:
; SHOW-NEXT:  Total Node Count: 4
; SHOW-NEXT:  Terminal Node Count: 1
; SHOW-NEXT:  Depth: 3

; Now, we read the cgdata in the machine outliner, enabling us to optimistically
; outline a singleton instance in local-one.ll that matches against the cgdata.
; We outline instances while disregarding the suffixes `.llvm.{number}` or `.__uniq.{number}` in names.
; RUN: llc -mtriple=arm64-apple-darwin -enable-machine-outliner -codegen-data-use-path=%t_cgdata -filetype=obj %t/local-one-ignore-suffix-1.ll -o %t_read_ignore_1
; RUN: llvm-objdump -d %t_read_ignore_1 | FileCheck %s
; RUN: llc -mtriple=arm64-apple-darwin -enable-machine-outliner -codegen-data-use-path=%t_cgdata -filetype=obj %t/local-one-ignore-suffix-2.ll -o %t_read_ignore_2
; RUN: llvm-objdump -d %t_read_ignore_2 | FileCheck %s

; CHECK: _OUTLINED_FUNCTION
; CHECK-NEXT:  mov
; CHECK-NEXT:  mov
; CHECK-NEXT:  b

; We don't ignore `.invalid.{number}`. So no outlining occurs.
; RUN: llc -mtriple=arm64-apple-darwin -enable-machine-outliner -codegen-data-use-path=%t_cgdata -filetype=obj %t/local-one-no-ignore-suffix.ll -o %t_read_no_ignore
; RUN: llvm-objdump -d %t_read_no_ignore | FileCheck %s --check-prefix=NOOUTLINE

; NOOUTLINE-NOT: _OUTLINED_FUNCTION

;--- local-two.ll
declare i32 @g(i32, i32, i32)
define i32 @f1() minsize {
  %1 = call i32 @g(i32 10, i32 1, i32 2);
  ret i32 %1
}
define i32 @f2() minsize {
  %1 = call i32 @g(i32 20, i32 1, i32 2);
  ret i32 %1
}

;--- local-one-ignore-suffix-1.ll
declare i32 @g.llvm.123(i32, i32, i32)
define i32 @f3() minsize {
  %1 = call i32 @g.llvm.123(i32 30, i32 1, i32 2);
 ret i32 %1
}

;--- local-one-ignore-suffix-2.ll
declare i32 @g.__uniq.456(i32, i32, i32)
define i32 @f4() minsize {
  %1 = call i32 @g.__uniq.456(i32 30, i32 1, i32 2);
 ret i32 %1
}

;--- local-one-no-ignore-suffix.ll
declare i32 @g.invalid.789(i32, i32, i32)
define i32 @f5() minsize {
  %1 = call i32 @g.invalid.789(i32 30, i32 1, i32 2);
 ret i32 %1
}

; Similarly, we outline functions that have already been processed in previous outliner runs.
; Assuming `-machine-outliner-reruns` is locally enabled, we might already have `OUTLINED_FUNCTION*` instances.
; First, we generate the cgdata file from a local outline instance found in local-two-content.ll.
; RUN: llc -mtriple=arm64-apple-darwin -enable-machine-outliner -codegen-data-generate=true -filetype=obj %t/local-two-content.ll -o %t_write_content
; RUN: llvm-cgdata --merge %t_write_content -o %t_cgdata_content
; RUN: llvm-cgdata --show %t_cgdata_content | FileCheck %s --check-prefix=SHOW

; Despite the target function names being different -- `OUTLINED_FUNCTION_0.content.123` vs. `OUTLINED_FUNCTION_1.content.123`,
; We compute the same hash based on the suffix `.content.{number}`, and optimistically outline them.
; RUN: llc -mtriple=arm64-apple-darwin -enable-machine-outliner -codegen-data-use-path=%t_cgdata_content -filetype=obj %t/local-one-content.ll -o %t_read_content
; RUN: llvm-objdump -d %t_read_content | FileCheck %s

;--- local-two-content.ll
declare i32 @OUTLINED_FUNCTION_0.content.123(i32, i32, i32)
define i32 @f6() minsize {
  %1 = call i32 @OUTLINED_FUNCTION_0.content.123(i32 10, i32 1, i32 2);
  ret i32 %1
}
define i32 @f7() minsize {
  %1 = call i32 @OUTLINED_FUNCTION_0.content.123(i32 20, i32 1, i32 2);
  ret i32 %1
}

;--- local-one-content.ll
declare i32 @OUTLINED_FUNCTION_1.content.123(i32, i32, i32)
define i32 @f8() minsize {
  %1 = call i32 @OUTLINED_FUNCTION_1.content.123(i32 30, i32 1, i32 2);
 ret i32 %1
}
