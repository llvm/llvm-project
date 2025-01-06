; This test verifies the globally outlined function name has the content hash.

; RUN: split-file %s %t

; Check if the outlined function name has the content hash depending the flag.
; RUN: llc -mtriple=arm64-apple-darwin -enable-machine-outliner -codegen-data-generate=true -append-content-hash-outlined-name=false -filetype=obj %t/local-two.ll -o %t_write_base
; RUN: llvm-objdump -d %t_write_base | FileCheck %s --check-prefix=BASE
; RUN: llc -mtriple=arm64-apple-darwin -enable-machine-outliner -codegen-data-generate=true -append-content-hash-outlined-name=true -filetype=obj %t/local-two.ll -o %t_write_suffix
; RUN: llvm-objdump -d %t_write_suffix | FileCheck %s --check-prefix=SUFFIX
; BASE-NOT: _OUTLINED_FUNCTION_{{.*}}.content.{{[0-9]+}}
; SUFFIX: _OUTLINED_FUNCTION_{{.*}}.content.{{[0-9]+}}

; Generate the cgdata file from each case and show they are identical.
; RUN: llvm-cgdata --merge %t_write_base -o %t_cgdata_base
; RUN: llvm-cgdata --merge %t_write_suffix -o %t_cgdata_suffix
; RUN: diff %t_cgdata_base %t_cgdata_suffix

; Read the cgdata in the machine outliner for optimistically outlining in local-one.ll.
; Check if the outlined function has the content hash depending the flag.
; RUN: llc -mtriple=arm64-apple-darwin -enable-machine-outliner -codegen-data-use-path=%t_cgdata_base -append-content-hash-outlined-name=false -filetype=obj %t/local-one.ll -o %t_read_base
; RUN: llvm-objdump -d %t_read_base | FileCheck %s --check-prefix=BASE
; RUN: llc -mtriple=arm64-apple-darwin -enable-machine-outliner -codegen-data-use-path=%t_cgdata_suffix -append-content-hash-outlined-name=true -filetype=obj %t/local-one.ll -o %t_read_suffix
; RUN: llvm-objdump -d %t_read_suffix | FileCheck %s --check-prefix=SUFFIX

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

;--- local-one.ll
declare i32 @g(i32, i32, i32)
define i32 @f3() minsize {
  %1 = call i32 @g(i32 30, i32 1, i32 2);
 ret i32 %1
}
