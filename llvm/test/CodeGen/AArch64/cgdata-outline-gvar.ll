; This test verifies that global variables are hashed based on their initial contents,
; allowing them to be outlined even if they appear different due to their names.

; RUN: split-file %s %t

; Check if the outlined function is created locally.
; RUN: llc -mtriple=arm64-apple-darwin -enable-machine-outliner -codegen-data-generate=true  -aarch64-enable-collect-loh=false -filetype=obj %t/local-two.ll -o %t_write_base
; RUN: llvm-objdump -d %t_write_base | FileCheck %s

; RUN: llvm-cgdata --merge %t_write_base -o %t_cgdata_base

; Read the cgdata in the machine outliner for optimistically outlining in local-one.ll.
; RUN: llc -mtriple=arm64-apple-darwin -enable-machine-outliner -codegen-data-use-path=%t_cgdata_base  -aarch64-enable-collect-loh=false -append-content-hash-outlined-name=false -filetype=obj %t/local-one.ll -o %t_read_base
; RUN: llvm-objdump -d %t_read_base | FileCheck %s

; The names of globals `.str` and `.str.4` are different, but their initial contents are identical.
; The outlined function now starts with a reference to that global ("hello\00").
; CHECK: _OUTLINED_FUNCTION_{{.*}}:
; CHECK-NEXT: adrp x1
; CHECK-NEXT: add x1, x1
; CHECK-NEXT: mov w2
; CHECK-NEXT: mov w3
; CHECK-NEXT: mov w4
; CHECK-NEXT: b

;--- local-two.ll
@.str = private unnamed_addr constant [6 x i8] c"hello\00", align 1
@.str.1 = private unnamed_addr constant [3 x i8] c"f1\00", align 1
@.str.2 = private unnamed_addr constant [3 x i8] c"f2\00", align 1

declare noundef i32 @goo(ptr noundef, ptr noundef, i32, i32, i32)
define i32 @f1() minsize {
entry:
  %call = tail call noundef i32 @goo(ptr noundef nonnull @.str.1, ptr noundef nonnull @.str, i32 1, i32 2, i32 3)
  ret i32 %call
}
define i32 @f2() minsize {
entry:
  %call = tail call noundef i32 @goo(ptr noundef nonnull @.str.2, ptr noundef nonnull @.str, i32 1, i32 2, i32 3)
  ret i32 %call
}

;--- local-one.ll
@.str.3 = private unnamed_addr constant [3 x i8] c"f3\00", align 1
@.str.4 = private unnamed_addr constant [6 x i8] c"hello\00", align 1

declare noundef i32 @goo(ptr noundef, ptr noundef, i32, i32, i32)
define i32 @f1() minsize {
entry:
  %call = tail call noundef i32 @goo(ptr noundef nonnull @.str.3, ptr noundef nonnull @.str.4, i32 1, i32 2, i32 3)
  ret i32 %call
}
