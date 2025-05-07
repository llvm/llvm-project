; This test verifies that global variables are hashed based on their initial contents,
; allowing them to be outlined even if they appear different due to their names.

; RUN: split-file %s %t

; The outlined function is created locally.
; Note that `.str.3` is commonly used in both `f1()` and `f2()`.
; RUN: llc -mtriple=arm64-apple-darwin -enable-machine-outliner -codegen-data-generate -aarch64-enable-collect-loh=false  \
; RUN:   %t/local-two.ll -o -  | FileCheck %s --check-prefix=WRITE

; WRITE-LABEL: _OUTLINED_FUNCTION_{{.*}}:
; WRITE:      adrp x1, l_.str.3
; WRITE-NEXT: add x1, x1, l_.str.3
; WRITE-NEXT: mov w2
; WRITE-NEXT: mov w3
; WRITE-NEXT: mov w4
; WRITE-NEXT: b

; Create an object file and merge it into the cgdata.
; RUN: llc -mtriple=arm64-apple-darwin -enable-machine-outliner -codegen-data-generate -aarch64-enable-collect-loh=false \
; RUN:   -filetype=obj %t/local-two.ll -o %t_write_base
; RUN: llvm-cgdata --merge %t_write_base -o %t_cgdata_base

; Read the cgdata in the machine outliner for optimistically outlining in local-one.ll.
; Note that the hash of `.str.5` in local-one.ll matches that of `.str.3` in an outlined tree in the cgdata.

; RUN: llc -mtriple=arm64-apple-darwin -enable-machine-outliner -codegen-data-use-path=%t_cgdata_base  -aarch64-enable-collect-loh=false \
; RUN:   %t/local-one.ll -o -  | FileCheck %s --check-prefix=READ

; READ-LABEL: _OUTLINED_FUNCTION_{{.*}}:
; READ:      adrp x1, l_.str.5
; READ-NEXT: add x1, x1, l_.str.5
; READ-NEXT: mov w2
; READ-NEXT: mov w3
; READ-NEXT: mov w4
; READ-NEXT: b

;--- local-two.ll
@.str.1 = private unnamed_addr constant [3 x i8] c"f1\00", align 1
@.str.2 = private unnamed_addr constant [3 x i8] c"f2\00", align 1
@.str.3 = private unnamed_addr constant [6 x i8] c"hello\00", align 1

declare noundef i32 @goo(ptr noundef, ptr noundef, i32, i32, i32)
define i32 @f1() minsize {
entry:
  %call = tail call noundef i32 @goo(ptr noundef nonnull @.str.1, ptr noundef nonnull @.str.3, i32 1, i32 2, i32 3)
  ret i32 %call
}
define i32 @f2() minsize {
entry:
  %call = tail call noundef i32 @goo(ptr noundef nonnull @.str.2, ptr noundef nonnull @.str.3, i32 1, i32 2, i32 3)
  ret i32 %call
}

;--- local-one.ll
@.str.4 = private unnamed_addr constant [3 x i8] c"f3\00", align 1
@.str.5 = private unnamed_addr constant [6 x i8] c"hello\00", align 1

declare noundef i32 @goo(ptr noundef, ptr noundef, i32, i32, i32)
define i32 @f1() minsize {
entry:
  %call = tail call noundef i32 @goo(ptr noundef nonnull @.str.4, ptr noundef nonnull @.str.5, i32 1, i32 2, i32 3)
  ret i32 %call
}
