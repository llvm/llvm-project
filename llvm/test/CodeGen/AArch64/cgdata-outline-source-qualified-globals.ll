; Verify that GFO source-qualifies name-based hashes of module-local globals
; while leaving content-based hashes alone.

; RUN: split-file %s %t
; RUN: llc -mtriple=arm64-apple-darwin -enable-machine-outliner -codegen-data-generate \
; RUN:   -filetype=obj %t/a.ll -o %t/a.o
; RUN: llvm-cgdata --merge %t/a.o -o %t/merged.cgdata
; RUN: llc -mtriple=arm64-apple-darwin -enable-machine-outliner -codegen-data-use-path=%t/merged.cgdata \
; RUN:   %t/b.ll -o - | FileCheck %s

; a.ll records two outlined sequences from source a.c: one referencing the
; internal global @cache (hashed by name) and one referencing the private
; string @.str.4 (hashed by content). In source b.c, the singleton f5
; referencing a same-named but distinct @cache must not match the hash tree,
; while the otherwise identically shaped singleton f6 referencing a
; same-content string with a different symbol name is outlined.
; CHECK-LABEL: _f5:
; CHECK-NOT:  _OUTLINED_FUNCTION
; CHECK:      adrp x1, _cache@PAGE
; CHECK:      b _goo
; CHECK-LABEL: _f6:
; CHECK:      b _OUTLINED_FUNCTION
; CHECK-LABEL: _OUTLINED_FUNCTION_{{.*}}:
; CHECK:      adrp x1, l_.str.9@PAGE
; CHECK-NEXT: add x1, x1, l_.str.9@PAGEOFF
; CHECK:      b _hoo

;--- a.ll
source_filename = "a.c"
@.str.1 = private unnamed_addr constant [3 x i8] c"f1\00", align 1
@.str.2 = private unnamed_addr constant [3 x i8] c"f2\00", align 1
@.str.3 = private unnamed_addr constant [3 x i8] c"f3\00", align 1
@.str.4 = private unnamed_addr constant [6 x i8] c"hello\00", align 1
@cache = internal global [4 x i32] zeroinitializer, align 4

declare noundef i32 @goo(ptr noundef, ptr noundef, i32, i32, i32)
declare noundef i32 @hoo(ptr noundef, ptr noundef, i32, i32, i32)

define i32 @f1() minsize {
entry:
  %call = tail call noundef i32 @goo(ptr noundef nonnull @.str.1, ptr noundef nonnull @cache, i32 1, i32 2, i32 3)
  ret i32 %call
}
define i32 @f2() minsize {
entry:
  %call = tail call noundef i32 @goo(ptr noundef nonnull @.str.2, ptr noundef nonnull @cache, i32 1, i32 2, i32 3)
  ret i32 %call
}
define i32 @f3() minsize {
entry:
  %call = tail call noundef i32 @hoo(ptr noundef nonnull @.str.1, ptr noundef nonnull @.str.4, i32 1, i32 2, i32 3)
  ret i32 %call
}
define i32 @f4() minsize {
entry:
  %call = tail call noundef i32 @hoo(ptr noundef nonnull @.str.3, ptr noundef nonnull @.str.4, i32 1, i32 2, i32 3)
  ret i32 %call
}

;--- b.ll
source_filename = "b.c"
@.str.5 = private unnamed_addr constant [3 x i8] c"f5\00", align 1
@.str.6 = private unnamed_addr constant [3 x i8] c"f6\00", align 1
@.str.9 = private unnamed_addr constant [6 x i8] c"hello\00", align 1
@cache = internal global [4 x i32] zeroinitializer, align 4

declare noundef i32 @goo(ptr noundef, ptr noundef, i32, i32, i32)
declare noundef i32 @hoo(ptr noundef, ptr noundef, i32, i32, i32)

define i32 @f5() minsize {
entry:
  %call = tail call noundef i32 @goo(ptr noundef nonnull @.str.5, ptr noundef nonnull @cache, i32 1, i32 2, i32 3)
  ret i32 %call
}
define i32 @f6() minsize {
entry:
  %call = tail call noundef i32 @hoo(ptr noundef nonnull @.str.6, ptr noundef nonnull @.str.9, i32 1, i32 2, i32 3)
  ret i32 %call
}
