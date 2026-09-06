; Verify that GFM source-qualifies name-based hashes of module-local globals
; while leaving content-based hashes alone.

; RUN: rm -rf %t && split-file %s %t
; RUN: llc -mtriple=arm64-apple-darwin -enable-global-merge-func=true \
; RUN:   -codegen-data-generate=true -filetype=obj %t/a.ll -o %t/a.o
; RUN: llc -mtriple=arm64-apple-darwin -enable-global-merge-func=true \
; RUN:   -codegen-data-generate=true -filetype=obj %t/b.ll -o %t/b.o
; RUN: llvm-cgdata --merge %t/a.o %t/b.o -o %t/merged.cgdata
; RUN: llc -mtriple=arm64-apple-darwin -enable-global-merge-func=true \
; RUN:   -codegen-data-use-path=%t/merged.cgdata %t/a.ll -o - | FileCheck %s

; The string stays a shared constant in the merging instance, while the local
; global becomes a parameter materialized by the thunk.
; CHECK-LABEL: _fa.Tgm:
; CHECK: adrp x0, l_.str@PAGE
; CHECK: bl _consume
; CHECK-LABEL: _fa:
; CHECK: adrp x0, _cacheLock@PAGE+4
; CHECK: b _fa.Tgm

;--- a.ll
source_filename = "a.c"
target triple = "arm64-apple-darwin"

@cacheLock = internal global [2 x i32] [i32 1, i32 2], align 4
@.str = private unnamed_addr constant [6 x i8] c"hello\00", align 1

declare i32 @consume(ptr, i32)

define i32 @fa() {
entry:
  %value = load volatile i32, ptr getelementptr inbounds ([2 x i32], ptr @cacheLock, i64 0, i64 1), align 4
  %call = call i32 @consume(ptr @.str, i32 %value)
  %a = mul i32 %call, 3
  %b = add i32 %a, 5
  %c = xor i32 %b, 7
  %d = mul i32 %c, 11
  ret i32 %d
}

;--- b.ll
source_filename = "b.c"
target triple = "arm64-apple-darwin"

@cacheLock = internal global [2 x i32] [i32 1, i32 2], align 4
@.str.1 = private unnamed_addr constant [6 x i8] c"hello\00", align 1

declare i32 @consume(ptr, i32)

define i32 @fb() {
entry:
  %value = load volatile i32, ptr getelementptr inbounds ([2 x i32], ptr @cacheLock, i64 0, i64 1), align 4
  %call = call i32 @consume(ptr @.str.1, i32 %value)
  %a = mul i32 %call, 3
  %b = add i32 %a, 5
  %c = xor i32 %b, 7
  %d = mul i32 %c, 11
  ret i32 %d
}
