; RUN: split-file %s %t
; RUN: not llc -mtriple=nvptx64 -mcpu=sm_50 < %t/main.ll 2>&1 | FileCheck %s
; RUN: not llc -mtriple=nvptx64 -mcpu=sm_50 < %t/parser-bad-ret.ll 2>&1 | FileCheck %s --check-prefix=PARSER-RET
; RUN: not llc -mtriple=nvptx64 -mcpu=sm_50 < %t/parser-bad-arg.ll 2>&1 | FileCheck %s --check-prefix=PARSER-ARG
; RUN: not llc -mtriple=nvptx64 -mcpu=sm_50 < %t/parser-bad-count.ll 2>&1 | FileCheck %s --check-prefix=PARSER-COUNT

; CHECK: Intrinsic has incorrect return type!
; CHECK-NEXT: declared return type is 'i16', expected 'i32' in canonical signature 'i32 (i32, i32, i32)'
; CHECK: Intrinsic called with incompatible signature
; CHECK-NEXT: argument 3 type mismatch (expected i32, got i16)
; CHECK: Intrinsic called with incompatible signature
; CHECK-NEXT: wrong number of arguments (expected 3, got 4), expected signature: i64 (i64, i64, i64), got signature: i64 (i64, i64, i64, i64)

; PARSER-RET: invalid intrinsic signature
; PARSER-RET-NEXT: for 'llvm.nvvm.sad.i': got i16 (i32, i32, i32), expected i32 (i32, i32, i32)

; PARSER-ARG: invalid intrinsic signature
; PARSER-ARG-NEXT: for 'llvm.nvvm.sad.ui': got i32 (i32, i32, i16), expected i32 (i32, i32, i32)

; PARSER-COUNT: invalid intrinsic signature
; PARSER-COUNT-NEXT: for 'llvm.nvvm.sad.ull': got i64 (i64, i64, i64, i64), expected i64 (i64, i64, i64)

;--- main.ll

; Invalid return type: @llvm.nvvm.sad.i declared with i16 return instead of i32.
define i16 @test_bad_ret(i32 %x, i32 %y, i32 %z) {
  %1 = call i16 @llvm.nvvm.sad.i(i32 %x, i32 %y, i32 %z)
  ret i16 %1
}

; Invalid argument type: @llvm.nvvm.sad.ui called with i16 third arg instead of i32.
define i32 @test_bad_arg(i32 %x, i32 %y, i16 %z) {
  %1 = call i32 @llvm.nvvm.sad.ui(i32 %x, i32 %y, i16 %z)
  ret i32 %1
}

; Invalid argument count: @llvm.nvvm.sad.ull called with 4 args instead of 3.
define i64 @test_bad_arg_count(i64 %x, i64 %y, i64 %z, i64 %w) {
  %1 = call i64 @llvm.nvvm.sad.ull(i64 %x, i64 %y, i64 %z, i64 %w)
  ret i64 %1
}

declare i16 @llvm.nvvm.sad.i(i32, i32, i32)
declare i32 @llvm.nvvm.sad.ui(i32, i32, i32)
declare i64 @llvm.nvvm.sad.ull(i64, i64, i64)

;--- parser-bad-ret.ll

; Parser error: no declaration, intrinsic called with wrong return type.
define i16 @test_parser_bad_ret(i32 %x, i32 %y, i32 %z) {
  %1 = call i16 @llvm.nvvm.sad.i(i32 %x, i32 %y, i32 %z)
  ret i16 %1
}

;--- parser-bad-arg.ll

; Parser error: no declaration, intrinsic called with wrong argument type.
define i32 @test_parser_bad_arg(i32 %x, i32 %y, i16 %z) {
  %1 = call i32 @llvm.nvvm.sad.ui(i32 %x, i32 %y, i16 %z)
  ret i32 %1
}

;--- parser-bad-count.ll

; Parser error: no declaration, intrinsic called with wrong argument count.
define i64 @test_parser_bad_count(i64 %x, i64 %y, i64 %z, i64 %w) {
  %1 = call i64 @llvm.nvvm.sad.ull(i64 %x, i64 %y, i64 %z, i64 %w)
  ret i64 %1
}
