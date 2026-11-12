; RUN: llc -asm-verbose=false < %s -wasm-keep-registers | FileCheck %s

; Test that 128-bit smul.with.overflow assembles as expected.

target triple = "wasm32-unknown-unknown"

define i128 @call_muloti4(i128 %a, i128 %b) nounwind {
entry:
  %smul = tail call { i128, i1 } @llvm.smul.with.overflow.i128(i128 %a, i128 %b)
  %cmp = extractvalue { i128, i1 } %smul, 1
  %smul.result = extractvalue { i128, i1 } %smul, 0
  %X = select i1 %cmp, i128 %smul.result, i128 42
  ret i128 %X
}

; CHECK: call __muloti4, $pop{{[0-9]*}}, $pop{{[0-9]*}}, $pop{{[0-9]*}}, $pop{{[0-9]*}}, $pop{{[0-9]*}}, $pop{{[0-9]*}}{{$}}

declare { i128, i1 } @llvm.smul.with.overflow.i128(i128, i128) nounwind readnone
