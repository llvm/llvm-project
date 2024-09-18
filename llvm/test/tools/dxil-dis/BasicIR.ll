; RUN: llc --filetype=obj %s -o - | dxil-dis -o - | FileCheck %s

; RUN: llc --filetype=obj %s --stop-after=dxil-write-bitcode -o %t && llvm-bcanalyzer --dump-blockinfo %t | FileCheck %s  --check-prefix=BLOCK_INFO

; CHECK: define internal i32 @foo(i32 %X, i32 %Y) {
; CHECK:   %Z = sub i32 %X, %Y
; CHECK:   %Q = add i32 %Z, %Y
; CHECK:   ret i32 %Q
; CHECK: }

; BLOCK_INFO:Stream type: LLVM IR
; Make sure uselist strtab and symtab is not in dxil.
; BLOCK_INFO-NOT:USELIST_BLOCK_ID
; BLOCK_INFO-NOT:STRTAB_BLOCK
; BLOCK_INFO-NOT:SYMTAB_BLOCK


target triple = "dxil-unknown-shadermodel6.7-library"

define i32 @foo(i32 %X, i32 %Y) {
  %Z = sub i32 %X, %Y
  %Q = add i32 %Z, %Y
  ret i32 %Q
}
