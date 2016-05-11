; Generate summary sections and test gold handling.
; RUN: opt -module-summary %s -o %t.o
; RUN: opt -module-summary %p/Inputs/thinlto.ll -o %t2.o

; Ensure gold generates imports files if requested for distributed backends.
; RUN: %gold -plugin %llvmshlibdir/LLVMgold.so \
; RUN:    --plugin-opt=thinlto \
; RUN:    --plugin-opt=thinlto-index-only \
; RUN:    --plugin-opt=thinlto-emit-imports-files \
; RUN:    -shared %t.o %t2.o -o %t3

; The imports file for this module contains the bitcode file for
; Inputs/thinlto.ll
; RUN: cat %t.o.imports | count 1
; RUN: cat %t.o.imports | FileCheck %s --check-prefix=IMPORTS1
; IMPORTS1: test/tools/gold/X86/Output/thinlto_emit_imports.ll.tmp2.o

; The imports file for Input/thinlto.ll is empty as it does not import anything.
; RUN: cat %t2.o.imports | count 0

declare void @g(...)

define void @f() {
entry:
  call void (...) @g()
  ret void
}
