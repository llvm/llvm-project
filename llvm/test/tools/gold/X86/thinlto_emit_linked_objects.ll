; First generate bitcode with a module summary index for each file
; RUN: opt -module-summary %s -o %t.o
; RUN: opt -module-summary %p/Inputs/thinlto_emit_linked_objects.ll -o %t2.o

; Next do the ThinLink step, specifying thinlto-index-only so that the gold
; plugin exits after generating individual indexes. The objects the linker
; decided to include in the link should be emitted into the file specified
; after 'thinlto-index-only='. Note that in this test both files should
; be included in the link, but in a case where there was an object in
; a library that had no strongly referenced symbols, that file would not
; be included in the link and listed in the emitted file. However, this
; requires gold version 1.12.
; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    --plugin-opt=thinlto \
; RUN:    --plugin-opt=thinlto-index-only=%t3 \
; RUN:    -o %t5 \
; RUN:    %t.o \
; RUN:    --start-lib %t2.o --end-lib

; RUN: cat %t3 | FileCheck %s
; CHECK: thinlto_emit_linked_objects.ll.tmp.o
; CHECK: thinlto_emit_linked_objects.ll.tmp2.o

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main() {
entry:
  call void (...) @foo()
  ret i32 0
}

declare void @foo(...)
