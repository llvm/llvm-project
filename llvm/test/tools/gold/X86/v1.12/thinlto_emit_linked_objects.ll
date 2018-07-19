; First generate bitcode with a module summary index for each file
; RUN: opt -module-summary %s -o %t.o
; RUN: opt -module-summary %p/Inputs/thinlto_emit_linked_objects.ll -o %t2.o

; Next do the ThinLink step, specifying thinlto-index-only so that the gold
; plugin exits after generating individual indexes. The objects the linker
; decided to include in the link should be emitted into the file specified
; after 'thinlto-index-only='. In this version of the test, only %t.o will
; be included in the link, and not %t2.o since it is within
; a library (--start-lib/--end-lib pair) and not strongly referenced.
; Note that the support for detecting this is in gold v1.12.
; RUN: rm -f %t.o.thinlto.bc
; RUN: rm -f %t2.o.thinlto.bc
; RUN: rm -f %t.o.imports
; RUN: rm -f %t2.o.imports
; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    --plugin-opt=thinlto \
; RUN:    --plugin-opt=thinlto-index-only=%t3 \
; RUN:    --plugin-opt=thinlto-emit-imports-files \
; RUN:    -m elf_x86_64 \
; RUN:    -o %t4 \
; RUN:    %t.o \
; RUN:    --start-lib %t2.o --end-lib

; Ensure that the expected output files are created, even for the file
; the linker decided not to include in the link.
; RUN: ls %t.o.thinlto.bc
; RUN: ls %t2.o.thinlto.bc
; RUN: ls %t.o.imports
; RUN: ls %t2.o.imports

; RUN: cat %t3 | FileCheck %s
; CHECK: thinlto_emit_linked_objects.ll.tmp.o
; CHECK-NOT: thinlto_emit_linked_objects.ll.tmp2.o

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @main() {
entry:
  ret i32 0
}
