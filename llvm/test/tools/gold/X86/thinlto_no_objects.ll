; Check that thinlto-index-only= always creates linked objects file, even
; if nothing to add there.

; Non-ThinLTO file should not get into list of linked objects.
; RUN: opt %s -o %t.o

; RUN: rm -f %t3
; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    --plugin-opt=thinlto \
; RUN:    --plugin-opt=thinlto-index-only=%t3 \
; RUN:    -o %t5 \
; RUN:    %t.o

; RUN: cat %t3 | count 0

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

