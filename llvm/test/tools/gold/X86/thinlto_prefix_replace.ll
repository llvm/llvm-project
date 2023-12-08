; Check that changing the output path via thinlto-prefix-replace works
; RUN: mkdir -p %t/oldpath
; RUN: opt -module-summary %s -o %t/oldpath/thinlto_prefix_replace.o
; Ensure that there is no existing file at the new path, so we properly
; test the creation of the new file there.
; RUN: rm -f %t/newpath/thinlto_prefix_replace.o.thinlto.bc
; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    --plugin-opt=thinlto \
; RUN:    --plugin-opt=thinlto-index-only \
; RUN:    --plugin-opt=thinlto-prefix-replace="%t/oldpath/;%t/newpath/" \
; RUN:    -shared %t/oldpath/thinlto_prefix_replace.o -o %t/thinlto_prefix_replace
; RUN: ls %t/newpath/thinlto_prefix_replace.o.thinlto.bc

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f() {
entry:
  ret void
}
