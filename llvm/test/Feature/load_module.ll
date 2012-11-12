; PR1318
; RUN: opt < %s -load=%llvmshlibdir/LLVMHello%shlibext -hello \
; RUN:   -disable-output 2>&1 | grep Hello
; REQUIRES: loadable_module
; FIXME: On Cygming, it might fail without building LLVMHello manually.

@junk = global i32 0

define i32* @somefunk() {
  ret i32* @junk
}

