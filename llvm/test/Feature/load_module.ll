; PR1318
; RUN: llvm-as < %s | opt -load=%llvmlibsdir/LLVMHello%shlibext -hello \
; RUN:   -disable-output - |& grep Hello

@junk = global i32 0

define i32* @somefunk() {
  ret i32* @junk
}

