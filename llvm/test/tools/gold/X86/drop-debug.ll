; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    --plugin-opt=emit-llvm -shared %p/Inputs/drop-debug.bc \
; RUN:    -o t2.bc 2>&1 | FileCheck %s

; drop-debug.bc was created from "void f(void) {}" with clang 3.5 and
; -gline-tables-only, so it contains old debug info.

; CHECK: warning: LLVM gold plugin: ignoring debug info with an invalid version (1) in {{.*}}/Inputs/drop-debug.bc
