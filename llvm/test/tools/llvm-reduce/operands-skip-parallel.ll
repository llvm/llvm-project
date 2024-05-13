; This reuses llvm/test/tools/llvm-reduce/operands-skip.ll
; REQUIRES: thread_support

; RUN: llvm-reduce --abort-on-invalid-reduction -j 2 %S/operands-skip.ll -o %t.1 --delta-passes=operands-skip --test FileCheck --test-arg %S/operands-skip.ll --test-arg --match-full-lines --test-arg --check-prefix=INTERESTING --test-arg --input-file
; RUN: FileCheck %S/operands-skip.ll --input-file %t.1 --check-prefixes=REDUCED

; RUN: llvm-reduce --abort-on-invalid-reduction -j 4 %S/operands-skip.ll -o %t.2 --delta-passes=operands-skip --test FileCheck --test-arg %S/operands-skip.ll --test-arg --match-full-lines --test-arg --check-prefix=INTERESTING --test-arg --input-file
; RUN: FileCheck %S/operands-skip.ll --input-file %t.2 --check-prefixes=REDUCED
