; UNSUPPORTED: system-zos

; Test malformed outer OffloadBinary is handled gracefully.
; RUN: printf "\020\377\020\255\012" > %t
; RUN: not llvm-offload-binary %t 2>&1 | FileCheck --check-prefix=MALFORMED-OUTER %s

; MALFORMED-OUTER: llvm-offload-binary: error: Invalid data was encountered while parsing the file

; Test malformed inner OffloadBinary is handled gracefully.
; RUN: printf "\020\377\020\255\012" > %t.malformed
; RUN: llvm-offload-binary -o %t.nested --image=file=%t.malformed,arch=nested,triple=x-y-z
; RUN: not llvm-offload-binary %t.nested 2>&1 | FileCheck --check-prefix=MALFORMED-INNER %s

; MALFORMED-INNER: llvm-offload-binary: error: Invalid data was encountered while parsing the file
