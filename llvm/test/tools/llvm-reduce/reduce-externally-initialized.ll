; RUN: llvm-reduce -abort-on-invalid-reduction --delta-passes=global-values --test FileCheck --test-arg --check-prefix=INTERESTING --test-arg %s --test-arg --input-file %s -o %t.0
; RUN: FileCheck --implicit-check-not=define --check-prefix=RESULT %s < %t.0

; INTERESTING: @externally_initialized_keep = externally_initialized global i32 0
; INTERESTING: @externally_initialized_drop

; RESULT: @externally_initialized_keep = externally_initialized global i32 0, align 4
; RESULT: @externally_initialized_drop = global i32 1, align 4
@externally_initialized_keep = externally_initialized global i32 0, align 4
@externally_initialized_drop = externally_initialized global i32 1, align 4

