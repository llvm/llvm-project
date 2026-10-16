; RUN: llvm-reduce -abort-on-invalid-reduction --delta-passes=global-values --test FileCheck --test-arg --check-prefix=INTERESTING --test-arg %s --test-arg --input-file %s -o %t.0
; RUN: FileCheck --implicit-check-not=define --check-prefix=RESULT %s < %t.0

; INTERESTING: @code_model_large_keep = global i32 0, code_model "large", align 4
; INTERESTING @code_model_large_drop = global i32 0

; RESULT: @code_model_large_keep = global i32 0, code_model "large", align 4{{$}}
; RESULT: @code_model_large_drop = global i32 0, align 4{{$}}
@code_model_large_keep = global i32 0, code_model "large", align 4
@code_model_large_drop = global i32 0, code_model "large", align 4

; INTERESTING: @code_model_tiny_keep = global i32 0, code_model "tiny", align 4
; INTERESTING @code_model_tiny_drop = global i32 0

; RESULT: @code_model_tiny_keep = global i32 0, code_model "tiny", align 4{{$}}
; RESULT: @code_model_tiny_drop = global i32 0, align 4{{$}}
@code_model_tiny_keep = global i32 0, code_model "tiny", align 4
@code_model_tiny_drop = global i32 0, code_model "tiny", align 4
