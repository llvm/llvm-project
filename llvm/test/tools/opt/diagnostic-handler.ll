; Verify that LLVMContext's default diagnostic handler does not exit on
; the first error reported via LLVMContext::emitError. opt should continue
; to run remaining passes (so the print-module pass at the end of the
; pipeline still emits the module to stdout), then return a non-zero exit
; status because the handler's HasErrors flag was set.

; RUN: not opt -S -passes='module(sancov-module),print<module-debuginfo>' \
; RUN:     -sanitizer-coverage-level=1 -sanitizer-coverage-stack-depth %s \
; RUN:     >%t.out 2>%t.err
; RUN: FileCheck --check-prefix=ERR --input-file=%t.err %s
; RUN: FileCheck --check-prefix=OUT --input-file=%t.out %s

; sancov-module emits an error when '__sancov_lowest_stack' has the wrong
; type. The pass returns safely and the pipeline continues.
@__sancov_lowest_stack = thread_local global i32 0

define i32 @f() {
  ret i32 0
}

; ERR: error: '__sancov_lowest_stack' should not be declared by the user

; The print-module pass after sancov-module still runs, proving opt did
; not exit on the first emitError.
; OUT: define i32 @f()
