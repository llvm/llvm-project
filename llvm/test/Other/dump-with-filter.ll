;; Make sure we can run -filter-print-funcs with -ir-dump-directory.
; RUN: rm -rf %t/logs
; RUN: opt %s -disable-output -passes='no-op-function' -print-before=no-op-function -print-after=no-op-function \
; RUN:   -ir-dump-directory %t/logs -filter-print-funcs=test2
; RUN: ls %t/logs | count 2
; RUN: rm -rf %t/logs

define void @test() {
    ret void
}

define void @test2() {
    ret void
}
