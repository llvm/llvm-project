; Test general-dynamic TLS access optimizations.
;
; If we access two different TLS variables, we need two calls to
; __tls_get_offset, but should load _GLOBAL_OFFSET_TABLE only once.
;
; RUN: llc < %s -mcpu=z10 -mtriple=s390x-linux-gnu -relocation-model=pic | grep "__tls_get_offset" | count 2
; RUN: llc < %s -mcpu=z10 -mtriple=s390x-linux-gnu -relocation-model=pic | grep "_GLOBAL_OFFSET_TABLE_" | count 1

@x = thread_local global i32 0
@y = thread_local global i32 0

define i32 @foo() {
  %valx = load i32, ptr @x
  %valy = load i32, ptr @y
  %add = add nsw i32 %valx, %valy
  ret i32 %add
}
