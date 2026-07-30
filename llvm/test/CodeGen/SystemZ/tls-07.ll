; Test local-dynamic TLS access optimizations.
;
; If we access two different local-dynamic TLS variables, we only
; need a single call to __tls_get_offset.
;
; RUN: llc < %s -mcpu=z10 -mtriple=s390x-linux-gnu -relocation-model=pic | grep "__tls_get_offset" | count 1

@x = thread_local(localdynamic) global i32 0
@y = thread_local(localdynamic) global i32 0

define i32 @foo() {
  %valx = load i32, ptr @x
  %valy = load i32, ptr @y
  %add = add nsw i32 %valx, %valy
  ret i32 %add
}
