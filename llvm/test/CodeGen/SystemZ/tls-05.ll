; Test general-dynamic TLS access optimizations.
;
; If we access the same TLS variable twice, there should only be
; a single call to __tls_get_offset.
;
; RUN: llc < %s -mcpu=z10 -mtriple=s390x-linux-gnu -relocation-model=pic | grep "__tls_get_offset" | count 1

@x = thread_local global i32 0

define i32 @foo() {
  %val = load i32, ptr @x
  %inc = add nsw i32 %val, 1
  store i32 %inc, ptr @x
  ret i32 %val
}
