; Check that the address sanitizer pass can be reused
; RUN: opt < %s -S -run-twice -passes=asan

define void @foo(ptr %b) nounwind uwtable sanitize_address {
  entry:
  store i64 0, ptr %b, align 1
  ret void
}
