; Check that the address sanitizer pass can be reused
; RUN: opt < %s -S -run-twice -passes='asan-pipeline'

define void @foo(i64* %b) nounwind uwtable sanitize_address {
  entry:
  store i64 0, i64* %b, align 1
  ret void
}
