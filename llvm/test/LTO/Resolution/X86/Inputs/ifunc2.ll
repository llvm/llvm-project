target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

define ptr @foo_resolver() {
  ret ptr inttoptr (i32 2 to ptr)
}
