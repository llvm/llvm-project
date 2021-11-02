target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

define i32 ()* @foo_resolver() {
  ret i32 ()* inttoptr (i32 2 to i32 ()*)
}
