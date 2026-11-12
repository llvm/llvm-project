; RUN: %lli -jit-kind=mcjit %s > /dev/null
; RUN: %lli %s > /dev/null

define i32 @main() nounwind uwtable {
entry:
  %retval = alloca i32, align 4
  store i32 0, ptr %retval
  ret i32 0
}
