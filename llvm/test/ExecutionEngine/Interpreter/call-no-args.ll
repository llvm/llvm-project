; RUN: %lli -jit-kind=mcjit -force-interpreter %s

declare void @exit(i32)
declare i32 @rand()

define i32 @main() {
  %ret = call i32 @rand()
  call void @exit(i32 0)
  ret i32 %ret
}
