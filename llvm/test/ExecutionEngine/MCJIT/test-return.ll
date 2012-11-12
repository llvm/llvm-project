; RUN: %lli -mtriple=%mcjit_triple -use-mcjit %s > /dev/null

define i32 @main() nounwind uwtable {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval
  ret i32 0
}
