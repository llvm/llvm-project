; RUN: %lli -mtriple=%mcjit_triple -use-mcjit %s > /dev/null

define i32 @_Z14func_exit_codev() nounwind uwtable {
entry:
  ret i32 0
}

define i32 @main() nounwind uwtable {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval
  %call = call i32 @_Z14func_exit_codev()
  ret i32 %call
}
