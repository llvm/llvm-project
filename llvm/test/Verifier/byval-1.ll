; RUN: not llvm-as < %s >& /dev/null
declare void @h(i32 byval %num)
