; RUN: not llvm-as < %s > /dev/null 2>&1
declare void @h(i32 byval %num)
