; RUN: llvm-as %s -o /dev/null
%struct.foo = type { i64 }

declare void @h(ptr byval(%struct.foo) %num)
