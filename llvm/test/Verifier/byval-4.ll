; RUN: llvm-as %s -o /dev/null -f
%struct.foo = type { i64 }

declare void @h(%struct.foo* byval %num)
