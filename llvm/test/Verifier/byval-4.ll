; RUN: llvm-as %s -disable-output
%struct.foo = type { i64 }

declare void @h(ptr byval(%struct.foo) %num)
