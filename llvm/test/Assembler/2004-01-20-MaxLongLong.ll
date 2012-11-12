; RUN: llvm-as < %s | llvm-dis | grep 9223372036854775808

global i64 -9223372036854775808

