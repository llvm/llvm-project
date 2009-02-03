; RUN: llvm-as < %s | llvm-dis | grep 18446744073709551615 | count 2

define [18446744073709551615 x i8]* @foo() {
  ret [18446744073709551615 x i8]* null
}
