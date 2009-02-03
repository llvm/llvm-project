; RUN: llvm-as < %s | llc -march=ia64

define double @test() {
  ret double 0.0
}
