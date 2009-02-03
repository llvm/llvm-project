; RUN: llvm-as < %s | llc -march=ppc32 | grep nop
target triple = "powerpc-apple-darwin8"


define void @bork() noreturn nounwind  {
entry:
        unreachable
}
