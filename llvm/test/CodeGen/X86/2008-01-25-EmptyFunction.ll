; RUN: llvm-as < %s | llc -march=x86 | grep nop
target triple = "i686-apple-darwin8"


define void @bork() noreturn nounwind  {
entry:
        unreachable
}
