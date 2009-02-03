; RUN: llvm-as < %s | llc -march=cpp -cppfname=WAKKA | not grep makeLLVMModule
; PR1515

define void @foo() {
  ret void
}

