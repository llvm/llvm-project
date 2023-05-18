; RUN: llc < %s

define void @main() {
        call void @llvm.assume(i1 1)
        ret void
}

declare void @llvm.assume(i1) nounwind

