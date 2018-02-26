; RUN: llc -O0 -mtriple=arm64 < %s

declare i8* @llvm.invariant.group.barrier(i8*)

define i8* @barrier(i8* %p) {
; CHECK: bl llvm.invariant.group.barrier
        %q = call i8* @llvm.invariant.group.barrier(i8* %p)
        ret i8* %q
}

