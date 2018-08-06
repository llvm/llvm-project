; RUN: llc -O0 -mtriple=arm64 < %s

declare i8* @llvm.launder.invariant.group(i8*)

define i8* @barrier(i8* %p) {
; CHECK: bl llvm.launder.invariant.group
        %q = call i8* @llvm.launder.invariant.group(i8* %p)
        ret i8* %q
}

