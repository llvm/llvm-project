; RUN: llc -O0 -mtriple=arm64 < %s

declare ptr @llvm.launder.invariant.group(ptr)

define ptr @barrier(ptr %p) {
; CHECK: bl llvm.launder.invariant.group
        %q = call ptr @llvm.launder.invariant.group(ptr %p)
        ret ptr %q
}

