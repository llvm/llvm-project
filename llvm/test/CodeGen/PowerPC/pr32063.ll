; RUN: llc -O2 < %s | FileCheck %s
target triple = "powerpc64le-linux-gnu"

define void @foo(i32 %v, ptr %p) {
        %1 = and i32 %v, -65536
        %2 = tail call i32 @llvm.bswap.i32(i32 %1)
        %conv = trunc i32 %2 to i16
        store i16 %conv, ptr %p
        ret void

; CHECK:     srwi
; CHECK:     sthbrx
; CHECK-NOT: stwbrx
}

declare i32 @llvm.bswap.i32(i32)
