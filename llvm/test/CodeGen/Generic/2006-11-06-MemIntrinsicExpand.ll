; RUN: llvm-as < %s | llc -march=x86 | not grep adc
; PR987

declare void @llvm.memcpy.i64(i8*, i8*, i64, i32)

define void @foo(i64 %a) {
        %b = add i64 %a, 1              ; <i64> [#uses=1]
        call void @llvm.memcpy.i64( i8* null, i8* null, i64 %b, i32 1 )
        ret void
}

