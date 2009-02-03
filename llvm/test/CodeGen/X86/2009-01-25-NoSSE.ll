; RUN: llvm-as < %s | llc -march=x86-64 -mattr=-sse,-sse2 | not grep xmm
; PR3402
target datalayout =
"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"
        %struct.ktermios = type { i32, i32, i32, i32, i8, [19 x i8], i32, i32 }

define void @foo() nounwind {
entry:
        %termios = alloca %struct.ktermios, align 8
        %termios1 = bitcast %struct.ktermios* %termios to i8*
        call void @llvm.memset.i64(i8* %termios1, i8 0, i64 44, i32 8)
        call void @bar(%struct.ktermios* %termios) nounwind
        ret void
}

declare void @llvm.memset.i64(i8* nocapture, i8, i64, i32) nounwind

declare void @bar(%struct.ktermios*)

