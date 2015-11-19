; The store into %p should end up with a known alignment of 1, since the memcpy
; is only known to access it with 1-byte alignment.
; RUN: opt < %s -scalarrepl -S | grep "store i16 1, .*, align 1"
; PR3720
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"

        %struct.st = type { i16 }

define void @f(i8* %p) nounwind {
entry:
        %s = alloca %struct.st, align 4  ; <%struct.st*> [#uses=2]
        %0 = getelementptr %struct.st, %struct.st* %s, i32 0, i32 0  ; <i16*> [#uses=1]
        store i16 1, i16* %0, align 4
        %s1 = bitcast %struct.st* %s to i8*  ; <i8*> [#uses=1]
        call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 1 %p, i8* align 1 %s1, i32 2, i1 false)
        ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i1) nounwind
