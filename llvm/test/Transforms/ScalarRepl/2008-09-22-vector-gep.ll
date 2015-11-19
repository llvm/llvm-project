; This test checks to see if scalarrepl also works when a gep with all zeroes is
; used instead of a bitcast to prepare a memmove pointer argument. Previously,
; this would not work when there was a vector involved in the struct, preventing
; scalarrepl from removing the alloca below.

; RUN: opt < %s -scalarrepl -S > %t
; RUN: cat %t | not grep alloca
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"

%struct.two = type <{ < 2 x i8 >, i16 }>

define void @main(%struct.two* %D, i16 %V) {
entry:
	%S = alloca %struct.two
        %S.2 = getelementptr %struct.two, %struct.two* %S, i32 0, i32 1
        store i16 %V, i16* %S.2
        ; This gep is effectively a bitcast to i8*, but is sometimes generated
        ; because the type of the first element in %struct.two is i8.
	%tmpS = getelementptr %struct.two, %struct.two* %S, i32 0, i32 0, i32 0 
	%tmpD = bitcast %struct.two* %D to i8*
        call void @llvm.memmove.p0i8.p0i8.i32(i8* %tmpD, i8* %tmpS, i32 4, i1 false)
        ret void
}

declare void @llvm.memmove.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i1) nounwind
