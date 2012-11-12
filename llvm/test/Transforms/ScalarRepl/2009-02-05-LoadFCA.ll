; RUN: opt < %s -scalarrepl -instcombine -inline -instcombine -S | grep "ret i32 42"
; PR3489
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "x86_64-apple-darwin10.0"
	%struct.anon = type <{ i32, i32, i32 }>

define i32 @f({ i64, i64 }) nounwind {
entry:
	%tmp = alloca { i64, i64 }, align 8		; <{ i64, i64 }*> [#uses=2]
	store { i64, i64 } %0, { i64, i64 }* %tmp
	%1 = bitcast { i64, i64 }* %tmp to %struct.anon*		; <%struct.anon*> [#uses=1]
	%2 = load %struct.anon* %1, align 8		; <%struct.anon> [#uses=1]
        %tmp3 = extractvalue %struct.anon %2, 0
	ret i32 %tmp3
}

define i32 @g() {
  %a = call i32 @f({i64,i64} { i64 42, i64 1123123123123123 })
  ret i32 %a
}
