; This testcase shows that scalarrepl is able to replace struct alloca's which
; are directly loaded from or stored to (using the first class aggregates
; feature).

; RUN: llvm-as < %s | opt -scalarrepl | llvm-dis > %t
; RUN: cat %t | not grep alloca

%struct.foo = type { i32, i32 }

define i32 @test(%struct.foo* %P) {
entry:
	%L = alloca %struct.foo, align 8		; <%struct.foo*> [#uses=2]
        %V = load %struct.foo* %P
        store %struct.foo %V, %struct.foo* %L

	%tmp4 = getelementptr %struct.foo* %L, i32 0, i32 0		; <i32*> [#uses=1]
	%tmp5 = load i32* %tmp4		; <i32> [#uses=1]
	ret i32 %tmp5
}

define %struct.foo @test2(i32 %A, i32 %B) {
entry:
	%L = alloca %struct.foo, align 8		; <%struct.foo*> [#uses=2]
        %L.0 = getelementptr %struct.foo* %L, i32 0, i32 0
        store i32 %A, i32* %L.0
        %L.1 = getelementptr %struct.foo* %L, i32 0, i32 1
        store i32 %B, i32* %L.1
        %V = load %struct.foo* %L
        ret %struct.foo %V
}
