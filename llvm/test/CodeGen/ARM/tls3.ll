; RUN: llvm-as < %s | llc -march=arm -mtriple=arm-linux-gnueabi | \
; RUN:     grep {tbss}

%struct.anon = type { i32, i32 }
@teste = internal thread_local global %struct.anon zeroinitializer		; <%struct.anon*> [#uses=1]

define i32 @main() {
entry:
	%tmp2 = load i32* getelementptr (%struct.anon* @teste, i32 0, i32 0), align 8		; <i32> [#uses=1]
	ret i32 %tmp2
}
