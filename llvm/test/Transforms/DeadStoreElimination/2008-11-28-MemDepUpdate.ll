; RUN: llvm-as < %s | opt -dse | llvm-dis
; PR3141
	%struct.ada__tags__dispatch_table = type { [1 x i32] }
	%struct.f393a00_1__object = type { %struct.ada__tags__dispatch_table*, i8 }
	%struct.f393a00_2__windmill = type { %struct.f393a00_1__object, i16 }

define void @f393a00_2__swap(%struct.f393a00_2__windmill* %a, %struct.f393a00_2__windmill* %b) {
entry:
	%t = alloca %struct.f393a00_2__windmill		; <%struct.f393a00_2__windmill*> [#uses=1]
	%0 = getelementptr %struct.f393a00_2__windmill* %t, i32 0, i32 0, i32 0		; <%struct.ada__tags__dispatch_table**> [#uses=1]
	%1 = load %struct.ada__tags__dispatch_table** null, align 4		; <%struct.ada__tags__dispatch_table*> [#uses=1]
	%2 = load %struct.ada__tags__dispatch_table** %0, align 8		; <%struct.ada__tags__dispatch_table*> [#uses=1]
	store %struct.ada__tags__dispatch_table* %2, %struct.ada__tags__dispatch_table** null, align 4
	store %struct.ada__tags__dispatch_table* %1, %struct.ada__tags__dispatch_table** null, align 4
	ret void
}
