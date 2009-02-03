; RUN: llvm-as < %s | opt -dse

%"java/lang/Object" = type { %struct.llvm_java_object_base }
%"java/lang/StringBuffer" = type { "java/lang/Object", i32, { "java/lang/Object", i32, [0 x i8] }*, i1 }
%struct.llvm_java_object_base = type opaque

define void @"java/lang/StringBuffer/ensureCapacity_unsynchronized(I)V"() {
bc0:
	%tmp = getelementptr %"java/lang/StringBuffer"* null, i32 0, i32 3		; <i1*> [#uses=1]
	br i1 false, label %bc16, label %bc7

bc16:		; preds = %bc0
	%tmp91 = getelementptr %"java/lang/StringBuffer"* null, i32 0, i32 2		; <{ "java/lang/Object", i32, [0 x i8] }**> [#uses=1]
	store { %"java/lang/Object", i32, [0 x i8] }* null, { %"java/lang/Object", i32, [0 x i8] }** %tmp91
	store i1 false, i1* %tmp
	ret void

bc7:		; preds = %bc0
	ret void
}
