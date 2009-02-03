; RUN: llvm-as < %s | opt -licm

%"java/lang/Object" = type { %struct.llvm_java_object_base }
%"java/lang/StringBuffer" = type { "java/lang/Object", i32, { "java/lang/Object", i32, [0 x i8] }*, i1 }
%struct.llvm_java_object_base = type opaque

define void @"java/lang/StringBuffer/setLength(I)V"(%struct.llvm_java_object_base*) {
bc0:
	br i1 false, label %bc40, label %bc65

bc65:		; preds = %bc0, %bc40
	ret void

bc40:		; preds = %bc0, %bc40
	%tmp75 = bitcast %struct.llvm_java_object_base* %0 to %"java/lang/StringBuffer"*		; <"java/lang/StringBuffer"*> [#uses=1]
	%tmp76 = getelementptr %"java/lang/StringBuffer"* %tmp75, i32 0, i32 1		; <i32*> [#uses=1]
	store i32 0, i32* %tmp76
	%tmp381 = bitcast %struct.llvm_java_object_base* %0 to %"java/lang/StringBuffer"*		; <"java/lang/StringBuffer"*> [#uses=1]
	%tmp392 = getelementptr %"java/lang/StringBuffer"* %tmp381, i32 0, i32 1		; <i32*> [#uses=1]
	%tmp403 = load i32* %tmp392		; <i32> [#uses=0]
	br i1 false, label %bc40, label %bc65
}
