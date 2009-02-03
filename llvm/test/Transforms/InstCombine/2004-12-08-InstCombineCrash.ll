; RUN: llvm-as %s -o - | opt -instcombine


%"java/lang/Object" = type { %struct.llvm_java_object_base }
%"java/lang/StringBuffer" = type { %"java/lang/Object", i32, { %"java/lang/Object", i32, [0 x i16] }*, i1 }
%struct.llvm_java_object_base = type opaque

define void @"java/lang/StringBuffer/append(Ljava/lang/String;)Ljava/lang/StringBuffer;"() {
bc0:
        %tmp53 = getelementptr %"java/lang/StringBuffer"* null, i32 0, i32 1            ; <i32*> [#uses=1]
        store i32 0, i32* %tmp53
        ret void
}

