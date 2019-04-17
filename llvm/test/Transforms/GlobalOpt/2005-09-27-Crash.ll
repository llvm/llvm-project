; RUN: opt < %s -globalopt -disable-output
        %RPyString = type { i32, %arraytype.Char }
        %arraytype.Char = type { i32, [0 x i8] }
        %arraytype.Signed = type { i32, [0 x i32] }
        %functiontype.1 = type { %RPyString* (i32) *} 
        %structtype.test = type { i32, %arraytype.Signed }
@structinstance.test = internal global { i32, { i32, [2 x i32] } } { i32 41, { i32, [2 x i32] } { i32 2, [2 x i32] [ i32 100, i32 101 ] } }              ; <{ i32, { i32, [2 x i32] } }*> [#uses=1]

define fastcc void @pypy_array_constant() {
block0:
        %tmp.9 = getelementptr %structtype.test, %structtype.test* bitcast ({ i32, { i32, [2 x i32] } }* @structinstance.test to %structtype.test*), i32 0, i32 0          ; <i32*> [#uses=0]
        ret void
}

define fastcc void @new.varsizestruct.rpy_string() {
        unreachable
}

define void @__entrypoint__pypy_array_constant() {
        call fastcc void @pypy_array_constant( )
        ret void
}

define void @__entrypoint__raised_LLVMException() {
        ret void
}

