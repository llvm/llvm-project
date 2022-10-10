; RUN: llvm-as < %s | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

@MyVar = external global i32            ; <i32*> [#uses=1]
@MyIntList = external global { i32*, i32 }               ; <{ \2*, i32 }*> [#uses=1]
@0 = external global i32             ; <i32*>:0 [#uses=0]
@AConst = constant i32 123              ; <i32*> [#uses=0]
@AString = constant [4 x i8] c"test"            ; <[4 x i8]*> [#uses=0]
@ZeroInit = global { [100 x i32], [40 x float] } zeroinitializer                ; <{ [100 x i32], [40 x float] }*> [#uses=0]

define i32 @foo(i32 %blah) {
        store i32 5, i32* @MyVar
        %idx = getelementptr { i32*, i32 }, { i32*, i32 }* @MyIntList, i64 0, i32 1             ; <i32*> [#uses=1]
        store i32 12, i32* %idx
        ret i32 %blah
}

@1 = default dllexport global i32 42
@2 = dllexport global i32 42
