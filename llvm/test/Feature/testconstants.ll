; RUN: llvm-as < %s | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

@somestr = constant [11 x i8] c"hello world"            ; <[11 x i8]*> [#uses=1]
@array = constant [2 x i32] [ i32 12, i32 52 ]          ; <[2 x i32]*> [#uses=1]
constant { i32, i32 } { i32 4, i32 3 }          ; <{ i32, i32 }*>:0 [#uses=0]

define [2 x i32]* @testfunction(i32 %i0, i32 %j0) {
        ret [2 x i32]* @array
}

define i8* @otherfunc(i32, double) {
        %somestr = getelementptr [11 x i8]* @somestr, i64 0, i64 0              ; <i8*> [#uses=1]
        ret i8* %somestr
}

define i8* @yetanotherfunc(i32, double) {
        ret i8* null
}

define i32 @negativeUnsigned() {
        ret i32 -1
}

define i32 @largeSigned() {
        ret i32 -394967296
}

