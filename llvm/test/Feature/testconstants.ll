; RUN: llvm-as < %s | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

@somestr = constant [11 x i8] c"hello world"            ; <ptr> [#uses=1]
@array = constant [2 x i32] [ i32 12, i32 52 ]          ; <ptr> [#uses=1]
@0 = constant { i32, i32 } { i32 4, i32 3 }        ; <ptr>:0 [#uses=0]

define ptr @testfunction(i32 %i0, i32 %j0) {
        ret ptr @array
}

define ptr @otherfunc(i32, double) {
        %somestr = getelementptr [11 x i8], ptr @somestr, i64 0, i64 0              ; <ptr> [#uses=1]
        ret ptr %somestr
}

define ptr @yetanotherfunc(i32, double) {
        ret ptr null
}

define i32 @negativeUnsigned() {
        ret i32 -1
}

define i32 @largeSigned() {
        ret i32 -394967296
}

