; RUN: not opt -S -verify < %s 2>&1 | FileCheck %s

; CHECK: invalid llvm.ptrauth global: global doesn't have an initializer
@no_init = external global { i8*, i32, i64, i64 }, section "llvm.ptrauth"

; CHECK: invalid llvm.ptrauth global: global isn't a struct
@not_struct = constant i8* null, section "llvm.ptrauth"

; CHECK: invalid llvm.ptrauth global: global doesn't have type '{ i8*, i32, i64, i64 }'
@bad_type = constant { i8*, i32, i32, i32 } zeroinitializer, section "llvm.ptrauth"

; CHECK: invalid llvm.ptrauth global: key isn't a constant integer
@bad_key = constant { i8*, i32, i64, i64 } { i8* null, i32 ptrtoint (i32* @g to i32), i64 0, i64 0}, section "llvm.ptrauth"

; CHECK: invalid llvm.ptrauth global: discriminator isn't a constant integer
@bad_disc = constant { i8*, i32, i64, i64 } { i8* null, i32 0, i64 0, i64 ptrtoint (i32* @g to i64)}, section "llvm.ptrauth"

; CHECK-NOT: invalid
@valid = private constant { i8*, i32, i64, i64 } { i8* getelementptr inbounds (i8, i8* bitcast (i32* @g to i8*), i64 2), i32 3, i64 0, i64 0 }, section "llvm.ptrauth"

@g = external global i32
