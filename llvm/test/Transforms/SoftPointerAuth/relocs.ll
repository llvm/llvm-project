; RUN: opt < %s -soft-ptrauth -S | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

; CHECK-NOT: @test1_reloc
; CHECK: @test1 = internal global { i8**, i32, i32, i8* } { i8** null, i32 1342177280, i32 0, i8* null }, align 8

@test1_reloc = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (i8*)* @test1_function to i8*), i32 1, i64 ptrtoint (i8** getelementptr inbounds ({ i8**, i32, i32, i8* }, { i8**, i32, i32, i8* }* @test1, i32 0, i32 3) to i64), i64 0 }, section "llvm.ptrauth", align 8
@test1 = internal constant { i8**, i32, i32, i8* } { i8** null, i32 1342177280, i32 0, i8* bitcast ({ i8*, i32, i64, i64 }* @test1_reloc to i8*) }, align 8

define internal void @test1_function(i8*) {
entry:
  ret void
}

; CHECK: define private void @ptrauth_soft_init() {
; CHECK: [[T0:%.*]] = call i8* @__ptrauth_sign(i8* bitcast (void (i8*)* @test1_function to i8*), i32 1, i64 ptrtoint (i8** getelementptr inbounds ({ i8**, i32, i32, i8* }, { i8**, i32, i32, i8* }* @test1, i32 0, i32 3) to i64)) [[NOUNWIND:#[0-9]+]]
; CHECK: [[T1:%.*]] = bitcast i8* [[T0]] to { i8*, i32, i64, i64 }*
; CHECK: [[T2:%.*]] = bitcast { i8*, i32, i64, i64 }* [[T1]] to i8*
; CHECK: store i8* [[T2]], i8** getelementptr inbounds ({ i8**, i32, i32, i8* }, { i8**, i32, i32, i8* }* @test1, i32 0, i32 3)

; CHECK: attributes [[NOUNWIND]] = { nounwind }
