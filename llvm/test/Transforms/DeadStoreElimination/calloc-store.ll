; RUN: opt < %s -basic-aa -dse -S | FileCheck %s

declare noalias i8* @calloc(i64, i64)

define i32* @test1() {
; CHECK-LABEL: test1
  %1 = tail call noalias i8* @calloc(i64 1, i64 4)
  %2 = bitcast i8* %1 to i32*
  ; This store is dead and should be removed
  store i32 0, i32* %2, align 4
; CHECK-NOT: store i32 0, i32* %2, align 4
  ret i32* %2
}

define i32* @test2() {
; CHECK-LABEL: test2
  %1 = tail call noalias i8* @calloc(i64 1, i64 4)
  %2 = bitcast i8* %1 to i32*
  %3 = getelementptr i32, i32* %2, i32 5
  store i32 0, i32* %3, align 4
; CHECK-NOT: store i32 0, i32* %2, align 4
  ret i32* %2
}

define i32* @test3(i32 *%arg) {
; CHECK-LABEL: test3
  store i32 0, i32* %arg, align 4
; CHECK: store i32 0, i32* %arg, align 4
  ret i32* %arg
}

declare void @clobber_memory(i8*)
define i8* @test4() {
; CHECK-LABEL: test4
  %1 = tail call noalias i8* @calloc(i64 1, i64 4)
  call void @clobber_memory(i8* %1)
  store i8 0, i8* %1, align 4
; CHECK: store i8 0, i8* %1, align 4
  ret i8* %1
}

define i32* @test5() {
; CHECK-LABEL: test5
  %1 = tail call noalias i8* @calloc(i64 1, i64 4)
  %2 = bitcast i8* %1 to i32*
  store volatile i32 0, i32* %2, align 4
; CHECK: store volatile i32 0, i32* %2, align 4
  ret i32* %2
}

define i8* @test6() {
; CHECK-LABEL: test6
  %1 = tail call noalias i8* @calloc(i64 1, i64 4)
  store i8 5, i8* %1, align 4
; CHECK: store i8 5, i8* %1, align 4
  ret i8* %1
}

define i8* @test7(i8 %arg) {
; CHECK-LABEL: test7
  %1 = tail call noalias i8* @calloc(i64 1, i64 4)
  store i8 %arg, i8* %1, align 4
; CHECK: store i8 %arg, i8* %1, align 4
  ret i8* %1
}
