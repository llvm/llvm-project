; RUN: opt < %s -basic-aa -dse -S | FileCheck %s

declare noalias ptr @calloc(i64, i64) inaccessiblememonly allockind("alloc,zeroed")

define ptr @test1() {
; CHECK-LABEL: test1
  %1 = tail call noalias ptr @calloc(i64 1, i64 4)
  ; This store is dead and should be removed
  store i32 0, ptr %1, align 4
; CHECK-NOT: store i32 0, ptr %1, align 4
  ret ptr %1
}

define ptr @test2() {
; CHECK-LABEL: test2
  %1 = tail call noalias ptr @calloc(i64 1, i64 4)
  %2 = getelementptr i32, ptr %1, i32 5
  store i32 0, ptr %2, align 4
; CHECK-NOT: store i32 0, ptr %1, align 4
  ret ptr %1
}

define ptr @test3(ptr %arg) {
; CHECK-LABEL: test3
  store i32 0, ptr %arg, align 4
; CHECK: store i32 0, ptr %arg, align 4
  ret ptr %arg
}

declare void @clobber_memory(ptr)
define ptr @test4() {
; CHECK-LABEL: test4
  %1 = tail call noalias ptr @calloc(i64 1, i64 4)
  call void @clobber_memory(ptr %1)
  store i8 0, ptr %1, align 4
; CHECK: store i8 0, ptr %1, align 4
  ret ptr %1
}

define ptr @test5() {
; CHECK-LABEL: test5
  %1 = tail call noalias ptr @calloc(i64 1, i64 4)
  store volatile i32 0, ptr %1, align 4
; CHECK: store volatile i32 0, ptr %1, align 4
  ret ptr %1
}

define ptr @test6() {
; CHECK-LABEL: test6
  %1 = tail call noalias ptr @calloc(i64 1, i64 4)
  store i8 5, ptr %1, align 4
; CHECK: store i8 5, ptr %1, align 4
  ret ptr %1
}

define ptr @test7(i8 %arg) {
; CHECK-LABEL: test7
  %1 = tail call noalias ptr @calloc(i64 1, i64 4)
  store i8 %arg, ptr %1, align 4
; CHECK: store i8 %arg, ptr %1, align 4
  ret ptr %1
}

define ptr @test8() {
; CHECK-LABEL: test8
; CHECK-NOT: store
  %p = tail call noalias ptr @calloc(i64 1, i64 4)
  store i8 0, ptr %p, align 1
  %p.1 = getelementptr i8, ptr %p, i32 1
  store i8 0, ptr %p.1, align 1
  %p.3 = getelementptr i8, ptr %p, i32 3
  store i8 0, ptr %p.3, align 1
  %p.2 = getelementptr i8, ptr %p, i32 2
  store i8 0, ptr %p.2, align 1
  ret ptr %p
}

define ptr @test9() {
; CHECK-LABEL: test9
; CHECK-NEXT:    %p = tail call noalias ptr @calloc(i64 1, i64 4)
; CHECK-NEXT:    store i8 5, ptr %p, align 1
; CHECK-NEXT:    ret ptr %p

  %p = tail call noalias ptr @calloc(i64 1, i64 4)
  store i8 5, ptr %p, align 1
  %p.1 = getelementptr i8, ptr %p, i32 1
  store i8 0, ptr %p.1, align 1
  %p.3 = getelementptr i8, ptr %p, i32 3
  store i8 0, ptr %p.3, align 1
  %p.2 = getelementptr i8, ptr %p, i32 2
  store i8 0, ptr %p.2, align 1
  ret ptr %p
}

define ptr @test10() {
; CHECK-LABEL: @test10(
; CHECK-NEXT:    [[P:%.*]] = tail call noalias ptr @calloc(i64 1, i64 4)
; CHECK-NEXT:    [[P_3:%.*]] = getelementptr i8, ptr [[P]], i32 3
; CHECK-NEXT:    store i8 5, ptr [[P_3]], align 1
; CHECK-NEXT:    ret ptr [[P]]
;

  %p = tail call noalias ptr @calloc(i64 1, i64 4)
  store i8 0, ptr %p, align 1
  %p.1 = getelementptr i8, ptr %p, i32 1
  store i8 0, ptr %p.1, align 1
  %p.3 = getelementptr i8, ptr %p, i32 3
  store i8 5, ptr %p.3, align 1
  %p.2 = getelementptr i8, ptr %p, i32 2
  store i8 0, ptr %p.2, align 1
  ret ptr %p
}

define ptr @test11() {
; CHECK-LABEL: @test11(
; CHECK-NEXT:    [[P:%.*]] = tail call noalias ptr @calloc(i64 1, i64 4)
; CHECK-NEXT:    ret ptr [[P]]
;

  %p = tail call noalias ptr @calloc(i64 1, i64 4)
  store i8 0, ptr %p, align 1
  %p.1 = getelementptr i8, ptr %p, i32 1
  store i8 0, ptr %p.1, align 1
  %p.3 = getelementptr i8, ptr %p, i32 3
  store i8 5, ptr %p.3, align 1
  %p.2 = getelementptr i8, ptr %p, i32 2
  store i8 0, ptr %p.2, align 1
  %p.3.2 = getelementptr i8, ptr %p, i32 3
  store i8 0, ptr %p.3.2, align 1
  ret ptr %p
}

define ptr @test12() {
; CHECK-LABEL: @test12(
; CHECK-NEXT:    [[P:%.*]] = tail call noalias ptr @calloc(i64 1, i64 4)
; CHECK-NEXT:    [[P_3:%.*]] = getelementptr i8, ptr [[P]], i32 3
; CHECK-NEXT:    store i8 5, ptr [[P_3]], align 1
; CHECK-NEXT:    call void @use(ptr [[P]])
; CHECK-NEXT:    [[P_3_2:%.*]] = getelementptr i8, ptr [[P]], i32 3
; CHECK-NEXT:    store i8 0, ptr [[P_3_2]], align 1
; CHECK-NEXT:    ret ptr [[P]]
;

  %p = tail call noalias ptr @calloc(i64 1, i64 4)
  %p.3 = getelementptr i8, ptr %p, i32 3
  store i8 5, ptr %p.3, align 1
  call void @use(ptr %p)
  %p.3.2 = getelementptr i8, ptr %p, i32 3
  store i8 0, ptr %p.3.2, align 1
  ret ptr %p
}

declare void @use(ptr) readonly
