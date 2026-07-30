; RUN: llc < %s -mtriple=x86_64-linux | FileCheck %s
%struct.S = type { i32}

@.str = private constant [10 x i8] c"ptr = %p\0A\00", align 1 ; <ptr> [#uses=1]
@.str1 = private constant [8 x i8] c"Failed \00", align 1 ; <ptr> [#uses=1]
@.str2 = private constant [2 x i8] c"0\00", align 1 ; <ptr> [#uses=1]
@.str3 = private constant [7 x i8] c"test.c\00", align 1 ; <ptr> [#uses=1]
@__PRETTY_FUNCTION__.2067 = internal constant [13 x i8] c"aligned_func\00" ; <ptr> [#uses=1]

define void @aligned_func(ptr byval(%struct.S) align 64 %obj) nounwind {
entry:
  %ptr = alloca ptr                               ; <ptr> [#uses=3]
  %p = alloca i64                                 ; <ptr> [#uses=3]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  store ptr %obj, ptr %ptr, align 8
  %0 = load ptr, ptr %ptr, align 8                    ; <ptr> [#uses=1]
  %1 = ptrtoint ptr %0 to i64                     ; <i64> [#uses=1]
  store i64 %1, ptr %p, align 8
  %2 = load ptr, ptr %ptr, align 8                    ; <ptr> [#uses=1]
  %3 = call i32 (ptr, ...) @printf(ptr @.str, ptr %2) nounwind ; <i32> [#uses=0]
  %4 = load i64, ptr %p, align 8                      ; <i64> [#uses=1]
  %5 = and i64 %4, 140737488355264                ; <i64> [#uses=1]
  %6 = load i64, ptr %p, align 8                      ; <i64> [#uses=1]
  %7 = icmp ne i64 %5, %6                         ; <i1> [#uses=1]
  br i1 %7, label %bb, label %bb2

bb:                                               ; preds = %entry
  %8 = call i32 @puts(ptr @.str1) nounwind ; <i32> [#uses=0]
  call void @__assert_fail(ptr @.str2, ptr @.str3, i32 18, ptr @__PRETTY_FUNCTION__.2067) noreturn nounwind
  unreachable

bb2:                                              ; preds = %entry
  br label %return

return:                                           ; preds = %bb2
  ret void
}

declare i32 @printf(ptr, ...) nounwind

declare i32 @puts(ptr)

declare void @__assert_fail(ptr, ptr, i32, ptr) noreturn nounwind

define void @main() nounwind {
entry:
; CHECK: main
; CHECK: andq    $-64, %rsp
  %s1 = alloca %struct.S                          ; <ptr> [#uses=4]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  %0 = getelementptr inbounds %struct.S, ptr %s1, i32 0, i32 0 ; <ptr> [#uses=1]
  store i32 1, ptr %0, align 4
  call void @aligned_func(ptr byval(%struct.S) align 64 %s1) nounwind
  br label %return

return:                                           ; preds = %entry
  ret void
}
