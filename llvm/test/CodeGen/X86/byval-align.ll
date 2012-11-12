; RUN: llc < %s -mtriple=x86_64-linux | FileCheck %s
%struct.S = type { i32}

@.str = private constant [10 x i8] c"ptr = %p\0A\00", align 1 ; <[10 x i8]*> [#uses=1]
@.str1 = private constant [8 x i8] c"Failed \00", align 1 ; <[8 x i8]*> [#uses=1]
@.str2 = private constant [2 x i8] c"0\00", align 1 ; <[2 x i8]*> [#uses=1]
@.str3 = private constant [7 x i8] c"test.c\00", align 1 ; <[7 x i8]*> [#uses=1]
@__PRETTY_FUNCTION__.2067 = internal constant [13 x i8] c"aligned_func\00" ; <[13 x i8]*> [#uses=1]

define void @aligned_func(%struct.S* byval align 64 %obj) nounwind {
entry:
  %ptr = alloca i8*                               ; <i8**> [#uses=3]
  %p = alloca i64                                 ; <i64*> [#uses=3]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  %obj1 = bitcast %struct.S* %obj to i8*          ; <i8*> [#uses=1]
  store i8* %obj1, i8** %ptr, align 8
  %0 = load i8** %ptr, align 8                    ; <i8*> [#uses=1]
  %1 = ptrtoint i8* %0 to i64                     ; <i64> [#uses=1]
  store i64 %1, i64* %p, align 8
  %2 = load i8** %ptr, align 8                    ; <i8*> [#uses=1]
  %3 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([10 x i8]* @.str, i64 0, i64 0), i8* %2) nounwind ; <i32> [#uses=0]
  %4 = load i64* %p, align 8                      ; <i64> [#uses=1]
  %5 = and i64 %4, 140737488355264                ; <i64> [#uses=1]
  %6 = load i64* %p, align 8                      ; <i64> [#uses=1]
  %7 = icmp ne i64 %5, %6                         ; <i1> [#uses=1]
  br i1 %7, label %bb, label %bb2

bb:                                               ; preds = %entry
  %8 = call i32 @puts(i8* getelementptr inbounds ([8 x i8]* @.str1, i64 0, i64 0)) nounwind ; <i32> [#uses=0]
  call void @__assert_fail(i8* getelementptr inbounds ([2 x i8]* @.str2, i64 0, i64 0), i8* getelementptr inbounds ([7 x i8]* @.str3, i64 0, i64 0), i32 18, i8* getelementptr inbounds ([13 x i8]* @__PRETTY_FUNCTION__.2067, i64 0, i64 0)) noreturn nounwind
  unreachable

bb2:                                              ; preds = %entry
  br label %return

return:                                           ; preds = %bb2
  ret void
}

declare i32 @printf(i8*, ...) nounwind

declare i32 @puts(i8*)

declare void @__assert_fail(i8*, i8*, i32, i8*) noreturn nounwind

define void @main() nounwind {
entry:
; CHECK: main
; CHECK: andq    $-64, %rsp
  %s1 = alloca %struct.S                          ; <%struct.S*> [#uses=4]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  %0 = getelementptr inbounds %struct.S* %s1, i32 0, i32 0 ; <i32*> [#uses=1]
  store i32 1, i32* %0, align 4
  call void @aligned_func(%struct.S* byval align 64 %s1) nounwind
  br label %return

return:                                           ; preds = %entry
  ret void
}
