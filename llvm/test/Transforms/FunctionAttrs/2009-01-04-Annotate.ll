; RUN: opt < %s -functionattrs -S | FileCheck %s

; CHECK: declare noalias i8* @fopen(i8* nocapture readonly, i8* nocapture readonly) #0
declare i8* @fopen(i8*, i8*)

; CHECK: declare i8 @strlen(i8* nocapture) #1
declare i8 @strlen(i8*)

; CHECK: declare noalias i32* @realloc(i32* nocapture, i32) #0
declare i32* @realloc(i32*, i32)

; Test deliberately wrong declaration
declare i32 @strcpy(...)

; CHECK-NOT: strcpy{{.*}}noalias
; CHECK-NOT: strcpy{{.*}}nocapture
; CHECK-NOT: strcpy{{.*}}nounwind
; CHECK-NOT: strcpy{{.*}}readonly

; CHECK: attributes #0 = { nounwind }
; CHECK: attributes #1 = { nounwind readonly }
