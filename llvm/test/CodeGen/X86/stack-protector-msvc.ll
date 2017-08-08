
; RUN: llc -mtriple=i386-pc-windows-msvc < %s -o - | FileCheck -check-prefix=MSVC-I386 %s
; RUN: llc -mtriple=x86_64-pc-windows-msvc < %s -o - | FileCheck -check-prefix=MSVC-64 %s

; MSVC-I386: movl ___security_cookie, %[[REG1:[a-z]*]]
; MSVC-I386: movl %[[REG1]], [[SLOT:[0-9]*]](%esp)
; MSVC-I386: calll _strcpy
; MSVC-I386: movl [[SLOT]](%esp), %ecx
; MSVC-I386: calll @__security_check_cookie@4
; MSVC-I386: retl

; MSVC-64: movq __security_cookie(%rip), %[[REG1:[a-z]*]]
; MSVC-64: movq	%[[REG1]], [[SLOT:[0-9]*]](%rsp)
; MSVC-64: callq strcpy
; MSVC-64: movq [[SLOT]](%rsp), %rcx
; MSVC-64: callq __security_check_cookie

@"\01LC" = internal constant [11 x i8] c"buf == %s\0A\00"    ; <[11 x i8]*> [#uses=1]

define void @test(i8* %a) nounwind ssp {
entry:
 %a_addr = alloca i8*    ; <i8**> [#uses=2]
 %buf = alloca [8 x i8]    ; <[8 x i8]*> [#uses=2]
  %"alloca point" = bitcast i32 0 to i32   ; <i32> [#uses=0]
 store i8* %a, i8** %a_addr
 %buf1 = bitcast [8 x i8]* %buf to i8*   ; <i8*> [#uses=1]
 %0 = load i8*, i8** %a_addr, align 4    ; <i8*> [#uses=1]
 %1 = call i8* @strcpy(i8* %buf1, i8* %0) nounwind   ; <i8*> [#uses=0]
  %buf2 = bitcast [8 x i8]* %buf to i8*    ; <i8*> [#uses=1]
 %2 = call i32 (i8*, ...) @printf(i8* getelementptr ([11 x i8], [11 x i8]* @"\01LC", i32 0, i32 0), i8* %buf2) nounwind    ; <i32> [#uses=0]
 br label %return

return:    ; preds = %entry
 ret void
}

declare i8* @strcpy(i8*, i8*) nounwind

declare i32 @printf(i8*, ...) nounwind

