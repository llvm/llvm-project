; RUN: llc -mtriple=armv7a   %s -o - | FileCheck %s
; RUN: llc -mtriple=thumbv7m %s -o - | FileCheck %s
; RUN: llc -mtriple=thumbv6m %s -o - | FileCheck %s

@str.1 = private unnamed_addr constant [58 x i8] c"+-------------------------------------------------------+\00"
@str.2 = private unnamed_addr constant [58 x i8] c"|                                                       |\00"

declare i32 @puts(i8* nocapture readonly)

; Check that we rematerialize the adr of str.1 instead of doing one adr and two
; movs.

; CHECK: adr r0, [[STR1:.LCPI[0-9]+_[0-9]+]]
; CHECK: bl puts
; CHECK: adr r0, {{.LCPI[0-9]+_[0-9]+}}
; CHECK: bl puts
; CHECK: adr r0, [[STR1]]
; CHECK: b{{l?}} puts
define void @fn() {
entry:
  %puts1 = tail call i32 @puts(i8* getelementptr inbounds ([58 x i8], [58 x i8]* @str.1, i32 0, i32 0))
  %puts2 = tail call i32 @puts(i8* getelementptr inbounds ([58 x i8], [58 x i8]* @str.2, i32 0, i32 0))
  %puts3 = tail call i32 @puts(i8* getelementptr inbounds ([58 x i8], [58 x i8]* @str.1, i32 0, i32 0))
  ret void
}
