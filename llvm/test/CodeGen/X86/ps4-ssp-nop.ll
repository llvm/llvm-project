; Verify that a ud2 is generated after the call to __stack_chk_fail.

; RUN: llc < %s -mtriple=x86_64-scei-ps4 -enable-selectiondag-sp=false -O0 -o - | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-sie-ps5  -enable-selectiondag-sp=false -O0 -o - | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-scei-ps4 -enable-selectiondag-sp=false -O2 -o - | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-sie-ps5  -enable-selectiondag-sp=false -O2 -o - | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-scei-ps4 -enable-selectiondag-sp=true  -O0 -o - | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-sie-ps5  -enable-selectiondag-sp=true  -O0 -o - | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-scei-ps4 -enable-selectiondag-sp=true  -O2 -o - | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-sie-ps5  -enable-selectiondag-sp=true  -O2 -o - | FileCheck %s


; CHECK: check_input:
; CHECK: callq __stack_chk_fail
; CHECK-NEXT: ud2
; CHECK: .size	check_input
; CHECK-NEXT: .cfi_endproc

@.str = private unnamed_addr constant [37 x i8] c"????????????????????????????????????\00", align 1

define signext i8 @check_input(ptr %input) nounwind uwtable ssp {
entry:
  %input.addr = alloca ptr, align 8
  %buf = alloca [16 x i8], align 16
  store ptr %input, ptr %input.addr, align 8
  %0 = load ptr, ptr %input.addr, align 8
  %call = call ptr @strcpy(ptr %buf, ptr %0) nounwind
  %1 = load i8, ptr %buf, align 1
  ret i8 %1
}

declare ptr @strcpy(ptr, ptr) nounwind

define i32 @main() nounwind uwtable ssp {
entry:
  %retval = alloca i32, align 4
  store i32 0, ptr %retval
  %call = call signext i8 @check_input(ptr @.str)
  %conv = sext i8 %call to i32
  ret i32 %conv
}
