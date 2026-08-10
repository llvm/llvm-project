; RUN: llc -mtriple aarch64-apple-darwin -O0 -global-isel -verify-machineinstrs %s -o - 2>&1 | FileCheck %s
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios13.0.0"

@t_val = thread_local global i32 0, align 4
@.str = private unnamed_addr constant [5 x i8] c"str1\00", align 1
@str1 = global ptr @.str, align 8
@.str.1 = private unnamed_addr constant [5 x i8] c"str2\00", align 1
@str2 = global ptr @.str.1, align 8
@.str.2 = private unnamed_addr constant [5 x i8] c"str3\00", align 1
@str3 = global ptr @.str.2, align 8
@.str.3 = private unnamed_addr constant [5 x i8] c"str4\00", align 1
@str4 = global ptr @.str.3, align 8
@.str.4 = private unnamed_addr constant [5 x i8] c"str5\00", align 1
@str5 = global ptr @.str.4, align 8
@.str.5 = private unnamed_addr constant [5 x i8] c"str6\00", align 1
@str6 = global ptr @.str.5, align 8
@.str.6 = private unnamed_addr constant [5 x i8] c"str7\00", align 1
@str7 = global ptr @.str.6, align 8
@.str.7 = private unnamed_addr constant [5 x i8] c"str8\00", align 1
@str8 = global ptr @.str.7, align 8
@.str.8 = private unnamed_addr constant [5 x i8] c"str9\00", align 1
@str9 = global ptr @.str.8, align 8
@.str.9 = private unnamed_addr constant [6 x i8] c"str10\00", align 1
@str10 = global ptr @.str.9, align 8
@.str.10 = private unnamed_addr constant [4 x i8] c"%s\0A\00", align 1
@.str.11 = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1
@.str.12 = private unnamed_addr constant [4 x i8] c"xyz\00", align 1


; This test checks that we don't re-use the register for the variable descriptor
; for the second ldr.
; CHECK:        adrp	x[[PTR1:[0-9]+]], _t_val@TLVPPAGE
; CHECK:	ldr	x0, [x[[PTR1]], _t_val@TLVPPAGEOFF]
; CHECK:	ldr	x[[FPTR:[0-9]+]], [x0]
; CHECK:        blr     x[[FPTR]]

define void @_Z4funcPKc(ptr %id) {
entry:
  %id.addr = alloca ptr, align 8
  store ptr %id, ptr %id.addr, align 8
  %0 = load ptr, ptr %id.addr, align 8
  %1 = load ptr, ptr @str1, align 8
  %cmp = icmp eq ptr %0, %1
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %2 = load ptr, ptr @str1, align 8
  %call = call i32 (ptr, ...) @printf(ptr @.str.10, ptr %2)
  %3 = load ptr, ptr @str2, align 8
  %call1 = call i32 (ptr, ...) @printf(ptr @.str.10, ptr %3)
  %4 = load ptr, ptr @str3, align 8
  %call2 = call i32 (ptr, ...) @printf(ptr @.str.10, ptr %4)
  %5 = load ptr, ptr @str4, align 8
  %call3 = call i32 (ptr, ...) @printf(ptr @.str.10, ptr %5)
  %6 = load ptr, ptr @str5, align 8
  %call4 = call i32 (ptr, ...) @printf(ptr @.str.10, ptr %6)
  %7 = load ptr, ptr @str6, align 8
  %call5 = call i32 (ptr, ...) @printf(ptr @.str.10, ptr %7)
  %8 = load ptr, ptr @str7, align 8
  %call6 = call i32 (ptr, ...) @printf(ptr @.str.10, ptr %8)
  %9 = load ptr, ptr @str8, align 8
  %call7 = call i32 (ptr, ...) @printf(ptr @.str.10, ptr %9)
  %10 = load ptr, ptr @str9, align 8
  %call8 = call i32 (ptr, ...) @printf(ptr @.str.10, ptr %10)
  %11 = load ptr, ptr @str10, align 8
  %call9 = call i32 (ptr, ...) @printf(ptr @.str.10, ptr %11)
  %12 = load i32, ptr @t_val, align 4
  %call10 = call i32 (ptr, ...) @printf(ptr @.str.11, i32 %12)
  br label %if.end56

if.else:                                          ; preds = %entry
  %13 = load ptr, ptr %id.addr, align 8
  %14 = load ptr, ptr @str2, align 8
  %cmp11 = icmp eq ptr %13, %14
  br i1 %cmp11, label %if.then12, label %if.else24

if.then12:                                        ; preds = %if.else
  %15 = load ptr, ptr @str1, align 8
  %call13 = call i32 (ptr, ...) @printf(ptr @.str.10, ptr %15)
  %16 = load ptr, ptr @str2, align 8
  %call14 = call i32 (ptr, ...) @printf(ptr @.str.10, ptr %16)
  %17 = load ptr, ptr @str3, align 8
  %call15 = call i32 (ptr, ...) @printf(ptr @.str.10, ptr %17)
  %18 = load ptr, ptr @str4, align 8
  %call16 = call i32 (ptr, ...) @printf(ptr @.str.10, ptr %18)
  %19 = load ptr, ptr @str5, align 8
  %call17 = call i32 (ptr, ...) @printf(ptr @.str.10, ptr %19)
  %20 = load ptr, ptr @str6, align 8
  %call18 = call i32 (ptr, ...) @printf(ptr @.str.10, ptr %20)
  %21 = load ptr, ptr @str7, align 8
  %call19 = call i32 (ptr, ...) @printf(ptr @.str.10, ptr %21)
  %22 = load ptr, ptr @str8, align 8
  %call20 = call i32 (ptr, ...) @printf(ptr @.str.10, ptr %22)
  %23 = load ptr, ptr @str9, align 8
  %call21 = call i32 (ptr, ...) @printf(ptr @.str.10, ptr %23)
  %24 = load ptr, ptr @str10, align 8
  %call22 = call i32 (ptr, ...) @printf(ptr @.str.10, ptr %24)
  %25 = load i32, ptr @t_val, align 4
  %call23 = call i32 (ptr, ...) @printf(ptr @.str.11, i32 %25)
  br label %if.end55

if.else24:                                        ; preds = %if.else
  %26 = load ptr, ptr %id.addr, align 8
  %27 = load ptr, ptr @str3, align 8
  %cmp25 = icmp eq ptr %26, %27
  br i1 %cmp25, label %if.then26, label %if.else27

if.then26:                                        ; preds = %if.else24
  br label %if.end54

if.else27:                                        ; preds = %if.else24
  %28 = load ptr, ptr %id.addr, align 8
  %29 = load ptr, ptr @str4, align 8
  %cmp28 = icmp eq ptr %28, %29
  br i1 %cmp28, label %if.then29, label %if.else30

if.then29:                                        ; preds = %if.else27
  br label %if.end53

if.else30:                                        ; preds = %if.else27
  %30 = load ptr, ptr %id.addr, align 8
  %31 = load ptr, ptr @str5, align 8
  %cmp31 = icmp eq ptr %30, %31
  br i1 %cmp31, label %if.then32, label %if.else33

if.then32:                                        ; preds = %if.else30
  br label %if.end52

if.else33:                                        ; preds = %if.else30
  %32 = load ptr, ptr %id.addr, align 8
  %33 = load ptr, ptr @str6, align 8
  %cmp34 = icmp eq ptr %32, %33
  br i1 %cmp34, label %if.then35, label %if.else36

if.then35:                                        ; preds = %if.else33
  br label %if.end51

if.else36:                                        ; preds = %if.else33
  %34 = load ptr, ptr %id.addr, align 8
  %35 = load ptr, ptr @str7, align 8
  %cmp37 = icmp eq ptr %34, %35
  br i1 %cmp37, label %if.then38, label %if.else39

if.then38:                                        ; preds = %if.else36
  br label %if.end50

if.else39:                                        ; preds = %if.else36
  %36 = load ptr, ptr %id.addr, align 8
  %37 = load ptr, ptr @str8, align 8
  %cmp40 = icmp eq ptr %36, %37
  br i1 %cmp40, label %if.then41, label %if.else42

if.then41:                                        ; preds = %if.else39
  br label %if.end49

if.else42:                                        ; preds = %if.else39
  %38 = load ptr, ptr %id.addr, align 8
  %39 = load ptr, ptr @str9, align 8
  %cmp43 = icmp eq ptr %38, %39
  br i1 %cmp43, label %if.then44, label %if.else45

if.then44:                                        ; preds = %if.else42
  br label %if.end48

if.else45:                                        ; preds = %if.else42
  %40 = load ptr, ptr %id.addr, align 8
  %41 = load ptr, ptr @str10, align 8
  %cmp46 = icmp eq ptr %40, %41
  br i1 %cmp46, label %if.then47, label %if.end

if.then47:                                        ; preds = %if.else45
  br label %if.end

if.end:                                           ; preds = %if.then47, %if.else45
  br label %if.end48

if.end48:                                         ; preds = %if.end, %if.then44
  br label %if.end49

if.end49:                                         ; preds = %if.end48, %if.then41
  br label %if.end50

if.end50:                                         ; preds = %if.end49, %if.then38
  br label %if.end51

if.end51:                                         ; preds = %if.end50, %if.then35
  br label %if.end52

if.end52:                                         ; preds = %if.end51, %if.then32
  br label %if.end53

if.end53:                                         ; preds = %if.end52, %if.then29
  br label %if.end54

if.end54:                                         ; preds = %if.end53, %if.then26
  br label %if.end55

if.end55:                                         ; preds = %if.end54, %if.then12
  br label %if.end56

if.end56:                                         ; preds = %if.end55, %if.then
  ret void
}
declare i32 @printf(ptr, ...)

