; This test checks that instrumented global C (null terminated) strings are put into a special section on Darwin.
; RUN: opt < %s -asan -asan-module -enable-new-pm=0 -S | FileCheck %s
; RUN: opt < %s -passes='asan-pipeline' -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

; Should be put into __asan_cstring section:
@.str.1 = private unnamed_addr constant [13 x i8] c"Hello world.\00", align 1
@.str.2 = private unnamed_addr constant [4 x i8] c"%s\0A\00", align 1

; CHECK: @.str.1 = internal constant { [13 x i8], [19 x i8] } { [13 x i8] c"Hello world.\00", [19 x i8] zeroinitializer }, section "__TEXT,__asan_cstring,regular", align 32
; CHECK: @.str.2 = internal constant { [4 x i8], [28 x i8] } { [4 x i8] c"%s\0A\00", [28 x i8] zeroinitializer }, section "__TEXT,__asan_cstring,regular", align 32

; Shouldn't be put into special section:
@.str.3 = private unnamed_addr constant [4 x i8] c"\00\01\02\03", align 1
@.str.4 = private unnamed_addr global [7 x i8] c"Hello.\00", align 1
@.str.5 = private unnamed_addr constant [8 x i8] c"Hello.\00\00", align 1

; CHECK: @.str.3 = internal constant { [4 x i8], [28 x i8] } { [4 x i8] c"\00\01\02\03", [28 x i8] zeroinitializer }, align 32
; CHECK: @.str.4 = private global { [7 x i8], [25 x i8] } { [7 x i8] c"Hello.\00", [25 x i8] zeroinitializer }, align 32
; CHECK: @.str.5 = internal constant { [8 x i8], [24 x i8] } { [8 x i8] c"Hello.\00\00", [24 x i8] zeroinitializer }, align 32
