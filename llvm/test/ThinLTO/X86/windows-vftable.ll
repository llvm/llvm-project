;; Test an alias pointing to a GEP.
; RUN: opt -module-summary %s -o %t1.bc
; RUN: cp %t1.bc %t2.bc
; RUN: llvm-lto2 run %t1.bc %t2.bc -r=%t1.bc,"??_7bad_array_new_length@stdext@@6B@",pl -r=%t1.bc,"??_Gbad_array_new_length@stdext@@UEAAPEAXI@Z",pl \
; RUN:   -r=%t1.bc,"?_Throw_bad_array_new_length@std@@YAXXZ",pl \
; RUN:   -r=%t2.bc,"??_7bad_array_new_length@stdext@@6B@", -r=%t2.bc,"??_Gbad_array_new_length@stdext@@UEAAPEAXI@Z", \
; RUN:   -r=%t2.bc,"?_Throw_bad_array_new_length@std@@YAXXZ", -o %t3 --save-temps
; RUN: llvm-dis < %t3.2.1.promote.bc | FileCheck %s

; CHECK: @anon = private unnamed_addr constant { [2 x ptr] } { [2 x ptr] [ptr null, ptr @"??_Gbad_array_new_length@stdext@@UEAAPEAXI@Z"] }, comdat($"??_7bad_array_new_length@stdext@@6B@")
; CHECK: @"??_7bad_array_new_length@stdext@@6B@" = unnamed_addr alias ptr, getelementptr inbounds ({ [4 x ptr] }, ptr @anon, i32 0, i32 0, i32 1){{$}}
; CHECK: define available_externally dso_local noundef ptr @"??_Gbad_array_new_length@stdext@@UEAAPEAXI@Z"(ptr noundef nonnull %this) {
; CHECK: define available_externally dso_local void @"?_Throw_bad_array_new_length@std@@YAXXZ"(ptr noundef nonnull %0) unnamed_addr {

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.26.0"

$"??_7bad_array_new_length@stdext@@6B@" = comdat largest
$"??_Gbad_array_new_length@stdext@@UEAAPEAXI@Z" = comdat any
$"?_Throw_bad_array_new_length@std@@YAXXZ" = comdat any

@anon = private unnamed_addr constant { [2 x ptr] } { [2 x ptr] [ptr null, ptr @"??_Gbad_array_new_length@stdext@@UEAAPEAXI@Z"] }, comdat($"??_7bad_array_new_length@stdext@@6B@")

@"??_7bad_array_new_length@stdext@@6B@" = unnamed_addr alias ptr, getelementptr inbounds ({ [4 x ptr] }, ptr @anon, i32 0, i32 0, i32 1)

define linkonce_odr dso_local noundef ptr @"??_Gbad_array_new_length@stdext@@UEAAPEAXI@Z"(ptr noundef nonnull %this) comdat {
entry:
  ret ptr %this
}

define linkonce_odr dso_local void @"?_Throw_bad_array_new_length@std@@YAXXZ"(ptr noundef nonnull %0) unnamed_addr comdat {
entry:
  store ptr @"??_7bad_array_new_length@stdext@@6B@", ptr %0, align 8
  ret void
}
