; RUN: llc < %s -mtriple=i386-linux-gnu -use-ctors | FileCheck %s --check-prefix=CHECK-DEFAULT
; RUN: llc < %s -mtriple=i386-apple-darwin | FileCheck %s --check-prefix=CHECK-DARWIN
; PR5329

@llvm.global_ctors = appending global [3 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 2000, void ()* @construct_2, i8* null }, { i32, void ()*, i8* } { i32 3000, void ()* @construct_3, i8* null }, { i32, void ()*, i8* } { i32 1000, void ()* @construct_1, i8* null }]
; CHECK-DEFAULT: .section        .ctors.62535,"aw",@progbits
; CHECK-DEFAULT: .long construct_3
; CHECK-DEFAULT: .section        .ctors.63535,"aw",@progbits
; CHECK-DEFAULT: .long construct_2
; CHECK-DEFAULT: .section        .ctors.64535,"aw",@progbits
; CHECK-DEFAULT: .long construct_1

; CHECK-DARWIN: .long _construct_1
; CHECK-DARWIN-NEXT: .long _construct_2
; CHECK-DARWIN-NEXT: .long _construct_3

@llvm.global_dtors = appending global [3 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 2000, void ()* @destruct_2, i8* null }, { i32, void ()*, i8* } { i32 1000, void ()* @destruct_1, i8* null }, { i32, void ()*, i8* } { i32 3000, void ()* @destruct_3, i8* null }]
; CHECK-DEFAULT: .section        .dtors.62535,"aw",@progbits
; CHECK-DEFAULT: .long destruct_3
; CHECK-DEFAULT: .section        .dtors.63535,"aw",@progbits
; CHECK-DEFAULT: .long destruct_2
; CHECK-DEFAULT: .section        .dtors.64535,"aw",@progbits
; CHECK-DEFAULT: .long destruct_1

; CHECK-DARWIN:      .long _destruct_1
; CHECK-DARWIN-NEXT: .long _destruct_2
; CHECK-DARWIN-NEXT: .long _destruct_3

declare void @construct_1()
declare void @construct_2()
declare void @construct_3()
declare void @destruct_1()
declare void @destruct_2()
declare void @destruct_3()
