; RUN: llc < %s -mtriple=i386-linux-gnu -use-ctors | FileCheck %s --check-prefix=CHECK-DEFAULT
; RUN: llc < %s -mtriple=i386-apple-darwin | FileCheck %s --check-prefix=CHECK-DARWIN
; PR5329

@llvm.global_ctors = appending global [3 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 2000, ptr @construct_2, ptr null }, { i32, ptr, ptr } { i32 3000, ptr @construct_3, ptr null }, { i32, ptr, ptr } { i32 1000, ptr @construct_1, ptr null }]
; CHECK-DEFAULT: .section        .ctors.62535,"aw",@progbits
; CHECK-DEFAULT: .long construct_3
; CHECK-DEFAULT: .section        .ctors.63535,"aw",@progbits
; CHECK-DEFAULT: .long construct_2
; CHECK-DEFAULT: .section        .ctors.64535,"aw",@progbits
; CHECK-DEFAULT: .long construct_1

; CHECK-DARWIN-LABEL: .section	__DATA,__mod_init_func,mod_init_funcs
; CHECK-DARWIN:      .long _construct_1
; CHECK-DARWIN-NEXT: .long l_register_call_dtors.1000
; CHECK-DARWIN-NEXT: .long _construct_2
; CHECK-DARWIN-NEXT: .long l_register_call_dtors.2000
; CHECK-DARWIN-NEXT: .long _construct_3
; CHECK-DARWIN-NEXT: .long l_register_call_dtors.3000

@llvm.global_dtors = appending global [3 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 2000, ptr @destruct_2, ptr null }, { i32, ptr, ptr } { i32 1000, ptr @destruct_1, ptr null }, { i32, ptr, ptr } { i32 3000, ptr @destruct_3, ptr null }]
; CHECK-DEFAULT: .section        .dtors.62535,"aw",@progbits
; CHECK-DEFAULT: .long destruct_3
; CHECK-DEFAULT: .section        .dtors.63535,"aw",@progbits
; CHECK-DEFAULT: .long destruct_2
; CHECK-DEFAULT: .section        .dtors.64535,"aw",@progbits
; CHECK-DEFAULT: .long destruct_1

; CHECK-DARWIN-NOT: mod_term_func

declare void @construct_1()
declare void @construct_2()
declare void @construct_3()
declare void @destruct_1()
declare void @destruct_2()
declare void @destruct_3()
