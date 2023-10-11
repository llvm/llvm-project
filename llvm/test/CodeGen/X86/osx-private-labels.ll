; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck %s
; Test all the cases where a L label is safe. Removing any entry from
; TargetLoweringObjectFileMachO::isSectionAtomizableBySymbols should cause
; this to fail.
; We also test some noteworthy cases that require an l label.

@private1 = private unnamed_addr constant [4 x i8] c"zed\00"
; CHECK: .section	__TEXT,__cstring,cstring_literals
; CHECK-NEXT: L_private1:

@private2 = private unnamed_addr constant [5 x i16] [i16 116, i16 101,
                                                     i16 115, i16 116, i16 0]
; CHECK: .section	__TEXT,__ustring
; CHECK-NEXT: .p2align	1
; CHECK-NEXT: l_private2:

; There is no dedicated 4 byte strings on MachO.

%struct.NSConstantString = type { i32*, i32, i8*, i32 }
@private3 = private constant %struct.NSConstantString { i32* null, i32 1992, i8* null, i32 0 }, section "__DATA,__cfstring"
; CHECK: .section	__DATA,__cfstring
; CHECK-NEXT: .p2align	4
; CHECK-NEXT: L_private3:

; There is no dedicated 1 or 2 byte constant section on MachO.

@private4 = private unnamed_addr constant i32 42
; CHECK: .section	__TEXT,__literal4,4byte_literals
; CHECK-NEXT: .p2align	2
; CHECK-NEXT: L_private4:

@private5 = private unnamed_addr constant i64 42
; CHECK: .section	__TEXT,__literal8,8byte_literals
; CHECK-NEXT: .p2align	3
; CHECK-NEXT: L_private5:

@private6 = private unnamed_addr constant i128 42
; CHECK: .section	__TEXT,__literal16,16byte_literals
; CHECK-NEXT: .p2align	4
; CHECK-NEXT: L_private6:

%struct._objc_class = type { i8* }
@private7 = private global %struct._objc_class* null, section "__OBJC,__cls_refs,literal_pointers,no_dead_strip"
; CHECK: .section	__OBJC,__cls_refs,literal_pointers,no_dead_strip
; CHECK: .p2align	3
; CHECK: L_private7:

@private8 = private global i32* null, section "__DATA,__nl_symbol_ptr,non_lazy_symbol_pointers"
; CHECK: .section	__DATA,__nl_symbol_ptr,non_lazy_symbol_pointers
; CHECK-NEXT: .p2align	3
; CHECK-NEXT: L_private8:

@private9 = private global i32* null, section "__DATA,__la_symbol_ptr,lazy_symbol_pointers"
; CHECK: .section	__DATA,__la_symbol_ptr,lazy_symbol_pointers
; CHECK-NEXT: .p2align	3
; CHECK-NEXT: L_private9:

@private10 = private global i32* null, section "__DATA,__mod_init_func,mod_init_funcs"
; CHECK: .section	__DATA,__mod_init_func,mod_init_funcs
; CHECK-NEXT: .p2align	3
; CHECK-NEXT: L_private10:

@private11 = private global i32* null, section "__DATA,__mod_term_func,mod_term_funcs"
; CHECK: .section	__DATA,__mod_term_func,mod_term_funcs
; CHECK-NEXT: .p2align	3
; CHECK-NEXT: L_private11:

@private12 = private global i32* null, section "__DATA,__foobar,interposing"
; CHECK: .section	__DATA,__foobar,interposing
; CHECK-NEXT: .p2align	3
; CHECK-NEXT: L_private12:

@private13 = private global i32 42, section "__DATA, __objc_classlist, regular, no_dead_strip"
; CHECK: .section	__DATA,__objc_classlist,regular,no_dead_strip
; CHECK-NEXT: .p2align	2
; CHECK-NEXT: _private13:

@private14 = private global [4 x i8] c"zed\00", section "__TEXT,__objc_classname,cstring_literals"
; CHECK: .section	__TEXT,__objc_classname,cstring_literals
; CHECK-NEXT: L_private14:

@private15 = private global [4 x i8] c"zed\00", section "__TEXT,__objc_methname,cstring_literals"
; CHECK: .section	__TEXT,__objc_methname,cstring_literals
; CHECK-NEXT: L_private15:

@private16 = private global [4 x i8] c"zed\00", section "__TEXT,__objc_methtype,cstring_literals"
; CHECK: .section	__TEXT,__objc_methtype,cstring_literals
; CHECK-NEXT: L_private16:
