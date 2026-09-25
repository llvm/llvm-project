; Test that two independent hierarchy trees (custom root + NSObject-based)
; resolve without conflict when the implementations are split across modules.
;
; Hierarchy:
;   Root (no superclass)
;     int a;                       // a at offset 8
;     `- RootSub
;          int b;                  // b at offset 12
;
;   NSObject (external, ABI-stable root)
;     `- ObjSub
;          int c;                  // c at offset 8
;
; RUN: rm -rf %t && split-file %s %t
; RUN: llvm-link %t/root.ll %t/root-sub.ll %t/obj-sub.ll -S -o %t/linked.ll
; RUN: opt -passes=objc-constant-ivar-offset -S %t/linked.ll | FileCheck %s

; CHECK: @"OBJC_IVAR_$_Root.a" = {{.*}}constant i64 8
; CHECK: @"OBJC_IVAR_$_RootSub.b" = {{.*}}constant i64 12
; CHECK: @"OBJC_IVAR_$_ObjSub.c" = {{.*}}constant i64 8

; Segments:
;   source.m - ingredients (requires Darwin SDK to regenerate)
;   gen      - refresher
;   *.ll     - hand-reduced cross-module IR

;--- source.m
@interface Root { int a; } @end
@interface RootSub : Root { int b; } @end
@implementation Root @end
@implementation RootSub @end

#import <objc/NSObject.h>
@interface ObjSub : NSObject { int c; } @end
@implementation ObjSub @end

;--- gen
clang -target x86_64-apple-macosx10.15 -fobjc-runtime=macosx-10.15 -S -emit-llvm -isysroot $(xcrun --show-sdk-path) source.m -o -

;--- root.ll
source_filename = "root.m"
target triple = "x86_64-apple-macosx10.15.0"

%struct._objc_cache = type opaque
%struct._class_t = type { ptr, ptr, ptr, ptr, ptr }
%struct._class_ro_t = type { i32, i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr }
%struct._ivar_t = type { ptr, ptr, ptr, i32, i32 }

@_objc_empty_cache = external global %struct._objc_cache
@RootName = private unnamed_addr constant [5 x i8] c"Root\00"
@IvarA = private unnamed_addr constant [2 x i8] c"a\00"
@IntTy = private unnamed_addr constant [2 x i8] c"i\00"
@"OBJC_IVAR_$_Root.a" = global i64 8, section "__DATA,__objc_ivar", align 8
@"_OBJC_IVARS_$_Root" = internal global { i32, i32, [1 x %struct._ivar_t] } { i32 32, i32 1, [1 x %struct._ivar_t] [%struct._ivar_t { ptr @"OBJC_IVAR_$_Root.a", ptr @IvarA, ptr @IntTy, i32 2, i32 4 }] }
@"_OBJC_METACLASS_RO_$_Root" = internal global %struct._class_ro_t { i32 1, i32 40, i32 40, ptr null, ptr @RootName, ptr null, ptr null, ptr null, ptr null, ptr null }
@"_OBJC_CLASS_RO_$_Root" = internal global %struct._class_ro_t { i32 0, i32 8, i32 12, ptr null, ptr @RootName, ptr null, ptr null, ptr @"_OBJC_IVARS_$_Root", ptr null, ptr null }
@"OBJC_METACLASS_$_Root" = global %struct._class_t { ptr @"OBJC_METACLASS_$_Root", ptr @"OBJC_METACLASS_$_Root", ptr @_objc_empty_cache, ptr null, ptr @"_OBJC_METACLASS_RO_$_Root" }
@"OBJC_CLASS_$_Root" = global %struct._class_t { ptr @"OBJC_METACLASS_$_Root", ptr null, ptr @_objc_empty_cache, ptr null, ptr @"_OBJC_CLASS_RO_$_Root" }
@RootClassList = private global [1 x ptr] [ptr @"OBJC_CLASS_$_Root"], section "__DATA,__objc_classlist,regular,no_dead_strip"
@llvm.compiler.used = appending global [1 x ptr] [ptr @RootClassList], section "llvm.metadata"

;--- root-sub.ll
source_filename = "root-sub.m"
target triple = "x86_64-apple-macosx10.15.0"

%struct._objc_cache = type opaque
%struct._class_t = type { ptr, ptr, ptr, ptr, ptr }
%struct._class_ro_t = type { i32, i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr }
%struct._ivar_t = type { ptr, ptr, ptr, i32, i32 }

@_objc_empty_cache = external global %struct._objc_cache
@"OBJC_CLASS_$_Root" = external global %struct._class_t
@"OBJC_METACLASS_$_Root" = external global %struct._class_t
@RootSubName = private unnamed_addr constant [8 x i8] c"RootSub\00"
@IvarB = private unnamed_addr constant [2 x i8] c"b\00"
@IntTy = private unnamed_addr constant [2 x i8] c"i\00"
@"OBJC_IVAR_$_RootSub.b" = global i64 12, section "__DATA,__objc_ivar", align 8
@"_OBJC_IVARS_$_RootSub" = internal global { i32, i32, [1 x %struct._ivar_t] } { i32 32, i32 1, [1 x %struct._ivar_t] [%struct._ivar_t { ptr @"OBJC_IVAR_$_RootSub.b", ptr @IvarB, ptr @IntTy, i32 2, i32 4 }] }
@"_OBJC_METACLASS_RO_$_RootSub" = internal global %struct._class_ro_t { i32 1, i32 40, i32 40, ptr null, ptr @RootSubName, ptr null, ptr null, ptr null, ptr null, ptr null }
@"_OBJC_CLASS_RO_$_RootSub" = internal global %struct._class_ro_t { i32 0, i32 12, i32 16, ptr null, ptr @RootSubName, ptr null, ptr null, ptr @"_OBJC_IVARS_$_RootSub", ptr null, ptr null }
@"OBJC_METACLASS_$_RootSub" = global %struct._class_t { ptr @"OBJC_METACLASS_$_Root", ptr @"OBJC_METACLASS_$_Root", ptr @_objc_empty_cache, ptr null, ptr @"_OBJC_METACLASS_RO_$_RootSub" }
@"OBJC_CLASS_$_RootSub" = global %struct._class_t { ptr @"OBJC_METACLASS_$_RootSub", ptr @"OBJC_CLASS_$_Root", ptr @_objc_empty_cache, ptr null, ptr @"_OBJC_CLASS_RO_$_RootSub" }
@RootSubClassList = private global [1 x ptr] [ptr @"OBJC_CLASS_$_RootSub"], section "__DATA,__objc_classlist,regular,no_dead_strip"
@llvm.compiler.used = appending global [1 x ptr] [ptr @RootSubClassList], section "llvm.metadata"

;--- obj-sub.ll
source_filename = "obj-sub.m"
target triple = "x86_64-apple-macosx10.15.0"

%struct._objc_cache = type opaque
%struct._class_t = type { ptr, ptr, ptr, ptr, ptr }
%struct._class_ro_t = type { i32, i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr }
%struct._ivar_t = type { ptr, ptr, ptr, i32, i32 }

@_objc_empty_cache = external global %struct._objc_cache
@"OBJC_CLASS_$_NSObject" = external global %struct._class_t
@"OBJC_METACLASS_$_NSObject" = external global %struct._class_t
@ObjSubName = private unnamed_addr constant [7 x i8] c"ObjSub\00"
@IvarC = private unnamed_addr constant [2 x i8] c"c\00"
@IntTy = private unnamed_addr constant [2 x i8] c"i\00"
@"OBJC_IVAR_$_ObjSub.c" = global i64 8, section "__DATA,__objc_ivar", align 8
@"_OBJC_IVARS_$_ObjSub" = internal global { i32, i32, [1 x %struct._ivar_t] } { i32 32, i32 1, [1 x %struct._ivar_t] [%struct._ivar_t { ptr @"OBJC_IVAR_$_ObjSub.c", ptr @IvarC, ptr @IntTy, i32 2, i32 4 }] }
@"_OBJC_METACLASS_RO_$_ObjSub" = internal global %struct._class_ro_t { i32 1, i32 40, i32 40, ptr null, ptr @ObjSubName, ptr null, ptr null, ptr null, ptr null, ptr null }
@"_OBJC_CLASS_RO_$_ObjSub" = internal global %struct._class_ro_t { i32 0, i32 8, i32 12, ptr null, ptr @ObjSubName, ptr null, ptr null, ptr @"_OBJC_IVARS_$_ObjSub", ptr null, ptr null }
@"OBJC_METACLASS_$_ObjSub" = global %struct._class_t { ptr @"OBJC_METACLASS_$_NSObject", ptr @"OBJC_METACLASS_$_NSObject", ptr @_objc_empty_cache, ptr null, ptr @"_OBJC_METACLASS_RO_$_ObjSub" }
@"OBJC_CLASS_$_ObjSub" = global %struct._class_t { ptr @"OBJC_METACLASS_$_ObjSub", ptr @"OBJC_CLASS_$_NSObject", ptr @_objc_empty_cache, ptr null, ptr @"_OBJC_CLASS_RO_$_ObjSub" }
@ObjSubClassList = private global [1 x ptr] [ptr @"OBJC_CLASS_$_ObjSub"], section "__DATA,__objc_classlist,regular,no_dead_strip"
@llvm.compiler.used = appending global [1 x ptr] [ptr @ObjSubClassList], section "llvm.metadata"
