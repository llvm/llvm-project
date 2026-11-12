; Test that a hierarchy split across modules is constified by the FullLTO
; post-link pipeline. The frontend cannot constify SubClass while SuperClass is
; only a declaration in the SubClass translation unit.
;
; Hierarchy:
;   NSObject
;     `- SuperClass
;          int x;                   // x at offset 8
;          `- SubClass
;               int y;              // y at offset 12
;
; RUN: rm -rf %t && split-file %s %t
; RUN: llvm-link %t/super.ll %t/sub.ll -S -o %t/linked.ll
; RUN: opt -passes=objc-constant-ivar-offset -S %t/linked.ll | FileCheck %s

; CHECK: @"OBJC_IVAR_$_SuperClass.x" = {{.*}}constant i64 8
; CHECK: @"OBJC_IVAR_$_SubClass.y" = {{.*}}constant i64 12

; Segments:
;   super.m / sub.m  - ingredients (requires Darwin SDK to regenerate)
;   gen              - refresher
;   super.ll / sub.ll - hand-reduced cross-module IR

;--- super.m
#import <objc/NSObject.h>
@interface SuperClass : NSObject { int x; } @end
@implementation SuperClass @end

;--- sub.m
#import <objc/NSObject.h>
@interface SuperClass : NSObject @end
@interface SubClass : SuperClass { int y; } @end
@implementation SubClass @end

;--- gen
clang -target x86_64-apple-macosx10.15 -fobjc-runtime=macosx-10.15 -S -emit-llvm -isysroot $(xcrun --show-sdk-path) super.m -o super.ll
clang -target x86_64-apple-macosx10.15 -fobjc-runtime=macosx-10.15 -S -emit-llvm -isysroot $(xcrun --show-sdk-path) sub.m -o sub.ll

;--- super.ll
source_filename = "super.m"
target triple = "x86_64-apple-macosx10.15.0"

%struct._objc_cache = type opaque
%struct._class_t = type { ptr, ptr, ptr, ptr, ptr }
%struct._class_ro_t = type { i32, i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr }
%struct._ivar_t = type { ptr, ptr, ptr, i32, i32 }

@_objc_empty_cache = external global %struct._objc_cache
@"OBJC_CLASS_$_NSObject" = external global %struct._class_t
@"OBJC_METACLASS_$_NSObject" = external global %struct._class_t
@SuperName = private unnamed_addr constant [11 x i8] c"SuperClass\00"
@IvarX = private unnamed_addr constant [2 x i8] c"x\00"
@IntTy = private unnamed_addr constant [2 x i8] c"i\00"
@"OBJC_IVAR_$_SuperClass.x" = global i64 8, section "__DATA,__objc_ivar", align 8
@"_OBJC_IVARS_$_SuperClass" = internal global { i32, i32, [1 x %struct._ivar_t] } { i32 32, i32 1, [1 x %struct._ivar_t] [%struct._ivar_t { ptr @"OBJC_IVAR_$_SuperClass.x", ptr @IvarX, ptr @IntTy, i32 2, i32 4 }] }
@"_OBJC_METACLASS_RO_$_SuperClass" = internal global %struct._class_ro_t { i32 1, i32 40, i32 40, ptr null, ptr @SuperName, ptr null, ptr null, ptr null, ptr null, ptr null }
@"_OBJC_CLASS_RO_$_SuperClass" = internal global %struct._class_ro_t { i32 0, i32 8, i32 12, ptr null, ptr @SuperName, ptr null, ptr null, ptr @"_OBJC_IVARS_$_SuperClass", ptr null, ptr null }
@"OBJC_METACLASS_$_SuperClass" = global %struct._class_t { ptr @"OBJC_METACLASS_$_NSObject", ptr @"OBJC_METACLASS_$_NSObject", ptr @_objc_empty_cache, ptr null, ptr @"_OBJC_METACLASS_RO_$_SuperClass" }
@"OBJC_CLASS_$_SuperClass" = global %struct._class_t { ptr @"OBJC_METACLASS_$_SuperClass", ptr @"OBJC_CLASS_$_NSObject", ptr @_objc_empty_cache, ptr null, ptr @"_OBJC_CLASS_RO_$_SuperClass" }
@SuperClassList = private global [1 x ptr] [ptr @"OBJC_CLASS_$_SuperClass"], section "__DATA,__objc_classlist,regular,no_dead_strip"
@llvm.compiler.used = appending global [1 x ptr] [ptr @SuperClassList], section "llvm.metadata"

;--- sub.ll
source_filename = "sub.m"
target triple = "x86_64-apple-macosx10.15.0"

%struct._objc_cache = type opaque
%struct._class_t = type { ptr, ptr, ptr, ptr, ptr }
%struct._class_ro_t = type { i32, i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr }
%struct._ivar_t = type { ptr, ptr, ptr, i32, i32 }

@_objc_empty_cache = external global %struct._objc_cache
@"OBJC_CLASS_$_SuperClass" = external global %struct._class_t
@"OBJC_METACLASS_$_SuperClass" = external global %struct._class_t
@"OBJC_METACLASS_$_NSObject" = external global %struct._class_t
@SubName = private unnamed_addr constant [9 x i8] c"SubClass\00"
@IvarY = private unnamed_addr constant [2 x i8] c"y\00"
@IntTy = private unnamed_addr constant [2 x i8] c"i\00"
@"OBJC_IVAR_$_SubClass.y" = global i64 12, section "__DATA,__objc_ivar", align 8
@"_OBJC_IVARS_$_SubClass" = internal global { i32, i32, [1 x %struct._ivar_t] } { i32 32, i32 1, [1 x %struct._ivar_t] [%struct._ivar_t { ptr @"OBJC_IVAR_$_SubClass.y", ptr @IvarY, ptr @IntTy, i32 2, i32 4 }] }
@"_OBJC_METACLASS_RO_$_SubClass" = internal global %struct._class_ro_t { i32 1, i32 40, i32 40, ptr null, ptr @SubName, ptr null, ptr null, ptr null, ptr null, ptr null }
@"_OBJC_CLASS_RO_$_SubClass" = internal global %struct._class_ro_t { i32 0, i32 12, i32 16, ptr null, ptr @SubName, ptr null, ptr null, ptr @"_OBJC_IVARS_$_SubClass", ptr null, ptr null }
@"OBJC_METACLASS_$_SubClass" = global %struct._class_t { ptr @"OBJC_METACLASS_$_NSObject", ptr @"OBJC_METACLASS_$_SuperClass", ptr @_objc_empty_cache, ptr null, ptr @"_OBJC_METACLASS_RO_$_SubClass" }
@"OBJC_CLASS_$_SubClass" = global %struct._class_t { ptr @"OBJC_METACLASS_$_SubClass", ptr @"OBJC_CLASS_$_SuperClass", ptr @_objc_empty_cache, ptr null, ptr @"_OBJC_CLASS_RO_$_SubClass" }
@SubClassList = private global [1 x ptr] [ptr @"OBJC_CLASS_$_SubClass"], section "__DATA,__objc_classlist,regular,no_dead_strip"
@llvm.compiler.used = appending global [1 x ptr] [ptr @SubClassList], section "llvm.metadata"
