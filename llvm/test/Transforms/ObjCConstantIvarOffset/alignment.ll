; Test that ivar sliding uses the subclass's max ivar alignment.
;
; Hierarchy:
;   NSObject
;     `- Super
;          char c;                  // c at offset 8
;          int hidden1;             // hidden1 at offset 12
;          int hidden2;             // hidden2 at offset 16
;          `- Sub
;               double d;           // d compiled at 16, slid to 24
;
; RUN: rm -rf %t && split-file %s %t
; RUN: llvm-link %t/super.ll %t/sub.ll -S -o %t/linked.ll
; RUN: opt -passes=objc-constant-ivar-offset -S %t/linked.ll | FileCheck %s

; CHECK: @"OBJC_IVAR_$_Sub.d" = {{.*}}constant i64 24
; CHECK: @"_OBJC_CLASS_RO_$_Sub" = internal global %struct._class_ro_t { i32 0, i32 24, i32 32,

; Segments:
;   super.m / sub.m  - ingredients (requires Darwin SDK to regenerate)
;   gen              - refresher
;   super.ll / sub.ll - hand-reduced cross-module IR

;--- super.m
#import <objc/NSObject.h>
@interface Super : NSObject { char c; } @end
@interface Super () { int hidden1; int hidden2; } @end
@implementation Super @end

;--- sub.m
#import <objc/NSObject.h>
@interface Super : NSObject { char c; } @end
@interface Sub : Super { double d; } @end
@implementation Sub @end

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
@SuperName = private unnamed_addr constant [6 x i8] c"Super\00"
@IvarC = private unnamed_addr constant [2 x i8] c"c\00"
@IvarHidden1 = private unnamed_addr constant [8 x i8] c"hidden1\00"
@IvarHidden2 = private unnamed_addr constant [8 x i8] c"hidden2\00"
@CharTy = private unnamed_addr constant [2 x i8] c"c\00"
@IntTy = private unnamed_addr constant [2 x i8] c"i\00"
@"OBJC_IVAR_$_Super.c" = global i64 8, section "__DATA,__objc_ivar", align 8
@"OBJC_IVAR_$_Super.hidden1" = global i64 12, section "__DATA,__objc_ivar", align 8
@"OBJC_IVAR_$_Super.hidden2" = global i64 16, section "__DATA,__objc_ivar", align 8
@"_OBJC_IVARS_$_Super" = internal global { i32, i32, [3 x %struct._ivar_t] } { i32 32, i32 3, [3 x %struct._ivar_t] [%struct._ivar_t { ptr @"OBJC_IVAR_$_Super.c", ptr @IvarC, ptr @CharTy, i32 0, i32 1 }, %struct._ivar_t { ptr @"OBJC_IVAR_$_Super.hidden1", ptr @IvarHidden1, ptr @IntTy, i32 2, i32 4 }, %struct._ivar_t { ptr @"OBJC_IVAR_$_Super.hidden2", ptr @IvarHidden2, ptr @IntTy, i32 2, i32 4 }] }
@"_OBJC_METACLASS_RO_$_Super" = internal global %struct._class_ro_t { i32 1, i32 40, i32 40, ptr null, ptr @SuperName, ptr null, ptr null, ptr null, ptr null, ptr null }
@"_OBJC_CLASS_RO_$_Super" = internal global %struct._class_ro_t { i32 0, i32 8, i32 20, ptr null, ptr @SuperName, ptr null, ptr null, ptr @"_OBJC_IVARS_$_Super", ptr null, ptr null }
@"OBJC_METACLASS_$_Super" = global %struct._class_t { ptr @"OBJC_METACLASS_$_NSObject", ptr @"OBJC_METACLASS_$_NSObject", ptr @_objc_empty_cache, ptr null, ptr @"_OBJC_METACLASS_RO_$_Super" }
@"OBJC_CLASS_$_Super" = global %struct._class_t { ptr @"OBJC_METACLASS_$_Super", ptr @"OBJC_CLASS_$_NSObject", ptr @_objc_empty_cache, ptr null, ptr @"_OBJC_CLASS_RO_$_Super" }
@SuperClassList = private global [1 x ptr] [ptr @"OBJC_CLASS_$_Super"], section "__DATA,__objc_classlist,regular,no_dead_strip"
@llvm.compiler.used = appending global [1 x ptr] [ptr @SuperClassList], section "llvm.metadata"

;--- sub.ll
source_filename = "sub.m"
target triple = "x86_64-apple-macosx10.15.0"

%struct._objc_cache = type opaque
%struct._class_t = type { ptr, ptr, ptr, ptr, ptr }
%struct._class_ro_t = type { i32, i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr }
%struct._ivar_t = type { ptr, ptr, ptr, i32, i32 }

@_objc_empty_cache = external global %struct._objc_cache
@"OBJC_CLASS_$_Super" = external global %struct._class_t
@"OBJC_METACLASS_$_Super" = external global %struct._class_t
@"OBJC_METACLASS_$_NSObject" = external global %struct._class_t
@SubName = private unnamed_addr constant [4 x i8] c"Sub\00"
@IvarD = private unnamed_addr constant [2 x i8] c"d\00"
@DoubleTy = private unnamed_addr constant [2 x i8] c"d\00"
@"OBJC_IVAR_$_Sub.d" = global i64 16, section "__DATA,__objc_ivar", align 8
@"_OBJC_IVARS_$_Sub" = internal global { i32, i32, [1 x %struct._ivar_t] } { i32 32, i32 1, [1 x %struct._ivar_t] [%struct._ivar_t { ptr @"OBJC_IVAR_$_Sub.d", ptr @IvarD, ptr @DoubleTy, i32 3, i32 8 }] }
@"_OBJC_METACLASS_RO_$_Sub" = internal global %struct._class_ro_t { i32 1, i32 40, i32 40, ptr null, ptr @SubName, ptr null, ptr null, ptr null, ptr null, ptr null }
@"_OBJC_CLASS_RO_$_Sub" = internal global %struct._class_ro_t { i32 0, i32 16, i32 24, ptr null, ptr @SubName, ptr null, ptr null, ptr @"_OBJC_IVARS_$_Sub", ptr null, ptr null }
@"OBJC_METACLASS_$_Sub" = global %struct._class_t { ptr @"OBJC_METACLASS_$_NSObject", ptr @"OBJC_METACLASS_$_Super", ptr @_objc_empty_cache, ptr null, ptr @"_OBJC_METACLASS_RO_$_Sub" }
@"OBJC_CLASS_$_Sub" = global %struct._class_t { ptr @"OBJC_METACLASS_$_Sub", ptr @"OBJC_CLASS_$_Super", ptr @_objc_empty_cache, ptr null, ptr @"_OBJC_CLASS_RO_$_Sub" }
@SubClassList = private global [1 x ptr] [ptr @"OBJC_CLASS_$_Sub"], section "__DATA,__objc_classlist,regular,no_dead_strip"
@llvm.compiler.used = appending global [1 x ptr] [ptr @SubClassList], section "llvm.metadata"
