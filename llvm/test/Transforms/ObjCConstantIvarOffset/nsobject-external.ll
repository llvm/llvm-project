; Test that external NSObject is recognized as an ABI-stable root after
; cross-module implementations are linked by FullLTO.
;
; Hierarchy:
;   NSObject (external, ABI-stable root)
;     `- Super
;          int x;                   // x at offset 8
;          int hidden;              // hidden at offset 12
;          `- Sub
;               int y;              // y compiled at 12, slid to 16
;
; RUN: rm -rf %t && split-file %s %t
; RUN: llvm-link %t/super.ll %t/sub.ll -S -o %t/linked.ll
; RUN: opt -passes=objc-constant-ivar-offset -S %t/linked.ll | FileCheck %s

; CHECK: @"OBJC_IVAR_$_Super.x" = {{.*}}constant i64 8
; CHECK: @"OBJC_IVAR_$_Super.hidden" = {{.*}}constant i64 12
; CHECK: @"OBJC_IVAR_$_Sub.y" = {{.*}}constant i64 16
; CHECK: @"_OBJC_CLASS_RO_$_Sub" = internal global %struct._class_ro_t { i32 0, i32 16, i32 20,

; Segments:
;   super.m / sub.m  - ingredients (requires Darwin SDK to regenerate)
;   gen              - refresher
;   super.ll / sub.ll - hand-reduced cross-module IR

;--- super.m
#import <objc/NSObject.h>
@interface Super : NSObject { int x; } @end
@interface Super () { int hidden; } @end
@implementation Super @end

;--- sub.m
#import <objc/NSObject.h>
@interface Super : NSObject { int x; } @end
@interface Sub : Super { int y; } @end
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
@IvarX = private unnamed_addr constant [2 x i8] c"x\00"
@IvarHidden = private unnamed_addr constant [7 x i8] c"hidden\00"
@IntTy = private unnamed_addr constant [2 x i8] c"i\00"
@"OBJC_IVAR_$_Super.x" = global i64 8, section "__DATA,__objc_ivar", align 8
@"OBJC_IVAR_$_Super.hidden" = global i64 12, section "__DATA,__objc_ivar", align 8
@"_OBJC_IVARS_$_Super" = internal global { i32, i32, [2 x %struct._ivar_t] } { i32 32, i32 2, [2 x %struct._ivar_t] [%struct._ivar_t { ptr @"OBJC_IVAR_$_Super.x", ptr @IvarX, ptr @IntTy, i32 2, i32 4 }, %struct._ivar_t { ptr @"OBJC_IVAR_$_Super.hidden", ptr @IvarHidden, ptr @IntTy, i32 2, i32 4 }] }
@"_OBJC_METACLASS_RO_$_Super" = internal global %struct._class_ro_t { i32 1, i32 40, i32 40, ptr null, ptr @SuperName, ptr null, ptr null, ptr null, ptr null, ptr null }
@"_OBJC_CLASS_RO_$_Super" = internal global %struct._class_ro_t { i32 0, i32 8, i32 16, ptr null, ptr @SuperName, ptr null, ptr null, ptr @"_OBJC_IVARS_$_Super", ptr null, ptr null }
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
@IvarY = private unnamed_addr constant [2 x i8] c"y\00"
@IntTy = private unnamed_addr constant [2 x i8] c"i\00"
@"OBJC_IVAR_$_Sub.y" = global i64 12, section "__DATA,__objc_ivar", align 8
@"_OBJC_IVARS_$_Sub" = internal global { i32, i32, [1 x %struct._ivar_t] } { i32 32, i32 1, [1 x %struct._ivar_t] [%struct._ivar_t { ptr @"OBJC_IVAR_$_Sub.y", ptr @IvarY, ptr @IntTy, i32 2, i32 4 }] }
@"_OBJC_METACLASS_RO_$_Sub" = internal global %struct._class_ro_t { i32 1, i32 40, i32 40, ptr null, ptr @SubName, ptr null, ptr null, ptr null, ptr null, ptr null }
@"_OBJC_CLASS_RO_$_Sub" = internal global %struct._class_ro_t { i32 0, i32 12, i32 16, ptr null, ptr @SubName, ptr null, ptr null, ptr @"_OBJC_IVARS_$_Sub", ptr null, ptr null }
@"OBJC_METACLASS_$_Sub" = global %struct._class_t { ptr @"OBJC_METACLASS_$_NSObject", ptr @"OBJC_METACLASS_$_Super", ptr @_objc_empty_cache, ptr null, ptr @"_OBJC_METACLASS_RO_$_Sub" }
@"OBJC_CLASS_$_Sub" = global %struct._class_t { ptr @"OBJC_METACLASS_$_Sub", ptr @"OBJC_CLASS_$_Super", ptr @_objc_empty_cache, ptr null, ptr @"_OBJC_CLASS_RO_$_Sub" }
@SubClassList = private global [1 x ptr] [ptr @"OBJC_CLASS_$_Sub"], section "__DATA,__objc_classlist,regular,no_dead_strip"
@llvm.compiler.used = appending global [1 x ptr] [ptr @SubClassList], section "llvm.metadata"
