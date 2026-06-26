; Test that two independent hierarchy trees (custom root + NSObject-based)
; resolve without conflict.
;
; Hierarchy:
;   Root (no superclass)
;     int a;                            // a at offset 8
;     `- RootSub
;          int b;                        // b at offset 12
;
;   NSObject (external, ABI-stable root)
;     `- ObjSub
;          int c;                        // c at offset 8
;
; --- FullLTO path ---
; RUN: rm -rf %t && split-file %s %t
; RUN: opt -passes=objc-constant-ivar-offset -S %t/body.ll | FileCheck %s --check-prefixes=ROOT,ROOTSUB,OBJSUB
;
; ROOT: @"OBJC_IVAR_$_Root.a" = {{.*}}constant i64 8
; ROOTSUB: @"OBJC_IVAR_$_RootSub.b" = {{.*}}constant i64 12
; OBJSUB: @"OBJC_IVAR_$_ObjSub.c" = {{.*}}constant i64 8
;
; --- ThinLTO path ---
; RUN: opt -module-summary %t/body.ll -o %t/body.bc
; RUN: llvm-lto2 run %t/body.bc -save-temps -o %t/out \
; RUN:   -r=%t/body.bc,_OBJC_CLASS_$_Root,plx \
; RUN:   -r=%t/body.bc,_OBJC_CLASS_$_RootSub,plx \
; RUN:   -r=%t/body.bc,_OBJC_CLASS_$_ObjSub,plx \
; RUN:   -r=%t/body.bc,_OBJC_CLASS_$_NSObject, \
; RUN:   -r=%t/body.bc,_OBJC_METACLASS_$_Root,plx \
; RUN:   -r=%t/body.bc,_OBJC_METACLASS_$_RootSub,plx \
; RUN:   -r=%t/body.bc,_OBJC_METACLASS_$_ObjSub,plx \
; RUN:   -r=%t/body.bc,_OBJC_METACLASS_$_NSObject, \
; RUN:   -r=%t/body.bc,_OBJC_IVAR_$_Root.a,plx \
; RUN:   -r=%t/body.bc,_OBJC_IVAR_$_RootSub.b,plx \
; RUN:   -r=%t/body.bc,_OBJC_IVAR_$_ObjSub.c,plx \
; RUN:   -r=%t/body.bc,__objc_empty_cache,
; RUN: llvm-dis %t/out.1.4.opt.bc -o - | FileCheck %s --check-prefixes=ROOT,ROOTSUB,OBJSUB

; Segments:
;   source.m - ingredients (requires Darwin SDK to regenerate)
;   gen      - refresher
;   body.ll  - single module used by both FullLTO and ThinLTO paths

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
# body.ll is auto-generated from source.m
;--- body.ll
; ModuleID = 'source.m'
source_filename = "source.m"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

%struct._objc_cache = type opaque
%struct._class_t = type { ptr, ptr, ptr, ptr, ptr }
%struct._class_ro_t = type { i32, i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr }
%struct._ivar_t = type { ptr, ptr, ptr, i32, i32 }

@_objc_empty_cache = external global %struct._objc_cache

; --- Root tree (custom root, no superclass) ---
@OBJC_CLASS_NAME_ = private unnamed_addr constant [5 x i8] c"Root\00", section "__TEXT,__objc_classname,cstring_literals", align 1
@"_OBJC_METACLASS_RO_$_Root" = internal global %struct._class_ro_t { i32 1, i32 40, i32 40, ptr null, ptr @OBJC_CLASS_NAME_, ptr null, ptr null, ptr null, ptr null, ptr null }, section "__DATA, __objc_const", align 8
@"OBJC_METACLASS_$_Root" = global %struct._class_t { ptr @"OBJC_METACLASS_$_Root", ptr @"OBJC_METACLASS_$_Root", ptr @_objc_empty_cache, ptr null, ptr @"_OBJC_METACLASS_RO_$_Root" }, section "__DATA, __objc_data", align 8
@"OBJC_IVAR_$_Root.a" = constant i64 8, section "__DATA, __objc_ivar", align 8
@OBJC_METH_VAR_NAME_ = private unnamed_addr constant [2 x i8] c"a\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@OBJC_METH_VAR_TYPE_ = private unnamed_addr constant [2 x i8] c"i\00", section "__TEXT,__objc_methtype,cstring_literals", align 1
@"_OBJC_$_INSTANCE_VARIABLES_Root" = internal global { i32, i32, [1 x %struct._ivar_t] } { i32 32, i32 1, [1 x %struct._ivar_t] [%struct._ivar_t { ptr @"OBJC_IVAR_$_Root.a", ptr @OBJC_METH_VAR_NAME_, ptr @OBJC_METH_VAR_TYPE_, i32 2, i32 4 }] }, section "__DATA, __objc_const", align 8
@"_OBJC_CLASS_RO_$_Root" = internal global %struct._class_ro_t { i32 0, i32 8, i32 12, ptr null, ptr @OBJC_CLASS_NAME_, ptr null, ptr null, ptr @"_OBJC_$_INSTANCE_VARIABLES_Root", ptr null, ptr null }, section "__DATA, __objc_const", align 8
@"OBJC_CLASS_$_Root" = global %struct._class_t { ptr @"OBJC_METACLASS_$_Root", ptr null, ptr @_objc_empty_cache, ptr null, ptr @"_OBJC_CLASS_RO_$_Root" }, section "__DATA, __objc_data", align 8

; --- RootSub (child of custom root) ---
@OBJC_CLASS_NAME_.1 = private unnamed_addr constant [8 x i8] c"RootSub\00", section "__TEXT,__objc_classname,cstring_literals", align 1
@"_OBJC_METACLASS_RO_$_RootSub" = internal global %struct._class_ro_t { i32 1, i32 40, i32 40, ptr null, ptr @OBJC_CLASS_NAME_.1, ptr null, ptr null, ptr null, ptr null, ptr null }, section "__DATA, __objc_const", align 8
@"OBJC_METACLASS_$_RootSub" = global %struct._class_t { ptr @"OBJC_METACLASS_$_Root", ptr @"OBJC_METACLASS_$_Root", ptr @_objc_empty_cache, ptr null, ptr @"_OBJC_METACLASS_RO_$_RootSub" }, section "__DATA, __objc_data", align 8
@"OBJC_IVAR_$_RootSub.b" = constant i64 12, section "__DATA, __objc_ivar", align 8
@OBJC_METH_VAR_NAME_.2 = private unnamed_addr constant [2 x i8] c"b\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@"_OBJC_$_INSTANCE_VARIABLES_RootSub" = internal global { i32, i32, [1 x %struct._ivar_t] } { i32 32, i32 1, [1 x %struct._ivar_t] [%struct._ivar_t { ptr @"OBJC_IVAR_$_RootSub.b", ptr @OBJC_METH_VAR_NAME_.2, ptr @OBJC_METH_VAR_TYPE_, i32 2, i32 4 }] }, section "__DATA, __objc_const", align 8
@"_OBJC_CLASS_RO_$_RootSub" = internal global %struct._class_ro_t { i32 0, i32 12, i32 16, ptr null, ptr @OBJC_CLASS_NAME_.1, ptr null, ptr null, ptr @"_OBJC_$_INSTANCE_VARIABLES_RootSub", ptr null, ptr null }, section "__DATA, __objc_const", align 8
@"OBJC_CLASS_$_RootSub" = global %struct._class_t { ptr @"OBJC_METACLASS_$_RootSub", ptr @"OBJC_CLASS_$_Root", ptr @_objc_empty_cache, ptr null, ptr @"_OBJC_CLASS_RO_$_RootSub" }, section "__DATA, __objc_data", align 8

; --- ObjSub (child of NSObject, the external ABI-stable root) ---
@"OBJC_METACLASS_$_NSObject" = external global %struct._class_t
@"OBJC_CLASS_$_NSObject" = external global %struct._class_t
@OBJC_CLASS_NAME_.3 = private unnamed_addr constant [7 x i8] c"ObjSub\00", section "__TEXT,__objc_classname,cstring_literals", align 1
@"_OBJC_METACLASS_RO_$_ObjSub" = internal global %struct._class_ro_t { i32 1, i32 40, i32 40, ptr null, ptr @OBJC_CLASS_NAME_.3, ptr null, ptr null, ptr null, ptr null, ptr null }, section "__DATA, __objc_const", align 8
@"OBJC_METACLASS_$_ObjSub" = global %struct._class_t { ptr @"OBJC_METACLASS_$_NSObject", ptr @"OBJC_METACLASS_$_NSObject", ptr @_objc_empty_cache, ptr null, ptr @"_OBJC_METACLASS_RO_$_ObjSub" }, section "__DATA, __objc_data", align 8
@"OBJC_IVAR_$_ObjSub.c" = constant i64 8, section "__DATA, __objc_ivar", align 8
@OBJC_METH_VAR_NAME_.4 = private unnamed_addr constant [2 x i8] c"c\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@"_OBJC_$_INSTANCE_VARIABLES_ObjSub" = internal global { i32, i32, [1 x %struct._ivar_t] } { i32 32, i32 1, [1 x %struct._ivar_t] [%struct._ivar_t { ptr @"OBJC_IVAR_$_ObjSub.c", ptr @OBJC_METH_VAR_NAME_.4, ptr @OBJC_METH_VAR_TYPE_, i32 2, i32 4 }] }, section "__DATA, __objc_const", align 8
@"_OBJC_CLASS_RO_$_ObjSub" = internal global %struct._class_ro_t { i32 0, i32 8, i32 12, ptr null, ptr @OBJC_CLASS_NAME_.3, ptr null, ptr null, ptr @"_OBJC_$_INSTANCE_VARIABLES_ObjSub", ptr null, ptr null }, section "__DATA, __objc_const", align 8
@"OBJC_CLASS_$_ObjSub" = global %struct._class_t { ptr @"OBJC_METACLASS_$_ObjSub", ptr @"OBJC_CLASS_$_NSObject", ptr @_objc_empty_cache, ptr null, ptr @"_OBJC_CLASS_RO_$_ObjSub" }, section "__DATA, __objc_data", align 8

; --- Class list and compiler.used ---
@"OBJC_LABEL_CLASS_$" = private global [3 x ptr] [ptr @"OBJC_CLASS_$_Root", ptr @"OBJC_CLASS_$_RootSub", ptr @"OBJC_CLASS_$_ObjSub"], section "__DATA,__objc_classlist,regular,no_dead_strip", align 8
@llvm.compiler.used = appending global [11 x ptr] [ptr @OBJC_CLASS_NAME_, ptr @OBJC_METH_VAR_NAME_, ptr @OBJC_METH_VAR_TYPE_, ptr @"_OBJC_$_INSTANCE_VARIABLES_Root", ptr @OBJC_CLASS_NAME_.1, ptr @OBJC_METH_VAR_NAME_.2, ptr @"_OBJC_$_INSTANCE_VARIABLES_RootSub", ptr @OBJC_CLASS_NAME_.3, ptr @OBJC_METH_VAR_NAME_.4, ptr @"_OBJC_$_INSTANCE_VARIABLES_ObjSub", ptr @"OBJC_LABEL_CLASS_$"], section "llvm.metadata"

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 2]}
!1 = !{i32 1, !"Objective-C Version", i32 2}
!2 = !{i32 1, !"Objective-C Image Info Version", i32 0}
!3 = !{i32 1, !"Objective-C Image Info Section", !"__DATA,__objc_imageinfo,regular,no_dead_strip"}
!4 = !{i32 1, !"Objective-C Garbage Collection", i8 0}
!5 = !{i32 1, !"Objective-C Class Properties", i32 64}
!6 = !{i32 1, !"Objective-C Enforce ClassRO Pointer Signing", i8 0}
!7 = !{i32 8, !"PIC Level", i32 2}
!8 = !{i32 7, !"uwtable", i32 2}
!9 = !{i32 7, !"frame-pointer", i32 2}
