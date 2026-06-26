; Test that ivar offsets are NOT promoted when the superclass is
; external and non-stable (incomplete hierarchy).
;
; Hierarchy:
;   UIViewController (external, no CLASS_RO visible)
;     `- MyController
;          int myIvar;                   // myIvar at offset 0 (unresolvable)
;
; --- FullLTO path ---
; RUN: rm -rf %t && split-file %s %t
; RUN: opt -passes=objc-constant-ivar-offset -S %t/body.ll | FileCheck %s --check-prefix=CONTROLLER
;
; CONTROLLER: @"OBJC_IVAR_$_MyController.myIvar" = {{.*}}global i64 0
; CONTROLLER-NOT: @"OBJC_IVAR_$_MyController.myIvar" = {{.*}}constant
;
; --- ThinLTO path ---
; RUN: opt -module-summary %t/body.ll -o %t/body.bc
; RUN: llvm-lto2 run %t/body.bc -save-temps -o %t/out \
; RUN:   -r=%t/body.bc,_OBJC_CLASS_$_MyController,plx \
; RUN:   -r=%t/body.bc,_OBJC_CLASS_$_UIViewController, \
; RUN:   -r=%t/body.bc,_OBJC_METACLASS_$_MyController,plx \
; RUN:   -r=%t/body.bc,_OBJC_METACLASS_$_UIViewController, \
; RUN:   -r=%t/body.bc,_OBJC_IVAR_$_MyController.myIvar,plx \
; RUN:   -r=%t/body.bc,__objc_empty_cache,
; RUN: llvm-dis %t/out.1.4.opt.bc -o - | FileCheck %s --check-prefix=CONTROLLER

; Segments:
;   source.m - ingredients (requires Darwin SDK to regenerate)
;   gen      - refresher
;   body.ll  - single module used by both FullLTO and ThinLTO paths

;--- source.m
@interface UIViewController @end
@interface MyController : UIViewController { int myIvar; } @end
@implementation MyController @end

;--- gen
clang -target x86_64-apple-macosx10.15 -fobjc-runtime=macosx-10.15 -S -emit-llvm source.m -o -
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
@"OBJC_METACLASS_$_UIViewController" = external global %struct._class_t
@OBJC_CLASS_NAME_ = private unnamed_addr constant [13 x i8] c"MyController\00", section "__TEXT,__objc_classname,cstring_literals", align 1
@"_OBJC_METACLASS_RO_$_MyController" = internal global %struct._class_ro_t { i32 1, i32 40, i32 40, ptr null, ptr @OBJC_CLASS_NAME_, ptr null, ptr null, ptr null, ptr null, ptr null }, section "__DATA, __objc_const", align 8
@"OBJC_METACLASS_$_MyController" = global %struct._class_t { ptr @"OBJC_METACLASS_$_UIViewController", ptr @"OBJC_METACLASS_$_UIViewController", ptr @_objc_empty_cache, ptr null, ptr @"_OBJC_METACLASS_RO_$_MyController" }, section "__DATA, __objc_data", align 8
@"OBJC_CLASS_$_UIViewController" = external global %struct._class_t
@"OBJC_IVAR_$_MyController.myIvar" = global i64 0, section "__DATA, __objc_ivar", align 8
@OBJC_METH_VAR_NAME_ = private unnamed_addr constant [7 x i8] c"myIvar\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@OBJC_METH_VAR_TYPE_ = private unnamed_addr constant [2 x i8] c"i\00", section "__TEXT,__objc_methtype,cstring_literals", align 1
@"_OBJC_$_INSTANCE_VARIABLES_MyController" = internal global { i32, i32, [1 x %struct._ivar_t] } { i32 32, i32 1, [1 x %struct._ivar_t] [%struct._ivar_t { ptr @"OBJC_IVAR_$_MyController.myIvar", ptr @OBJC_METH_VAR_NAME_, ptr @OBJC_METH_VAR_TYPE_, i32 2, i32 4 }] }, section "__DATA, __objc_const", align 8
@"_OBJC_CLASS_RO_$_MyController" = internal global %struct._class_ro_t { i32 0, i32 0, i32 4, ptr null, ptr @OBJC_CLASS_NAME_, ptr null, ptr null, ptr @"_OBJC_$_INSTANCE_VARIABLES_MyController", ptr null, ptr null }, section "__DATA, __objc_const", align 8
@"OBJC_CLASS_$_MyController" = global %struct._class_t { ptr @"OBJC_METACLASS_$_MyController", ptr @"OBJC_CLASS_$_UIViewController", ptr @_objc_empty_cache, ptr null, ptr @"_OBJC_CLASS_RO_$_MyController" }, section "__DATA, __objc_data", align 8
@"OBJC_LABEL_CLASS_$" = private global [1 x ptr] [ptr @"OBJC_CLASS_$_MyController"], section "__DATA,__objc_classlist,regular,no_dead_strip", align 8
@llvm.compiler.used = appending global [5 x ptr] [ptr @OBJC_CLASS_NAME_, ptr @OBJC_METH_VAR_NAME_, ptr @OBJC_METH_VAR_TYPE_, ptr @"_OBJC_$_INSTANCE_VARIABLES_MyController", ptr @"OBJC_LABEL_CLASS_$"], section "llvm.metadata"

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}

!0 = !{i32 1, !"Objective-C Version", i32 2}
!1 = !{i32 1, !"Objective-C Image Info Version", i32 0}
!2 = !{i32 1, !"Objective-C Image Info Section", !"__DATA,__objc_imageinfo,regular,no_dead_strip"}
!3 = !{i32 1, !"Objective-C Garbage Collection", i8 0}
!4 = !{i32 1, !"Objective-C Class Properties", i32 64}
!5 = !{i32 1, !"Objective-C Enforce ClassRO Pointer Signing", i8 0}
!6 = !{i32 8, !"PIC Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"frame-pointer", i32 2}
