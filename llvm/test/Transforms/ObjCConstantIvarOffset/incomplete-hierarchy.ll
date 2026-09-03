; Test that ivar offsets are not promoted when the superclass is external and
; is not a known stable root.
;
; Hierarchy:
;   UIViewController (external, no CLASS_RO visible)
;     `- MyController
;          int myIvar;              // myIvar at offset 0 (unresolvable)
;
; RUN: rm -rf %t && split-file %s %t
; RUN: llvm-link %t/controller.ll -S -o %t/linked.ll
; RUN: opt -passes=objc-constant-ivar-offset -S %t/linked.ll | FileCheck %s

; CHECK: @"OBJC_IVAR_$_MyController.myIvar" = {{.*}}global i64 0
; CHECK-NOT: @"OBJC_IVAR_$_MyController.myIvar" = {{.*}}constant

; Segments:
;   source.m      - ingredients
;   gen           - refresher
;   controller.ll - hand-reduced IR

;--- source.m
@interface UIViewController @end
@interface MyController : UIViewController { int myIvar; } @end
@implementation MyController @end

;--- gen
clang -target x86_64-apple-macosx10.15 -fobjc-runtime=macosx-10.15 -S -emit-llvm source.m -o -

;--- controller.ll
source_filename = "controller.m"
target triple = "x86_64-apple-macosx10.15.0"

%struct._objc_cache = type opaque
%struct._class_t = type { ptr, ptr, ptr, ptr, ptr }
%struct._class_ro_t = type { i32, i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr }
%struct._ivar_t = type { ptr, ptr, ptr, i32, i32 }

@_objc_empty_cache = external global %struct._objc_cache
@"OBJC_CLASS_$_UIViewController" = external global %struct._class_t
@"OBJC_METACLASS_$_UIViewController" = external global %struct._class_t
@ControllerName = private unnamed_addr constant [13 x i8] c"MyController\00"
@IvarMyIvar = private unnamed_addr constant [7 x i8] c"myIvar\00"
@IntTy = private unnamed_addr constant [2 x i8] c"i\00"
@"OBJC_IVAR_$_MyController.myIvar" = global i64 0, section "__DATA,__objc_ivar", align 8
@"_OBJC_IVARS_$_MyController" = internal global { i32, i32, [1 x %struct._ivar_t] } { i32 32, i32 1, [1 x %struct._ivar_t] [%struct._ivar_t { ptr @"OBJC_IVAR_$_MyController.myIvar", ptr @IvarMyIvar, ptr @IntTy, i32 2, i32 4 }] }
@"_OBJC_METACLASS_RO_$_MyController" = internal global %struct._class_ro_t { i32 1, i32 40, i32 40, ptr null, ptr @ControllerName, ptr null, ptr null, ptr null, ptr null, ptr null }
@"_OBJC_CLASS_RO_$_MyController" = internal global %struct._class_ro_t { i32 0, i32 0, i32 4, ptr null, ptr @ControllerName, ptr null, ptr null, ptr @"_OBJC_IVARS_$_MyController", ptr null, ptr null }
@"OBJC_METACLASS_$_MyController" = global %struct._class_t { ptr @"OBJC_METACLASS_$_UIViewController", ptr @"OBJC_METACLASS_$_UIViewController", ptr @_objc_empty_cache, ptr null, ptr @"_OBJC_METACLASS_RO_$_MyController" }
@"OBJC_CLASS_$_MyController" = global %struct._class_t { ptr @"OBJC_METACLASS_$_MyController", ptr @"OBJC_CLASS_$_UIViewController", ptr @_objc_empty_cache, ptr null, ptr @"_OBJC_CLASS_RO_$_MyController" }
@ControllerClassList = private global [1 x ptr] [ptr @"OBJC_CLASS_$_MyController"], section "__DATA,__objc_classlist,regular,no_dead_strip"
@llvm.compiler.used = appending global [1 x ptr] [ptr @ControllerClassList], section "llvm.metadata"
