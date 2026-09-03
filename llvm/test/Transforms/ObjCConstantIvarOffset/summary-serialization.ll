; Test that ObjC class hierarchy info is correctly serialized into
; per-module summary bitcode (FS_OBJC_CLASS_INFO records).
;
; Hierarchy:
;   NSObject
;     `- Super
;          int x;                        // x at offset 8
;          int hidden;                   // hidden at offset 12 (extension)
;
; RUN: rm -rf %t && split-file %s %t
; RUN: opt -module-summary %t/body.ll -o %t/out.bc
; RUN: llvm-bcanalyzer --dump %t/out.bc | FileCheck %s
;
; Check that the class info record contains correct values:
; op2=instanceStart(8), op3=instanceSize(16), op4=maxIvarAlignment(4)
; CHECK: <OBJC_CLASS_INFO op0={{[0-9]+}} op1={{[0-9]+}} op2=8 op3=16 op4=4/>

;--- body.ll
source_filename = "summary.m"
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
