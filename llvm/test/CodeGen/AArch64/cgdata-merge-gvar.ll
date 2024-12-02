; This test verifies that global variables are hashed based on their initial contents,
; allowing them to be merged even if they appear different due to their names.
; Now they become identical functions that can be merged without creating a paramter.

; RUN: rm -rf %t && split-file %s %t

; RUN: llc -mtriple=arm64-apple-darwin -enable-global-merge-func=true -global-merging-skip-no-params=false < %t/string.ll | FileCheck %s
; RUN: llc -mtriple=arm64-apple-darwin -enable-global-merge-func=true -global-merging-skip-no-params=false < %t/ns-const.ll | FileCheck %s
; RUN: llc -mtriple=arm64-apple-darwin -enable-global-merge-func=true -global-merging-skip-no-params=false < %t/objc-ref.ll | FileCheck %s

; CHECK: _f1.Tgm
; CHECK: _f2.Tgm

;--- string.ll

@.str = private unnamed_addr constant [6 x i8] c"hello\00", align 1
@.str.1 = private unnamed_addr constant [6 x i8] c"hello\00", align 1

declare noundef i32 @goo(ptr noundef)

define i32 @f1() {
entry:
  %call = tail call noundef i32 @goo(ptr noundef nonnull @.str)
  %add = add nsw i32 %call, 1
  ret i32 %add
}

define i32 @f2() {
entry:
  %call = tail call noundef i32 @goo(ptr noundef nonnull @.str.1)
  %add = add nsw i32 %call, 1
  ret i32 %add
}

;--- ns-const.ll

%struct.__NSConstantString_tag = type { ptr, i32, ptr, i64 }
@__CFConstantStringClassReference = external global [0 x i32]
@.str.2 = private unnamed_addr constant [9 x i8] c"cfstring\00", section "__TEXT,__cstring,cstring_literals", align 1
@_unnamed_cfstring_ = private global %struct.__NSConstantString_tag { ptr @__CFConstantStringClassReference, i32 1992, ptr @.str.2, i64 8 }, section "__DATA,__cfstring", align 8

@.str.3 = private unnamed_addr constant [9 x i8] c"cfstring\00", section "__TEXT,__cstring,cstring_literals", align 1
@_unnamed_cfstring_.2 = private global %struct.__NSConstantString_tag { ptr @__CFConstantStringClassReference, i32 1992, ptr @.str.3, i64 8 }, section "__DATA,__cfstring", align 8

declare noundef i32 @hoo(ptr noundef)

define i32 @f1() {
entry:
  %call = tail call i32 @hoo(ptr noundef nonnull @_unnamed_cfstring_)
  %add = sub nsw i32 %call, 1
  ret i32 %add
}

define i32 @f2() {
entry:
  %call = tail call i32 @hoo(ptr noundef nonnull @_unnamed_cfstring_.2)
  %add = sub nsw i32 %call, 1
  ret i32 %add
}

;--- objc-ref.ll

%struct._class_t = type { ptr, ptr, ptr, ptr, ptr }

@"OBJC_CLASS_$_MyClass" = external global %struct._class_t
@"OBJC_CLASSLIST_REFERENCES_$_" = internal global ptr @"OBJC_CLASS_$_MyClass", section "__DATA,__objc_classrefs,regular,no_dead_strip", align 8
@"OBJC_CLASSLIST_REFERENCES_$_.1" = internal global ptr @"OBJC_CLASS_$_MyClass", section "__DATA,__objc_classrefs,regular,no_dead_strip", align 8

@OBJC_METH_VAR_NAME_ = private unnamed_addr constant [6 x i8] c"hello\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@OBJC_METH_VAR_NAME_.1 = private unnamed_addr constant [6 x i8] c"hello\00", section "__TEXT,__objc_methname,cstring_literals", align 1

@OBJC_SELECTOR_REFERENCES_ = internal externally_initialized global ptr @OBJC_METH_VAR_NAME_, section "__DATA,__objc_selrefs,literal_pointers,no_dead_strip", align 8
@OBJC_SELECTOR_REFERENCES_.1 = internal externally_initialized global ptr @OBJC_METH_VAR_NAME_.1, section "__DATA,__objc_selrefs,literal_pointers,no_dead_strip", align 8

define i32 @f1() {
entry:
  %0 = load ptr, ptr @"OBJC_CLASSLIST_REFERENCES_$_", align 8
  %1 = load ptr, ptr @OBJC_SELECTOR_REFERENCES_, align 8
  %call = tail call noundef i32 @objc_msgSend(ptr noundef %0, ptr noundef %1)
  ret i32 %call
}

declare ptr @objc_msgSend(ptr, ptr, ...)

define i32 @f2() {
entry:
  %0 = load ptr, ptr @"OBJC_CLASSLIST_REFERENCES_$_.1", align 8
  %1 = load ptr, ptr @OBJC_SELECTOR_REFERENCES_.1, align 8
  %call = tail call noundef i32 @objc_msgSend(ptr noundef %0, ptr noundef %1)
  ret i32 %call
}
