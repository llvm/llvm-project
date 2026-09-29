; This test verifies that global variables (objc metadata) are hashed based on their initial contents,
; allowing them to be merged even if they appear different due to their names.
; Now they become identical functions that can be merged without creating a parameter

; RUN: llc -mtriple=arm64-apple-darwin -enable-global-merge-func=true -global-merging-skip-no-params=false < %s | FileCheck %s

; CHECK: _f1.Tgm
; CHECK: _f2.Tgm

%struct._class_t = type { ptr, ptr, ptr, ptr, ptr }

@"OBJC_CLASS_$_MyClass" = external global %struct._class_t
@"OBJC_CLASSLIST_REFERENCES_$_" = internal global ptr @"OBJC_CLASS_$_MyClass", section "__DATA,__objc_classrefs,regular,no_dead_strip", align 8
@"OBJC_CLASSLIST_REFERENCES_$_.1" = internal global ptr @"OBJC_CLASS_$_MyClass", section "__DATA,__objc_classrefs,regular,no_dead_strip", align 8

@OBJC_METH_VAR_NAME_ = private unnamed_addr constant [6 x i8] c"hello\00", section "__TEXT,__objc_methname,cstring_literals", align 1
@OBJC_METH_VAR_NAME_.1 = private unnamed_addr constant [6 x i8] c"hello\00", section "__TEXT,__objc_methname,cstring_literals", align 1

@OBJC_SELECTOR_REFERENCES_ = internal externally_initialized global ptr @OBJC_METH_VAR_NAME_, section "__DATA,__objc_selrefs,literal_pointers,no_dead_strip", align 8
@OBJC_SELECTOR_REFERENCES_.1 = internal externally_initialized global ptr @OBJC_METH_VAR_NAME_.1, section "__DATA,__objc_selrefs,literal_pointers,no_dead_strip", align 8

declare ptr @objc_msgSend(ptr, ptr, ...)

define i32 @f1() {
entry:
  %0 = load ptr, ptr @"OBJC_CLASSLIST_REFERENCES_$_", align 8
  %1 = load ptr, ptr @OBJC_SELECTOR_REFERENCES_, align 8
  %call = tail call i32 @objc_msgSend(ptr noundef %0, ptr noundef %1)
  ret i32 %call
}

define i32 @f2() {
entry:
  %0 = load ptr, ptr @"OBJC_CLASSLIST_REFERENCES_$_.1", align 8
  %1 = load ptr, ptr @OBJC_SELECTOR_REFERENCES_.1, align 8
  %call = tail call i32 @objc_msgSend(ptr noundef %0, ptr noundef %1)
  ret i32 %call
}
