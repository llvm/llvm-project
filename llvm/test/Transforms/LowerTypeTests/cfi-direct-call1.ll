; RUN: opt -passes=lowertypetests -S %s | FileCheck --check-prefix=FULL %s
; RUN: opt -passes=lowertypetests -lowertypetests-summary-action=import -lowertypetests-read-summary=%p/Inputs/cfi-direct-call1.yaml -S %s | FileCheck --check-prefix=THIN %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux"

define hidden i32 @local_func1() !type !3 !type !4 {
  ret i32 11
}

define hidden i32 @local_func2() !type !3 !type !4 {
  ret i32 22
}

define hidden i32 @local_func3(i32 %i) local_unnamed_addr !type !5 !type !6 {
entry:
  %add = add nsw i32 %i, 44
  ret i32 %add
}

declare !type !3 !type !4 extern_weak i32 @extern_weak()
declare !type !3 !type !4 i32 @extern_decl()
declare i1 @llvm.type.test(ptr, metadata)

define hidden i32 @main(i32 %argc) {
entry:
  %cmp.i = icmp sgt i32 %argc, 1
  %fptr1 = select i1 %cmp.i, ptr @local_func1, ptr @local_func2
  %fptr2 = select i1 %cmp.i, ptr @extern_weak, ptr @extern_decl
  %0 = tail call i1 @llvm.type.test(ptr nonnull %fptr1, metadata !"_ZTSFivE")

  %call2 = tail call i32 %fptr1()
  %1 = tail call i1 @llvm.type.test(ptr %fptr2, metadata !"_ZTSFivE")

  %call4 = tail call i32 %fptr2()
  %call5 = tail call i32 @extern_decl()
  %call7 = tail call i32 @extern_weak()
  %call6 = tail call i32 @local_func1()
  %call8 = tail call i32 @local_func3(i32 4)
  ret i32 12
}

!3 = !{i64 0, !"_ZTSFivE"}
!4 = !{i64 0, !"_ZTSFivE.generalized"}
!5 = !{i64 0, !"_ZTSFiiE"}
!6 = !{i64 0, !"_ZTSFiiE.generalized"}

; Make sure local_func1 and local_func2 have been renamed to <name>.cfi
; FULL: define hidden i32 @local_func1.cfi()
; FULL: define hidden i32 @local_func2.cfi()

; There are no indirect calls of local_func3 type, it should not be renamed
; FULL: define hidden i32 @local_func3(i32 %i)

; Indirect references to local_func1 and local_func2 must to through jump table
; FULL: %fptr1 = select i1 %cmp.i, ptr @local_func1, ptr @local_func2

; Indirect references to extern_weak and extern_decl must go through jump table
; FULL: %0 = select i1 icmp ne (ptr @extern_weak, ptr null), ptr getelementptr inbounds ([4 x [8 x i8]], ptr @.cfi.jumptable, i64 0, i64 2), ptr null
; FULL: %fptr2 = select i1 %cmp.i, ptr %0, ptr getelementptr inbounds ([4 x [8 x i8]], ptr @.cfi.jumptable, i64 0, i64 3)

; Direct calls to extern_weak and extern_decl should go to original names
; FULL: %call5 = tail call i32 @extern_decl()
; FULL: %call7 = tail call i32 @extern_weak()

; Direct call to local_func1 should to the renamed version
; FULL: %call6 = tail call i32 @local_func1.cfi()

; local_func3 hasn't been renamed, direct call should go to the original name
; FULL: %call8 = tail call i32 @local_func3(i32 4)

; Check which jump table entries are created
; FULL: define private void @.cfi.jumptable(){{.*}}
; FULL-NEXT: entry:
; FULL-NEXT: call void asm{{.*}}local_func1.cfi{{.*}}local_func2.cfi{{.*}}extern_weak{{.*}}extern_decl

; Make sure all local functions have been renamed to <name>.cfi
; THIN: define hidden i32 @local_func1.cfi()
; THIN: define hidden i32 @local_func2.cfi()
; THIN: define hidden i32 @local_func3.cfi(i32 %i){{.*}}

; Indirect references to local_func1 and local_func2 must to through jump table
; THIN: %fptr1 = select i1 %cmp.i, ptr @local_func1, ptr @local_func2

; Indirect references to extern_weak and extern_decl must go through jump table
; THIN: %0 = select i1 icmp ne (ptr @extern_weak, ptr null), ptr @extern_weak.cfi_jt, ptr null
; THIN: %fptr2 = select i1 %cmp.i, ptr %0, ptr @extern_decl.cfi_jt

; Direct calls to extern_weak and extern_decl should go to original names
; THIN: %call5 = tail call i32 @extern_decl()
; THIN: %call7 = tail call i32 @extern_weak()

; Direct call to local_func1 should to the renamed version
; THIN: %call6 = tail call i32 @local_func1.cfi()
; THIN: %call8 = tail call i32 @local_func3.cfi(i32 4)

