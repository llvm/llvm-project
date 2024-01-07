; RUN: llc -verify-machineinstrs < %s | FileCheck %s --check-prefix=CHECK

; CHECK: "?mi_new_test@@YAPEAX_K@Z":
; CHECK-NEXT: # %bb.0:
; CHECK-NEXT: jmp     mi_new

; CHECK: "?builtin_malloc_test@@YAPEAX_K@Z":
; CHECK-NEXT: # %bb.0:
; CHECK-NEXT: jmp     malloc

; // Built with: clang-cl.exe /c patchable-prologue-tailcall.cpp /O2 /hotpatch -Xclang -emit-llvm
;
; typedef unsigned long long size_t;
; 
; extern "C" {
;     void* mi_new(size_t size);
;  }
;
; void *mi_new_test(size_t count)
; {
;     return mi_new(count);
; }
; 
; void *builtin_malloc_test(size_t count)
; {
;     return __builtin_malloc(count);
; }

; ModuleID = 'patchable-prologue-tailcall.cpp'
source_filename = "patchable-prologue-tailcall.cpp"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.38.33133"

; Function Attrs: mustprogress nounwind sspstrong uwtable
define dso_local noundef ptr @"?mi_new_test@@YAPEAX_K@Z"(i64 noundef %count) local_unnamed_addr #0 {
entry:
  %call = tail call ptr @mi_new(i64 noundef %count) #4
  ret ptr %call
}

declare dso_local ptr @mi_new(i64 noundef) local_unnamed_addr #1

; Function Attrs: mustprogress nofree nounwind sspstrong willreturn memory(inaccessiblemem: readwrite) uwtable
define dso_local noalias noundef ptr @"?builtin_malloc_test@@YAPEAX_K@Z"(i64 noundef %count) local_unnamed_addr #2 {
entry:
  %call = tail call ptr @malloc(i64 noundef %count) #4
  ret ptr %call
}

; Function Attrs: mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite)
declare dso_local noalias noundef ptr @malloc(i64 noundef) local_unnamed_addr #3

attributes #0 = { mustprogress nounwind sspstrong uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "patchable-function"="prologue-short-redirect" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { mustprogress nofree nounwind sspstrong willreturn memory(inaccessiblemem: readwrite) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "patchable-function"="prologue-short-redirect" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite) "alloc-family"="malloc" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #4 = { nounwind }

!llvm.linker.options = !{!0, !1}
!llvm.module.flags = !{!2, !3, !4, !5}
!llvm.ident = !{!6}

!0 = !{!"/DEFAULTLIB:libcmt.lib"}
!1 = !{!"/DEFAULTLIB:oldnames.lib"}
!2 = !{i32 1, !"wchar_size", i32 2}
!3 = !{i32 8, !"PIC Level", i32 2}
!4 = !{i32 7, !"uwtable", i32 2}
!5 = !{i32 1, !"MaxTLSAlign", i32 65536}
!6 = !{!"clang version 18.0.0git (https://github.com/llvm/llvm-project.git 0ccf9e29e9626a3a6813b35608c8959178c50a6f)"}
