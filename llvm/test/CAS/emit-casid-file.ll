; RUN: rm -rf %t && mkdir -p %t
; RUN: llc -O0 -cas-friendly-debug-info --filetype=obj --cas-backend --cas=%t/cas --mccas-native %s --mccas-emit-casid-file -o %t/test.o
; RUN: cat %t/test.o.casid | FileCheck %s --check-prefix=NATIVE_FILENAME
; NATIVE_FILENAME: CASID:Jllvmcas://{{.*}}
;
; RUN: rm -rf %t && mkdir -p %t
; RUN: llc -O0 -cas-friendly-debug-info --filetype=obj --cas-backend --cas=%t/cas --mccas-verify %s --mccas-emit-casid-file -o %t/test.o
; RUN: cat %t/test.o.casid | FileCheck %s --check-prefix=VERIFY_FILENAME
; VERIFY_FILENAME: CASID:Jllvmcas://{{.*}}
;
; RUN: rm -rf %t && mkdir -p %t
; RUN: llc -O0 -cas-friendly-debug-info --filetype=obj --cas-backend --cas=%t/cas --mccas-casid %s --mccas-emit-casid-file -o %t/test.o
; RUN: not cat %t/test.o.casid
;
; RUN: rm -rf %t && mkdir -p %t
; RUN: llc -O0 -cas-friendly-debug-info --filetype=obj --cas-backend --cas=%t/cas --mccas-native %s --mccas-emit-casid-file -o -
; RUN: not cat %t/test.o.casid
;
; RUN: rm -rf %t && mkdir -p %t
; RUN: llc -O0 -cas-friendly-debug-info --filetype=obj --cas-backend --cas=%t/cas --mccas-verify %s --mccas-emit-casid-file -o -
; RUN: not cat %t/test.o.casid
;
; RUN: rm -rf %t && mkdir -p %t
; RUN: llc -O0 -cas-friendly-debug-info --filetype=obj --cas-backend --cas=%t/cas --mccas-casid %s --mccas-emit-casid-file -o -
; RUN: not cat %t/test.o.casid

; REQUIRES: aarch64-registered-target

; ModuleID = '/Users/shubham/Development/test109275485/a.cpp'
source_filename = "/Users/shubham/Development/test109275485/a.cpp"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-macosx14.0.0"

; Function Attrs: mustprogress noinline nounwind optnone ssp uwtable(sync)
define noundef i32 @_Z3fooi(i32 noundef %x) #0 {
entry:
  %x.addr = alloca i32, align 4
  store i32 %x, ptr %x.addr, align 4
  %0 = load i32, ptr %x.addr, align 4
  %add = add nsw i32 %0, 2
  ret i32 %add
}

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"uwtable", i32 1}
!3 = !{i32 7, !"frame-pointer", i32 1}
!4 = !{!"clang version 18.0.0 (git@github.com:apple/llvm-project.git bd5fc55041b3dfab2de1640638ce4b5e8a016998)"}
