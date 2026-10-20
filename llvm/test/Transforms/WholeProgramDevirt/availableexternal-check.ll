; RUN: opt -S -passes=wholeprogramdevirt -whole-program-visibility %s | FileCheck %s

; This test is reduced from C++ code like this:
; class A :public std::exception {
; public:
;   A() {};
;   const char* what () const throw () {return "A";}
; };
; long test(std::exception *p) {
;   const char* ch = p->what();
;   ...;
; }
;
; Build command is "clang++ -O2 -target x86_64-unknown-linux -flto=full \
; -fwhole-program-vtables -static-libstdc++  -Wl,-plugin-opt=-whole-program-visibility"
;
; _ZTVSt9exception's visibility is 1 (Linkage Unit), and available_externally.
; If any GV is available_externally, icall.branch.funnel should not be generated.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux"

@_ZTVSt9exception = available_externally constant { [5 x ptr] } { [5 x ptr] [ptr null, ptr null, ptr null, ptr null, ptr @_ZNKSt9exception4whatEv] }, !type !0, !type !1
@_ZTV1A.0 = constant [5 x ptr] [ptr null, ptr null, ptr null, ptr null, ptr @_ZNK1A4whatEv], !type !3, !type !4, !type !5, !type !6

declare ptr @_ZNKSt9exception4whatEv()

define ptr @_Z4testPSt9exception() {
  %1 = load ptr, ptr null, align 8
  %2 = call i1 @llvm.type.test(ptr %1, metadata !"_ZTSSt9exception")
  tail call void @llvm.assume(i1 %2)
  %3 = getelementptr i8, ptr %1, i64 16
  %4 = load ptr, ptr %3, align 8
  %5 = tail call ptr %4(ptr null)
  ret ptr %5
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #0

declare ptr @_ZNK1A4whatEv()

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i1 @llvm.type.test(ptr, metadata) #1

; CHECK-NOT: call void (...) @llvm.icall.branch.funnel

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!0 = !{i64 16, !"_ZTSSt9exception"}
!1 = !{i64 32, !"_ZTSMSt9exceptionKDoFPKcvE.virtual"}
!3 = !{i32 16, !"_ZTS1A"}
!4 = !{i32 32, !"_ZTSM1AKDoFPKcvE.virtual"}
!5 = !{i32 16, !"_ZTSSt9exception"}
!6 = !{i32 32, !"_ZTSMSt9exceptionKDoFPKcvE.virtual"}
