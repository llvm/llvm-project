; Test to ensure that we correctly handle a type test not used for a virtual call.
; If it isn't removed correctly by WPD then we could incorrectly get an Unsat
; (resulting in an unreachable in the IR).

; REQUIRES: x86-registered-target

; RUN: opt -thinlto-bc -o %t.o %s

; RUN: llvm-lto2 run %t.o -save-temps -pass-remarks=. \
; RUN:   -whole-program-visibility \
; RUN:	 -r=%t.o,_ZTVN12_GLOBAL__N_18RealFileE,px \
; RUN:   -o %t2
; RUN: llvm-dis %t2.1.4.opt.bc -o - | FileCheck %s --check-prefix=CHECK-IR

; Try again without LTO unit splitting.
; RUN: opt -thinlto-bc -thinlto-split-lto-unit=false -o %t3.o %s
; RUN: llvm-lto2 run %t.o -save-temps -pass-remarks=. \
; RUN:   -whole-program-visibility \
; RUN:	 -r=%t.o,_ZTVN12_GLOBAL__N_18RealFileE,px \
; RUN:   -o %t4
; RUN: llvm-dis %t4.1.4.opt.bc -o - | FileCheck %s --check-prefix=CHECK-IR

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

%"class.llvm::vfs::File" = type { ptr }
%"class.llvm::vfs::Status" = type <{ %"class.std::__cxx11::basic_string", %"class.llvm::sys::fs::UniqueID", %"struct.std::chrono::time_point", i32, i32, i64, i32, i32, i8, [7 x i8] }>
%"class.std::__cxx11::basic_string" = type { %"struct.std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider", i64, %union.anon }
%"struct.std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider" = type { ptr }
%union.anon = type { i64, [8 x i8] }
%"class.llvm::sys::fs::UniqueID" = type { i64, i64 }
%"struct.std::chrono::time_point" = type { %"struct.std::chrono::duration" }
%"struct.std::chrono::duration" = type { i64 }
%"class.(anonymous namespace)::RealFile" = type { %"class.llvm::vfs::File", i32, [4 x i8], %"class.llvm::vfs::Status", %"class.std::__cxx11::basic_string" }

@_ZTVN12_GLOBAL__N_18RealFileE = unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr null, ptr @_ZN12_GLOBAL__N_18RealFileD2Ev] }, align 8, !type !74

define internal void @_ZN12_GLOBAL__N_18RealFileD2Ev(ptr %this) unnamed_addr #0 align 2 {
entry:
; CHECK-IR: store
  store ptr getelementptr inbounds ({ [3 x ptr] }, ptr @_ZTVN12_GLOBAL__N_18RealFileE, i64 0, inrange i32 0, i64 2), ptr %this, align 8
  %0 = tail call i1 @llvm.type.test(ptr getelementptr inbounds ({ [3 x ptr] }, ptr @_ZTVN12_GLOBAL__N_18RealFileE, i64 0, inrange i32 0, i64 2), metadata !"4$09c6cc733fc6accb91e5d7b87cb48f2d")
  tail call void @llvm.assume(i1 %0)
; CHECK-IR-NEXT: ret void
  ret void
}

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1)

; Make sure we don't inline or otherwise optimize out the direct calls.
attributes #0 = { noinline optnone }

!74 = !{i64 16, !"4$09c6cc733fc6accb91e5d7b87cb48f2d"}
