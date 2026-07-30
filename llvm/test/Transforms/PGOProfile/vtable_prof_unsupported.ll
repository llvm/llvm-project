; RUN: opt < %s -passes=pgo-instr-gen -enable-vtable-value-profiling -S 2>&1 | FileCheck %s

; Test that unsupported warning is emitted for non-ELF object files.
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-macosx14.0.0"

; CHECK: warning: {{.*}} VTable value profiling is presently not supported for non-ELF object formats

@_ZTV4Base = constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr null, ptr @_ZN4Base4funcEi] }, !type !0, !type !1
@_ZTV7Derived = constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr null, ptr @_ZN7Derived4funcEi] }, !type !0, !type !1, !type !2, !type !3

@llvm.compiler.used = appending global [2 x ptr] [ptr @_ZTV4Base, ptr @_ZTV7Derived], section "llvm.metadata"

define i32 @_Z4funci(i32 %a) {
entry:
  %call = call ptr @_Z10createTypev()
  %vtable = load ptr, ptr %call
  %0 = call i1 @llvm.public.type.test(ptr %vtable, metadata !"_ZTS7Derived")
  call void @llvm.assume(i1 %0)
  %1 = load ptr, ptr %vtable
  %call1 = call i32 %1(ptr  %call, i32 %a)
  ret i32 %call1
}

declare ptr @_Z10createTypev() 
declare i1 @llvm.public.type.test(ptr, metadata)
declare void @llvm.assume(i1)
declare i32 @_ZN4Base4funcEi(ptr, i32)
declare i32 @_ZN7Derived4funcEi(ptr , i32)

!0 = !{i64 16, !"_ZTS4Base"}
!1 = !{i64 16, !"_ZTSM4BaseFiiE.virtual"}
!2 = !{i64 16, !"_ZTS7Derived"}
!3 = !{i64 16, !"_ZTSM7DerivedFiiE.virtual"}
