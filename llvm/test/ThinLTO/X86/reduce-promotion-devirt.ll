; Test to ensure that devirtualization promotes an internal virtual target to a
; uniqued hidden symbol from a static function, and later on the static funciton
; is promoted to the same hidden symbol.

; Set up
; RUN: rm -rf %t
; RUN: mkdir -p %t
; RUN: split-file %s %t

; RUN: opt -thinlto-bc %t/a.ll -o %t/a.bc
; RUN: opt -thinlto-bc %t/b.ll -o %t/b.bc

; RUN: llvm-lto2 run %t/a.bc %t/b.bc \
; RUN:   --whole-program-visibility-enabled-in-lto=true \
; RUN:   -disable-always-rename-promoted-locals \
; RUN:   -save-temps -o %t/lto-out \
; RUN:   -r %t/a.bc,test,px \
; RUN:   -r %t/b.bc,_ZN1B1fEi,p \
; RUN:   -r %t/b.bc,test2,px \
; RUN:   -r %t/b.bc,_ZTV1B,px

; RUN: llvm-nm %t/lto-out.1 | FileCheck %s --check-prefix=NM-INDEX1
; RUN: llvm-nm %t/lto-out.2 | FileCheck %s --check-prefix=NM-INDEX2

; NM-INDEX1: U _ZN1A1nEi.llvm.
; NM-INDEX2-NOT: U _ZN1A1nEi
; NM-INDEX2: T _ZN1A1nEi.llvm.
; NM-INDEX2-NOT: U _ZN1A1nEi

; RUN: llvm-dis %t/lto-out.1.4.opt.bc -o - | FileCheck %s --check-prefix=OPT-INDEX1
; RUN: llvm-dis %t/lto-out.2.4.opt.bc -o - | FileCheck %s --check-prefix=OPT-INDEX2

; OPT-INDEX1: %call = tail call i32 @_ZN1A1nEi.llvm.
; OPT-INDEX1: declare void @_ZN1A1nEi.llvm.

; OPT-INDEX2: ptr @_ZN1A1nEi.llvm.
; OPT-INDEX2: define hidden i32 @_ZN1A1nEi.llvm.
; OPT-INDEX2: tail call i32 @_ZN1A1nEi.llvm.

;--- a.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%struct.A = type { ptr }

define i32 @test(ptr %obj, i32 %a) {
entry:
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"_ZTS1A")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr ptr, ptr %vtable, i32 1
  %fptr1 = load ptr, ptr %fptrptr, align 8

  ; the call was devirtualized.
  %call = tail call i32 %fptr1(ptr nonnull %obj, i32 %a)
  ret i32 %call
}

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1)

;--- b.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%struct.A = type { ptr }
%struct.B = type { %struct.A }

@_ZTV1B = constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr poison, ptr @_ZN1B1fEi, ptr @_ZN1A1nEi] }, !type !0, !type !1

define i32 @_ZN1B1fEi(ptr %this, i32 %a) #0 {
   ret i32 0;
}

define internal i32 @_ZN1A1nEi(ptr %this, i32 %a) #0 {
   ret i32 0;
}

define i32 @test2(ptr %obj, i32 %a) {
entry:
  %vtable2 = load ptr, ptr %obj
  %p2 = call i1 @llvm.type.test(ptr %vtable2, metadata !"_ZTS1B")
  call void @llvm.assume(i1 %p2)

  %fptrptr = getelementptr ptr, ptr %vtable2, i32 1
  %fptr33 = load ptr, ptr %fptrptr, align 8

  %call4 = tail call i32 %fptr33(ptr nonnull %obj, i32 %a)
  ret i32 %call4
}

attributes #0 = { noinline optnone }

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1)

!0 = !{i64 16, !"_ZTS1A"}
!1 = !{i64 16, !"_ZTS1B"}
