; RUN: opt -always-inline -S < %s | FileCheck %s
; RUN: opt -passes=always-inline -S < %s | FileCheck %s

; We used to misclassify inalloca as a static alloca in the inliner. This only
; arose with for alwaysinline functions, because the normal inliner refuses to
; inline such things.

; Generated using this C++ source:
; struct Foo {
;   Foo();
;   Foo(const Foo &o);
;   ~Foo();
;   int a;
; };
; __forceinline void h(Foo o) {}
; __forceinline void g() { h(Foo()); }
; void f() { g(); }

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i386-pc-windows-msvc19.0.24210"

%struct.Foo = type { i32 }

declare i8* @llvm.stacksave()
declare void @llvm.stackrestore(i8*)

declare x86_thiscallcc %struct.Foo* @"\01??0Foo@@QAE@XZ"(%struct.Foo* returned) unnamed_addr
declare x86_thiscallcc void @"\01??1Foo@@QAE@XZ"(%struct.Foo*) unnamed_addr

define void @f() {
entry:
  call void @g()
  ret void
}

define internal void @g() alwaysinline {
entry:
  %inalloca.save = call i8* @llvm.stacksave()
  %argmem = alloca inalloca <{ %struct.Foo }>, align 4
  %0 = getelementptr inbounds <{ %struct.Foo }>, <{ %struct.Foo }>* %argmem, i32 0, i32 0
  %call = call x86_thiscallcc %struct.Foo* @"\01??0Foo@@QAE@XZ"(%struct.Foo* %0)
  call void @h(<{ %struct.Foo }>* inalloca %argmem)
  call void @llvm.stackrestore(i8* %inalloca.save)
  ret void
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @h(<{ %struct.Foo }>* inalloca) alwaysinline {
entry:
  %o = getelementptr inbounds <{ %struct.Foo }>, <{ %struct.Foo }>* %0, i32 0, i32 0
  call x86_thiscallcc void @"\01??1Foo@@QAE@XZ"(%struct.Foo* %o)
  ret void
}

; CHECK: define void @f()
; CHECK:   %[[STACKSAVE:.*]] = call i8* @llvm.stacksave()
; CHECK:   %[[ARGMEM:.*]] = alloca inalloca <{ %struct.Foo }>, align 4
; CHECK:   %[[GEP1:.*]] = getelementptr inbounds <{ %struct.Foo }>, <{ %struct.Foo }>* %[[ARGMEM]], i32 0, i32 0
; CHECK:   %[[CALL:.*]] = call x86_thiscallcc %struct.Foo* @"\01??0Foo@@QAE@XZ"(%struct.Foo* %[[GEP1]])
; CHECK:   %[[GEP2:.*]] = getelementptr inbounds <{ %struct.Foo }>, <{ %struct.Foo }>* %[[ARGMEM]], i32 0, i32 0
; CHECK:   call x86_thiscallcc void @"\01??1Foo@@QAE@XZ"(%struct.Foo* %[[GEP2]])
; CHECK:   call void @llvm.stackrestore(i8* %[[STACKSAVE]])
; CHECK:   ret void
