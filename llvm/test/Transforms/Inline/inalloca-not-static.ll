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

declare ptr @llvm.stacksave()
declare void @llvm.stackrestore(ptr)

declare x86_thiscallcc ptr @"\01??0Foo@@QAE@XZ"(ptr returned) unnamed_addr
declare x86_thiscallcc void @"\01??1Foo@@QAE@XZ"(ptr) unnamed_addr

define void @f() {
entry:
  call void @g()
  ret void
}

define internal void @g() alwaysinline {
entry:
  %inalloca.save = call ptr @llvm.stacksave()
  %argmem = alloca inalloca <{ %struct.Foo }>, align 4
  %call = call x86_thiscallcc ptr @"\01??0Foo@@QAE@XZ"(ptr %argmem)
  call void @h(ptr inalloca(<{ %struct.Foo }>) %argmem)
  call void @llvm.stackrestore(ptr %inalloca.save)
  ret void
}

; Function Attrs: alwaysinline inlinehint nounwind
define internal void @h(ptr inalloca(<{ %struct.Foo }>)) alwaysinline {
entry:
  call x86_thiscallcc void @"\01??1Foo@@QAE@XZ"(ptr %0)
  ret void
}

; CHECK: define void @f()
; CHECK:   %[[STACKSAVE:.*]] = call ptr @llvm.stacksave()
; CHECK:   %[[ARGMEM:.*]] = alloca inalloca <{ %struct.Foo }>, align 4
; CHECK:   %[[CALL:.*]] = call x86_thiscallcc ptr @"\01??0Foo@@QAE@XZ"(ptr %[[ARGMEM]])
; CHECK:   call x86_thiscallcc void @"\01??1Foo@@QAE@XZ"(ptr %[[ARGMEM]])
; CHECK:   call void @llvm.stackrestore(ptr %[[STACKSAVE]])
; CHECK:   ret void
