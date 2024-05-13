; RUN: llc -verify-machineinstrs < %s -mtriple=i686-win32 | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=i686-win32 -O0 | FileCheck %s

; IR simplified from the following C++ snippet compiled for i686-windows-msvc:

; struct A { A(); ~A(); int a; };
;
; struct B {
;   virtual int  f(int);
;   virtual int  g(A, int, A);
;   virtual void h(A, int, A);
;   virtual A    i(A, int, A);
;   virtual A    j(int);
; };
;
; int  (B::*mp_f)(int)       = &B::f;
; int  (B::*mp_g)(A, int, A) = &B::g;
; void (B::*mp_h)(A, int, A) = &B::h;
; A    (B::*mp_i)(A, int, A) = &B::i;
; A    (B::*mp_j)(int)       = &B::j;

; Each member pointer creates a thunk.  The ones with inalloca are required to
; tail calls by the ABI, even at O0.

%struct.B = type { ptr }
%struct.A = type { i32 }

; CHECK-LABEL: f_thunk:
; CHECK: jmpl
; CHECK-NOT: ret
define x86_thiscallcc i32 @f_thunk(ptr %this, i32) {
entry:
  %vtable = load ptr, ptr %this
  %1 = load ptr, ptr %vtable
  %2 = musttail call x86_thiscallcc i32 %1(ptr %this, i32 %0)
  ret i32 %2
}

; Inalloca thunks shouldn't require any stores to the stack.
; CHECK-LABEL: g_thunk:
; CHECK-NOT: mov %{{.*}}, {{.*(.*esp.*)}}
; CHECK: jmpl
; CHECK-NOT: ret
define x86_thiscallcc i32 @g_thunk(ptr %this, ptr inalloca(<{ %struct.A, i32, %struct.A }>)) {
entry:
  %vtable = load ptr, ptr %this
  %vfn = getelementptr inbounds ptr, ptr %vtable, i32 1
  %1 = load ptr, ptr %vfn
  %2 = musttail call x86_thiscallcc i32 %1(ptr %this, ptr inalloca(<{ %struct.A, i32, %struct.A }>) %0)
  ret i32 %2
}

; Preallocated thunks shouldn't require any stores to the stack.
; CHECK-LABEL: g_thunk_preallocated:
; CHECK-NOT: mov %{{.*}}, {{.*(.*esp.*)}}
; CHECK: jmpl
; CHECK-NOT: ret
define x86_thiscallcc i32 @g_thunk_preallocated(ptr %this, ptr preallocated(<{ %struct.A, i32, %struct.A }>)) {
entry:
  %vtable = load ptr, ptr %this
  %vfn = getelementptr inbounds ptr, ptr %vtable, i32 1
  %1 = load ptr, ptr %vfn
  %2 = musttail call x86_thiscallcc i32 %1(ptr %this, ptr preallocated(<{ %struct.A, i32, %struct.A }>) %0)
  ret i32 %2
}

; CHECK-LABEL: h_thunk:
; CHECK: jmpl
; CHECK-NOT: mov %{{.*}}, {{.*(.*esp.*)}}
; CHECK-NOT: ret
define x86_thiscallcc void @h_thunk(ptr %this, ptr inalloca(<{ %struct.A, i32, %struct.A }>)) {
entry:
  %vtable = load ptr, ptr %this
  %vfn = getelementptr inbounds ptr, ptr %vtable, i32 2
  %1 = load ptr, ptr %vfn
  musttail call x86_thiscallcc void %1(ptr %this, ptr inalloca(<{ %struct.A, i32, %struct.A }>) %0)
  ret void
}

; CHECK-LABEL: h_thunk_preallocated:
; CHECK: jmpl
; CHECK-NOT: mov %{{.*}}, {{.*(.*esp.*)}}
; CHECK-NOT: ret
define x86_thiscallcc void @h_thunk_preallocated(ptr %this, ptr preallocated(<{ %struct.A, i32, %struct.A }>)) {
entry:
  %vtable = load ptr, ptr %this
  %vfn = getelementptr inbounds ptr, ptr %vtable, i32 2
  %1 = load ptr, ptr %vfn
  musttail call x86_thiscallcc void %1(ptr %this, ptr preallocated(<{ %struct.A, i32, %struct.A }>) %0)
  ret void
}

; CHECK-LABEL: i_thunk:
; CHECK-NOT: mov %{{.*}}, {{.*(.*esp.*)}}
; CHECK: jmpl
; CHECK-NOT: ret
define x86_thiscallcc ptr @i_thunk(ptr %this, ptr inalloca(<{ ptr, %struct.A, i32, %struct.A }>)) {
entry:
  %vtable = load ptr, ptr %this
  %vfn = getelementptr inbounds ptr, ptr %vtable, i32 3
  %1 = load ptr, ptr %vfn
  %2 = musttail call x86_thiscallcc ptr %1(ptr %this, ptr inalloca(<{ ptr, %struct.A, i32, %struct.A }>) %0)
  ret ptr %2
}

; CHECK-LABEL: i_thunk_preallocated:
; CHECK-NOT: mov %{{.*}}, {{.*(.*esp.*)}}
; CHECK: jmpl
; CHECK-NOT: ret
define x86_thiscallcc ptr @i_thunk_preallocated(ptr %this, ptr preallocated(<{ ptr, %struct.A, i32, %struct.A }>)) {
entry:
  %vtable = load ptr, ptr %this
  %vfn = getelementptr inbounds ptr, ptr %vtable, i32 3
  %1 = load ptr, ptr %vfn
  %2 = musttail call x86_thiscallcc ptr %1(ptr %this, ptr preallocated(<{ ptr, %struct.A, i32, %struct.A }>) %0)
  ret ptr %2
}

; CHECK-LABEL: j_thunk:
; CHECK: jmpl
; CHECK-NOT: ret
define x86_thiscallcc void @j_thunk(ptr noalias sret(%struct.A) %agg.result, ptr %this, i32) {
entry:
  %vtable = load ptr, ptr %this
  %vfn = getelementptr inbounds ptr, ptr %vtable, i32 4
  %1 = load ptr, ptr %vfn
  musttail call x86_thiscallcc void %1(ptr sret(%struct.A) %agg.result, ptr %this, i32 %0)
  ret void
}

; CHECK-LABEL: _stdcall_thunk@8:
; CHECK-NOT: mov %{{.*}}, {{.*(.*esp.*)}}
; CHECK: jmpl
; CHECK-NOT: ret
define x86_stdcallcc i32 @stdcall_thunk(ptr inalloca(<{ ptr, %struct.A }>)) {
entry:
  %this_ptr = getelementptr inbounds <{ ptr, %struct.A }>, ptr %0, i32 0, i32 0
  %this = load ptr, ptr %this_ptr
  %vtable = load ptr, ptr %this
  %vfn = getelementptr inbounds ptr, ptr %vtable, i32 1
  %1 = load ptr, ptr %vfn
  %2 = musttail call x86_stdcallcc i32 %1(ptr inalloca(<{ ptr, %struct.A }>) %0)
  ret i32 %2
}

; CHECK-LABEL: _stdcall_thunk_preallocated@8:
; CHECK-NOT: mov %{{.*}}, {{.*(.*esp.*)}}
; CHECK: jmpl
; CHECK-NOT: ret
define x86_stdcallcc i32 @stdcall_thunk_preallocated(ptr preallocated(<{ ptr, %struct.A }>)) {
entry:
  %this_ptr = getelementptr inbounds <{ ptr, %struct.A }>, ptr %0, i32 0, i32 0
  %this = load ptr, ptr %this_ptr
  %vtable = load ptr, ptr %this
  %vfn = getelementptr inbounds ptr, ptr %vtable, i32 1
  %1 = load ptr, ptr %vfn
  %2 = musttail call x86_stdcallcc i32 %1(ptr preallocated(<{ ptr, %struct.A }>) %0)
  ret i32 %2
}

; CHECK-LABEL: @fastcall_thunk@8:
; CHECK-NOT: mov %{{.*}}, {{.*(.*esp.*)}}
; CHECK: jmpl
; CHECK-NOT: ret
define x86_fastcallcc i32 @fastcall_thunk(ptr inreg %this, ptr inalloca(<{ %struct.A }>)) {
entry:
  %vtable = load ptr, ptr %this
  %vfn = getelementptr inbounds ptr, ptr %vtable, i32 1
  %1 = load ptr, ptr %vfn
  %2 = musttail call x86_fastcallcc i32 %1(ptr inreg %this, ptr inalloca(<{ %struct.A }>) %0)
  ret i32 %2
}

; CHECK-LABEL: @fastcall_thunk_preallocated@8:
; CHECK-NOT: mov %{{.*}}, {{.*(.*esp.*)}}
; CHECK: jmpl
; CHECK-NOT: ret
define x86_fastcallcc i32 @fastcall_thunk_preallocated(ptr inreg %this, ptr preallocated(<{ %struct.A }>)) {
entry:
  %vtable = load ptr, ptr %this
  %vfn = getelementptr inbounds ptr, ptr %vtable, i32 1
  %1 = load ptr, ptr %vfn
  %2 = musttail call x86_fastcallcc i32 %1(ptr inreg %this, ptr preallocated(<{ %struct.A }>) %0)
  ret i32 %2
}
