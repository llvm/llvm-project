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

%struct.B = type { i32 (...)** }
%struct.A = type { i32 }

; CHECK-LABEL: f_thunk:
; CHECK: jmpl
; CHECK-NOT: ret
define x86_thiscallcc i32 @f_thunk(%struct.B* %this, i32) {
entry:
  %1 = bitcast %struct.B* %this to i32 (%struct.B*, i32)***
  %vtable = load i32 (%struct.B*, i32)**, i32 (%struct.B*, i32)*** %1
  %2 = load i32 (%struct.B*, i32)*, i32 (%struct.B*, i32)** %vtable
  %3 = musttail call x86_thiscallcc i32 %2(%struct.B* %this, i32 %0)
  ret i32 %3
}

; Inalloca thunks shouldn't require any stores to the stack.
; CHECK-LABEL: g_thunk:
; CHECK-NOT: mov %{{.*}}, {{.*(.*esp.*)}}
; CHECK: jmpl
; CHECK-NOT: ret
define x86_thiscallcc i32 @g_thunk(%struct.B* %this, <{ %struct.A, i32, %struct.A }>* inalloca) {
entry:
  %1 = bitcast %struct.B* %this to i32 (%struct.B*, <{ %struct.A, i32, %struct.A }>*)***
  %vtable = load i32 (%struct.B*, <{ %struct.A, i32, %struct.A }>*)**, i32 (%struct.B*, <{ %struct.A, i32, %struct.A }>*)*** %1
  %vfn = getelementptr inbounds i32 (%struct.B*, <{ %struct.A, i32, %struct.A }>*)*, i32 (%struct.B*, <{ %struct.A, i32, %struct.A }>*)** %vtable, i32 1
  %2 = load i32 (%struct.B*, <{ %struct.A, i32, %struct.A }>*)*, i32 (%struct.B*, <{ %struct.A, i32, %struct.A }>*)** %vfn
  %3 = musttail call x86_thiscallcc i32 %2(%struct.B* %this, <{ %struct.A, i32, %struct.A }>* inalloca %0)
  ret i32 %3
}

; Preallocated thunks shouldn't require any stores to the stack.
; CHECK-LABEL: g_thunk_preallocated:
; CHECK-NOT: mov %{{.*}}, {{.*(.*esp.*)}}
; CHECK: jmpl
; CHECK-NOT: ret
define x86_thiscallcc i32 @g_thunk_preallocated(%struct.B* %this, <{ %struct.A, i32, %struct.A }>* preallocated(<{ %struct.A, i32, %struct.A }>)) {
entry:
  %1 = bitcast %struct.B* %this to i32 (%struct.B*, <{ %struct.A, i32, %struct.A }>*)***
  %vtable = load i32 (%struct.B*, <{ %struct.A, i32, %struct.A }>*)**, i32 (%struct.B*, <{ %struct.A, i32, %struct.A }>*)*** %1
  %vfn = getelementptr inbounds i32 (%struct.B*, <{ %struct.A, i32, %struct.A }>*)*, i32 (%struct.B*, <{ %struct.A, i32, %struct.A }>*)** %vtable, i32 1
  %2 = load i32 (%struct.B*, <{ %struct.A, i32, %struct.A }>*)*, i32 (%struct.B*, <{ %struct.A, i32, %struct.A }>*)** %vfn
  %3 = musttail call x86_thiscallcc i32 %2(%struct.B* %this, <{ %struct.A, i32, %struct.A }>* preallocated(<{ %struct.A, i32, %struct.A }>) %0)
  ret i32 %3
}

; CHECK-LABEL: h_thunk:
; CHECK: jmpl
; CHECK-NOT: mov %{{.*}}, {{.*(.*esp.*)}}
; CHECK-NOT: ret
define x86_thiscallcc void @h_thunk(%struct.B* %this, <{ %struct.A, i32, %struct.A }>* inalloca) {
entry:
  %1 = bitcast %struct.B* %this to void (%struct.B*, <{ %struct.A, i32, %struct.A }>*)***
  %vtable = load void (%struct.B*, <{ %struct.A, i32, %struct.A }>*)**, void (%struct.B*, <{ %struct.A, i32, %struct.A }>*)*** %1
  %vfn = getelementptr inbounds void (%struct.B*, <{ %struct.A, i32, %struct.A }>*)*, void (%struct.B*, <{ %struct.A, i32, %struct.A }>*)** %vtable, i32 2
  %2 = load void (%struct.B*, <{ %struct.A, i32, %struct.A }>*)*, void (%struct.B*, <{ %struct.A, i32, %struct.A }>*)** %vfn
  musttail call x86_thiscallcc void %2(%struct.B* %this, <{ %struct.A, i32, %struct.A }>* inalloca %0)
  ret void
}

; CHECK-LABEL: h_thunk_preallocated:
; CHECK: jmpl
; CHECK-NOT: mov %{{.*}}, {{.*(.*esp.*)}}
; CHECK-NOT: ret
define x86_thiscallcc void @h_thunk_preallocated(%struct.B* %this, <{ %struct.A, i32, %struct.A }>* preallocated(<{ %struct.A, i32, %struct.A }>)) {
entry:
  %1 = bitcast %struct.B* %this to void (%struct.B*, <{ %struct.A, i32, %struct.A }>*)***
  %vtable = load void (%struct.B*, <{ %struct.A, i32, %struct.A }>*)**, void (%struct.B*, <{ %struct.A, i32, %struct.A }>*)*** %1
  %vfn = getelementptr inbounds void (%struct.B*, <{ %struct.A, i32, %struct.A }>*)*, void (%struct.B*, <{ %struct.A, i32, %struct.A }>*)** %vtable, i32 2
  %2 = load void (%struct.B*, <{ %struct.A, i32, %struct.A }>*)*, void (%struct.B*, <{ %struct.A, i32, %struct.A }>*)** %vfn
  musttail call x86_thiscallcc void %2(%struct.B* %this, <{ %struct.A, i32, %struct.A }>* preallocated(<{ %struct.A, i32, %struct.A }>) %0)
  ret void
}

; CHECK-LABEL: i_thunk:
; CHECK-NOT: mov %{{.*}}, {{.*(.*esp.*)}}
; CHECK: jmpl
; CHECK-NOT: ret
define x86_thiscallcc %struct.A* @i_thunk(%struct.B* %this, <{ %struct.A*, %struct.A, i32, %struct.A }>* inalloca) {
entry:
  %1 = bitcast %struct.B* %this to %struct.A* (%struct.B*, <{ %struct.A*, %struct.A, i32, %struct.A }>*)***
  %vtable = load %struct.A* (%struct.B*, <{ %struct.A*, %struct.A, i32, %struct.A }>*)**, %struct.A* (%struct.B*, <{ %struct.A*, %struct.A, i32, %struct.A }>*)*** %1
  %vfn = getelementptr inbounds %struct.A* (%struct.B*, <{ %struct.A*, %struct.A, i32, %struct.A }>*)*, %struct.A* (%struct.B*, <{ %struct.A*, %struct.A, i32, %struct.A }>*)** %vtable, i32 3
  %2 = load %struct.A* (%struct.B*, <{ %struct.A*, %struct.A, i32, %struct.A }>*)*, %struct.A* (%struct.B*, <{ %struct.A*, %struct.A, i32, %struct.A }>*)** %vfn
  %3 = musttail call x86_thiscallcc %struct.A* %2(%struct.B* %this, <{ %struct.A*, %struct.A, i32, %struct.A }>* inalloca %0)
  ret %struct.A* %3
}

; CHECK-LABEL: i_thunk_preallocated:
; CHECK-NOT: mov %{{.*}}, {{.*(.*esp.*)}}
; CHECK: jmpl
; CHECK-NOT: ret
define x86_thiscallcc %struct.A* @i_thunk_preallocated(%struct.B* %this, <{ %struct.A*, %struct.A, i32, %struct.A }>* preallocated(<{ %struct.A*, %struct.A, i32, %struct.A }>)) {
entry:
  %1 = bitcast %struct.B* %this to %struct.A* (%struct.B*, <{ %struct.A*, %struct.A, i32, %struct.A }>*)***
  %vtable = load %struct.A* (%struct.B*, <{ %struct.A*, %struct.A, i32, %struct.A }>*)**, %struct.A* (%struct.B*, <{ %struct.A*, %struct.A, i32, %struct.A }>*)*** %1
  %vfn = getelementptr inbounds %struct.A* (%struct.B*, <{ %struct.A*, %struct.A, i32, %struct.A }>*)*, %struct.A* (%struct.B*, <{ %struct.A*, %struct.A, i32, %struct.A }>*)** %vtable, i32 3
  %2 = load %struct.A* (%struct.B*, <{ %struct.A*, %struct.A, i32, %struct.A }>*)*, %struct.A* (%struct.B*, <{ %struct.A*, %struct.A, i32, %struct.A }>*)** %vfn
  %3 = musttail call x86_thiscallcc %struct.A* %2(%struct.B* %this, <{ %struct.A*, %struct.A, i32, %struct.A }>* preallocated(<{ %struct.A*, %struct.A, i32, %struct.A }>) %0)
  ret %struct.A* %3
}

; CHECK-LABEL: j_thunk:
; CHECK: jmpl
; CHECK-NOT: ret
define x86_thiscallcc void @j_thunk(%struct.A* noalias sret(%struct.A) %agg.result, %struct.B* %this, i32) {
entry:
  %1 = bitcast %struct.B* %this to void (%struct.A*, %struct.B*, i32)***
  %vtable = load void (%struct.A*, %struct.B*, i32)**, void (%struct.A*, %struct.B*, i32)*** %1
  %vfn = getelementptr inbounds void (%struct.A*, %struct.B*, i32)*, void (%struct.A*, %struct.B*, i32)** %vtable, i32 4
  %2 = load void (%struct.A*, %struct.B*, i32)*, void (%struct.A*, %struct.B*, i32)** %vfn
  musttail call x86_thiscallcc void %2(%struct.A* sret(%struct.A) %agg.result, %struct.B* %this, i32 %0)
  ret void
}

; CHECK-LABEL: _stdcall_thunk@8:
; CHECK-NOT: mov %{{.*}}, {{.*(.*esp.*)}}
; CHECK: jmpl
; CHECK-NOT: ret
define x86_stdcallcc i32 @stdcall_thunk(<{ %struct.B*, %struct.A }>* inalloca) {
entry:
  %this_ptr = getelementptr inbounds <{ %struct.B*, %struct.A }>, <{ %struct.B*, %struct.A }>* %0, i32 0, i32 0
  %this = load %struct.B*, %struct.B** %this_ptr
  %1 = bitcast %struct.B* %this to i32 (<{ %struct.B*, %struct.A }>*)***
  %vtable = load i32 (<{ %struct.B*, %struct.A }>*)**, i32 (<{ %struct.B*, %struct.A }>*)*** %1
  %vfn = getelementptr inbounds i32 (<{ %struct.B*, %struct.A }>*)*, i32 (<{ %struct.B*, %struct.A }>*)** %vtable, i32 1
  %2 = load i32 (<{ %struct.B*, %struct.A }>*)*, i32 (<{ %struct.B*, %struct.A }>*)** %vfn
  %3 = musttail call x86_stdcallcc i32 %2(<{ %struct.B*, %struct.A }>* inalloca %0)
  ret i32 %3
}

; CHECK-LABEL: _stdcall_thunk_preallocated@8:
; CHECK-NOT: mov %{{.*}}, {{.*(.*esp.*)}}
; CHECK: jmpl
; CHECK-NOT: ret
define x86_stdcallcc i32 @stdcall_thunk_preallocated(<{ %struct.B*, %struct.A }>* preallocated(<{ %struct.B*, %struct.A }>)) {
entry:
  %this_ptr = getelementptr inbounds <{ %struct.B*, %struct.A }>, <{ %struct.B*, %struct.A }>* %0, i32 0, i32 0
  %this = load %struct.B*, %struct.B** %this_ptr
  %1 = bitcast %struct.B* %this to i32 (<{ %struct.B*, %struct.A }>*)***
  %vtable = load i32 (<{ %struct.B*, %struct.A }>*)**, i32 (<{ %struct.B*, %struct.A }>*)*** %1
  %vfn = getelementptr inbounds i32 (<{ %struct.B*, %struct.A }>*)*, i32 (<{ %struct.B*, %struct.A }>*)** %vtable, i32 1
  %2 = load i32 (<{ %struct.B*, %struct.A }>*)*, i32 (<{ %struct.B*, %struct.A }>*)** %vfn
  %3 = musttail call x86_stdcallcc i32 %2(<{ %struct.B*, %struct.A }>* preallocated(<{ %struct.B*, %struct.A }>) %0)
  ret i32 %3
}

; CHECK-LABEL: @fastcall_thunk@8:
; CHECK-NOT: mov %{{.*}}, {{.*(.*esp.*)}}
; CHECK: jmpl
; CHECK-NOT: ret
define x86_fastcallcc i32 @fastcall_thunk(%struct.B* inreg %this, <{ %struct.A }>* inalloca) {
entry:
  %1 = bitcast %struct.B* %this to i32 (%struct.B*, <{ %struct.A }>*)***
  %vtable = load i32 (%struct.B*, <{ %struct.A }>*)**, i32 (%struct.B*, <{ %struct.A }>*)*** %1
  %vfn = getelementptr inbounds i32 (%struct.B*, <{ %struct.A }>*)*, i32 (%struct.B*, <{ %struct.A }>*)** %vtable, i32 1
  %2 = load i32 (%struct.B*, <{ %struct.A }>*)*, i32 (%struct.B*, <{ %struct.A }>*)** %vfn
  %3 = musttail call x86_fastcallcc i32 %2(%struct.B* inreg %this, <{ %struct.A }>* inalloca %0)
  ret i32 %3
}

; CHECK-LABEL: @fastcall_thunk_preallocated@8:
; CHECK-NOT: mov %{{.*}}, {{.*(.*esp.*)}}
; CHECK: jmpl
; CHECK-NOT: ret
define x86_fastcallcc i32 @fastcall_thunk_preallocated(%struct.B* inreg %this, <{ %struct.A }>* preallocated(<{ %struct.A }>)) {
entry:
  %1 = bitcast %struct.B* %this to i32 (%struct.B*, <{ %struct.A }>*)***
  %vtable = load i32 (%struct.B*, <{ %struct.A }>*)**, i32 (%struct.B*, <{ %struct.A }>*)*** %1
  %vfn = getelementptr inbounds i32 (%struct.B*, <{ %struct.A }>*)*, i32 (%struct.B*, <{ %struct.A }>*)** %vtable, i32 1
  %2 = load i32 (%struct.B*, <{ %struct.A }>*)*, i32 (%struct.B*, <{ %struct.A }>*)** %vfn
  %3 = musttail call x86_fastcallcc i32 %2(%struct.B* inreg %this, <{ %struct.A }>* preallocated(<{ %struct.A }>) %0)
  ret i32 %3
}
