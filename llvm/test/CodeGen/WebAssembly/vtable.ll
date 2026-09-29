; RUN: llc < %s -asm-verbose=false -wasm-disable-explicit-locals -wasm-keep-registers | FileCheck %s --check-prefix=TYPEINFONAME
; RUN: llc < %s -asm-verbose=false -wasm-disable-explicit-locals -wasm-keep-registers | FileCheck %s --check-prefix=VTABLE
; RUN: llc < %s -asm-verbose=false -wasm-disable-explicit-locals -wasm-keep-registers | FileCheck %s --check-prefix=TYPEINFO

; Test that simple vtables assemble as expected.
;
; The class hierarchy is:
;   struct A;
;   struct B : public A;
;   struct C : public A;
;   struct D : public B;
; Each with a virtual dtor and method foo.

target triple = "wasm32-unknown-unknown"

%struct.A = type { ptr }
%struct.B = type { %struct.A }
%struct.C = type { %struct.A }
%struct.D = type { %struct.B }

@_ZTVN10__cxxabiv117__class_type_infoE = external global ptr
@_ZTVN10__cxxabiv120__si_class_type_infoE = external global ptr

; TYPEINFONAME-LABEL: _ZTS1A:
; TYPEINFONAME-NEXT: .asciz "1A"
@_ZTS1A = constant [3 x i8] c"1A\00"
; TYPEINFONAME-LABEL: _ZTS1B:
; TYPEINFONAME-NEXT: .asciz "1B"
@_ZTS1B = constant [3 x i8] c"1B\00"
; TYPEINFONAME-LABEL: _ZTS1C:
; TYPEINFONAME-NEXT: .asciz "1C"
@_ZTS1C = constant [3 x i8] c"1C\00"
; TYPEINFONAME-LABEL: _ZTS1D:
; TYPEINFONAME-NEXT: .asciz "1D"
@_ZTS1D = constant [3 x i8] c"1D\00"

; VTABLE:       .type _ZTV1A,@object
; VTABLE-NEXT:  .section .rodata._ZTV1A,
; VTABLE-NEXT:  .globl _ZTV1A
; VTABLE-LABEL: _ZTV1A:
; VTABLE-NEXT:  .int32 0
; VTABLE-NEXT:  .int32 _ZTI1A
; VTABLE-NEXT:  .int32 _ZN1AD2Ev
; VTABLE-NEXT:  .int32 _ZN1AD0Ev
; VTABLE-NEXT:  .int32 _ZN1A3fooEv
; VTABLE-NEXT:  .size _ZTV1A, 20
@_ZTV1A = constant [5 x ptr] [ptr null, ptr @_ZTI1A, ptr @_ZN1AD2Ev, ptr @_ZN1AD0Ev, ptr @_ZN1A3fooEv], align 4
; VTABLE:       .type _ZTV1B,@object
; VTABLE-NEXT:  .section .rodata._ZTV1B,
; VTABLE-NEXT:  .globl _ZTV1B
; VTABLE-LABEL: _ZTV1B:
; VTABLE-NEXT:  .int32 0
; VTABLE-NEXT:  .int32 _ZTI1B
; VTABLE-NEXT:  .int32 _ZN1AD2Ev
; VTABLE-NEXT:  .int32 _ZN1BD0Ev
; VTABLE-NEXT:  .int32 _ZN1B3fooEv
; VTABLE-NEXT:  .size _ZTV1B, 20
@_ZTV1B = constant [5 x ptr] [ptr null, ptr @_ZTI1B, ptr @_ZN1AD2Ev, ptr @_ZN1BD0Ev, ptr @_ZN1B3fooEv], align 4
; VTABLE:       .type _ZTV1C,@object
; VTABLE-NEXT:  .section .rodata._ZTV1C,
; VTABLE-NEXT:  .globl _ZTV1C
; VTABLE-LABEL: _ZTV1C:
; VTABLE-NEXT:  .int32 0
; VTABLE-NEXT:  .int32 _ZTI1C
; VTABLE-NEXT:  .int32 _ZN1AD2Ev
; VTABLE-NEXT:  .int32 _ZN1CD0Ev
; VTABLE-NEXT:  .int32 _ZN1C3fooEv
; VTABLE-NEXT:  .size _ZTV1C, 20
@_ZTV1C = constant [5 x ptr] [ptr null, ptr @_ZTI1C, ptr @_ZN1AD2Ev, ptr @_ZN1CD0Ev, ptr @_ZN1C3fooEv], align 4
; VTABLE:       .type _ZTV1D,@object
; VTABLE-NEXT:  .section .rodata._ZTV1D,
; VTABLE-NEXT:  .globl _ZTV1D
; VTABLE-LABEL: _ZTV1D:
; VTABLE-NEXT:  .int32 0
; VTABLE-NEXT:  .int32 _ZTI1D
; VTABLE-NEXT:  .int32 _ZN1AD2Ev
; VTABLE-NEXT:  .int32 _ZN1DD0Ev
; VTABLE-NEXT:  .int32 _ZN1D3fooEv
; VTABLE-NEXT:  .size _ZTV1D, 20
@_ZTV1D = constant [5 x ptr] [ptr null, ptr @_ZTI1D, ptr @_ZN1AD2Ev, ptr @_ZN1DD0Ev, ptr @_ZN1D3fooEv], align 4

; TYPEINFO:       .type _ZTI1A,@object
; TYPEINFO:       .globl _ZTI1A
; TYPEINFO-LABEL: _ZTI1A:
; TYPEINFO-NEXT:  .int32 _ZTVN10__cxxabiv117__class_type_infoE+8
; TYPEINFO-NEXT:  .int32 _ZTS1A
; TYPEINFO-NEXT:  .size _ZTI1A, 8
@_ZTI1A = constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i32 2), ptr @_ZTS1A }
; TYPEINFO:       .type _ZTI1B,@object
; TYPEINFO:       .globl _ZTI1B
; TYPEINFO-LABEL: _ZTI1B:
; TYPEINFO-NEXT:  .int32 _ZTVN10__cxxabiv120__si_class_type_infoE+8
; TYPEINFO-NEXT:  .int32 _ZTS1B
; TYPEINFO-NEXT:  .int32 _ZTI1A
; TYPEINFO-NEXT:  .size _ZTI1B, 12
@_ZTI1B = constant { ptr, ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv120__si_class_type_infoE, i32 2), ptr @_ZTS1B, ptr @_ZTI1A }
; TYPEINFO:       .type _ZTI1C,@object
; TYPEINFO:       .globl _ZTI1C
; TYPEINFO-LABEL: _ZTI1C:
; TYPEINFO-NEXT:  .int32 _ZTVN10__cxxabiv120__si_class_type_infoE+8
; TYPEINFO-NEXT:  .int32 _ZTS1C
; TYPEINFO-NEXT:  .int32 _ZTI1A
; TYPEINFO-NEXT:  .size _ZTI1C, 12
@_ZTI1C = constant { ptr, ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv120__si_class_type_infoE, i32 2), ptr @_ZTS1C, ptr @_ZTI1A }
; TYPEINFO:       .type _ZTI1D,@object
; TYPEINFO:       .globl _ZTI1D
; TYPEINFO-LABEL: _ZTI1D:
; TYPEINFO-NEXT:  .int32 _ZTVN10__cxxabiv120__si_class_type_infoE+8
; TYPEINFO-NEXT:  .int32 _ZTS1D
; TYPEINFO-NEXT:  .int32 _ZTI1B
; TYPEINFO-NEXT:  .size _ZTI1D, 12
@_ZTI1D = constant { ptr, ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv120__si_class_type_infoE, i32 2), ptr @_ZTS1D, ptr @_ZTI1B }

@g = global i32 0, align 4

define void @_ZN1A3fooEv(ptr %this) {
entry:
  store i32 2, ptr @g, align 4
  ret void
}

define void @_ZN1B3fooEv(ptr %this) {
entry:
  store i32 4, ptr @g, align 4
  ret void
}

define void @_ZN1C3fooEv(ptr %this) {
entry:
  store i32 6, ptr @g, align 4
  ret void
}

define void @_ZN1D3fooEv(ptr %this) {
entry:
  store i32 8, ptr @g, align 4
  ret void
}

define linkonce_odr void @_ZN1AD0Ev(ptr %this) {
entry:
  tail call void @_ZdlPv(ptr %this)
  ret void
}

define linkonce_odr void @_ZN1BD0Ev(ptr %this) {
entry:
  tail call void @_ZdlPv(ptr %this)
  ret void
}

define linkonce_odr void @_ZN1CD0Ev(ptr %this) {
entry:
  tail call void @_ZdlPv(ptr %this)
  ret void
}

define linkonce_odr ptr @_ZN1AD2Ev(ptr returned %this) {
entry:
  ret ptr %this
}

define linkonce_odr void @_ZN1DD0Ev(ptr %this) {
entry:
  tail call void @_ZdlPv(ptr %this)
  ret void
}

declare void @_ZdlPv(ptr)
