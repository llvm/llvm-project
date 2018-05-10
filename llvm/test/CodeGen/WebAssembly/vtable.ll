; RUN: llc < %s -asm-verbose=false -disable-wasm-explicit-locals | FileCheck %s --check-prefix=TYPEINFONAME
; RUN: llc < %s -asm-verbose=false -disable-wasm-explicit-locals | FileCheck %s --check-prefix=VTABLE
; RUN: llc < %s -asm-verbose=false -disable-wasm-explicit-locals | FileCheck %s --check-prefix=TYPEINFO

; Test that simple vtables assemble as expected.
;
; The class hierarchy is:
;   struct A;
;   struct B : public A;
;   struct C : public A;
;   struct D : public B;
; Each with a virtual dtor and method foo.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

%struct.A = type { i32 (...)** }
%struct.B = type { %struct.A }
%struct.C = type { %struct.A }
%struct.D = type { %struct.B }

@_ZTVN10__cxxabiv117__class_type_infoE = external global i8*
@_ZTVN10__cxxabiv120__si_class_type_infoE = external global i8*

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
; VTABLE-NEXT:  .section .data.rel.ro._ZTV1A,
; VTABLE-NEXT:  .globl _ZTV1A
; VTABLE-LABEL: _ZTV1A:
; VTABLE-NEXT:  .int32 0
; VTABLE-NEXT:  .int32 _ZTI1A
; VTABLE-NEXT:  .int32 _ZN1AD2Ev
; VTABLE-NEXT:  .int32 _ZN1AD0Ev
; VTABLE-NEXT:  .int32 _ZN1A3fooEv
; VTABLE-NEXT:  .size _ZTV1A, 20
@_ZTV1A = constant [5 x i8*] [i8* null, i8* bitcast ({ i8*, i8* }* @_ZTI1A to i8*), i8* bitcast (%struct.A* (%struct.A*)* @_ZN1AD2Ev to i8*), i8* bitcast (void (%struct.A*)* @_ZN1AD0Ev to i8*), i8* bitcast (void (%struct.A*)* @_ZN1A3fooEv to i8*)], align 4
; VTABLE:       .type _ZTV1B,@object
; VTABLE-NEXT:  .section .data.rel.ro._ZTV1B,
; VTABLE-NEXT:  .globl _ZTV1B
; VTABLE-LABEL: _ZTV1B:
; VTABLE-NEXT:  .int32 0
; VTABLE-NEXT:  .int32 _ZTI1B
; VTABLE-NEXT:  .int32 _ZN1AD2Ev
; VTABLE-NEXT:  .int32 _ZN1BD0Ev
; VTABLE-NEXT:  .int32 _ZN1B3fooEv
; VTABLE-NEXT:  .size _ZTV1B, 20
@_ZTV1B = constant [5 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTI1B to i8*), i8* bitcast (%struct.A* (%struct.A*)* @_ZN1AD2Ev to i8*), i8* bitcast (void (%struct.B*)* @_ZN1BD0Ev to i8*), i8* bitcast (void (%struct.B*)* @_ZN1B3fooEv to i8*)], align 4
; VTABLE:       .type _ZTV1C,@object
; VTABLE-NEXT:  .section .data.rel.ro._ZTV1C,
; VTABLE-NEXT:  .globl _ZTV1C
; VTABLE-LABEL: _ZTV1C:
; VTABLE-NEXT:  .int32 0
; VTABLE-NEXT:  .int32 _ZTI1C
; VTABLE-NEXT:  .int32 _ZN1AD2Ev
; VTABLE-NEXT:  .int32 _ZN1CD0Ev
; VTABLE-NEXT:  .int32 _ZN1C3fooEv
; VTABLE-NEXT:  .size _ZTV1C, 20
@_ZTV1C = constant [5 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTI1C to i8*), i8* bitcast (%struct.A* (%struct.A*)* @_ZN1AD2Ev to i8*), i8* bitcast (void (%struct.C*)* @_ZN1CD0Ev to i8*), i8* bitcast (void (%struct.C*)* @_ZN1C3fooEv to i8*)], align 4
; VTABLE:       .type _ZTV1D,@object
; VTABLE-NEXT:  .section .data.rel.ro._ZTV1D,
; VTABLE-NEXT:  .globl _ZTV1D
; VTABLE-LABEL: _ZTV1D:
; VTABLE-NEXT:  .int32 0
; VTABLE-NEXT:  .int32 _ZTI1D
; VTABLE-NEXT:  .int32 _ZN1AD2Ev
; VTABLE-NEXT:  .int32 _ZN1DD0Ev
; VTABLE-NEXT:  .int32 _ZN1D3fooEv
; VTABLE-NEXT:  .size _ZTV1D, 20
@_ZTV1D = constant [5 x i8*] [i8* null, i8* bitcast ({ i8*, i8*, i8* }* @_ZTI1D to i8*), i8* bitcast (%struct.A* (%struct.A*)* @_ZN1AD2Ev to i8*), i8* bitcast (void (%struct.D*)* @_ZN1DD0Ev to i8*), i8* bitcast (void (%struct.D*)* @_ZN1D3fooEv to i8*)], align 4

; TYPEINFO:       .type _ZTI1A,@object
; TYPEINFO:       .globl _ZTI1A
; TYPEINFO-LABEL: _ZTI1A:
; TYPEINFO-NEXT:  .int32 _ZTVN10__cxxabiv117__class_type_infoE+8
; TYPEINFO-NEXT:  .int32 _ZTS1A
; TYPEINFO-NEXT:  .size _ZTI1A, 8
@_ZTI1A = constant { i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv117__class_type_infoE, i32 2) to i8*), i8* getelementptr inbounds ([3 x i8], [3 x i8]* @_ZTS1A, i32 0, i32 0) }
; TYPEINFO:       .type _ZTI1B,@object
; TYPEINFO:       .globl _ZTI1B
; TYPEINFO-LABEL: _ZTI1B:
; TYPEINFO-NEXT:  .int32 _ZTVN10__cxxabiv120__si_class_type_infoE+8
; TYPEINFO-NEXT:  .int32 _ZTS1B
; TYPEINFO-NEXT:  .int32 _ZTI1A
; TYPEINFO-NEXT:  .size _ZTI1B, 12
@_ZTI1B = constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i32 2) to i8*), i8* getelementptr inbounds ([3 x i8], [3 x i8]* @_ZTS1B, i32 0, i32 0), i8* bitcast ({ i8*, i8* }* @_ZTI1A to i8*) }
; TYPEINFO:       .type _ZTI1C,@object
; TYPEINFO:       .globl _ZTI1C
; TYPEINFO-LABEL: _ZTI1C:
; TYPEINFO-NEXT:  .int32 _ZTVN10__cxxabiv120__si_class_type_infoE+8
; TYPEINFO-NEXT:  .int32 _ZTS1C
; TYPEINFO-NEXT:  .int32 _ZTI1A
; TYPEINFO-NEXT:  .size _ZTI1C, 12
@_ZTI1C = constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i32 2) to i8*), i8* getelementptr inbounds ([3 x i8], [3 x i8]* @_ZTS1C, i32 0, i32 0), i8* bitcast ({ i8*, i8* }* @_ZTI1A to i8*) }
; TYPEINFO:       .type _ZTI1D,@object
; TYPEINFO:       .globl _ZTI1D
; TYPEINFO-LABEL: _ZTI1D:
; TYPEINFO-NEXT:  .int32 _ZTVN10__cxxabiv120__si_class_type_infoE+8
; TYPEINFO-NEXT:  .int32 _ZTS1D
; TYPEINFO-NEXT:  .int32 _ZTI1B
; TYPEINFO-NEXT:  .size _ZTI1D, 12
@_ZTI1D = constant { i8*, i8*, i8* } { i8* bitcast (i8** getelementptr inbounds (i8*, i8** @_ZTVN10__cxxabiv120__si_class_type_infoE, i32 2) to i8*), i8* getelementptr inbounds ([3 x i8], [3 x i8]* @_ZTS1D, i32 0, i32 0), i8* bitcast ({ i8*, i8*, i8* }* @_ZTI1B to i8*) }

@g = global i32 0, align 4

define void @_ZN1A3fooEv(%struct.A* %this) {
entry:
  store i32 2, i32* @g, align 4
  ret void
}

define void @_ZN1B3fooEv(%struct.B* %this) {
entry:
  store i32 4, i32* @g, align 4
  ret void
}

define void @_ZN1C3fooEv(%struct.C* %this) {
entry:
  store i32 6, i32* @g, align 4
  ret void
}

define void @_ZN1D3fooEv(%struct.D* %this) {
entry:
  store i32 8, i32* @g, align 4
  ret void
}

define linkonce_odr void @_ZN1AD0Ev(%struct.A* %this) {
entry:
  %0 = bitcast %struct.A* %this to i8*
  tail call void @_ZdlPv(i8* %0)
  ret void
}

define linkonce_odr void @_ZN1BD0Ev(%struct.B* %this) {
entry:
  %0 = bitcast %struct.B* %this to i8*
  tail call void @_ZdlPv(i8* %0)
  ret void
}

define linkonce_odr void @_ZN1CD0Ev(%struct.C* %this) {
entry:
  %0 = bitcast %struct.C* %this to i8*
  tail call void @_ZdlPv(i8* %0)
  ret void
}

define linkonce_odr %struct.A* @_ZN1AD2Ev(%struct.A* returned %this) {
entry:
  ret %struct.A* %this
}

define linkonce_odr void @_ZN1DD0Ev(%struct.D* %this) {
entry:
  %0 = bitcast %struct.D* %this to i8*
  tail call void @_ZdlPv(i8* %0)
  ret void
}

declare void @_ZdlPv(i8*)
