; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; Tests parsing for the dso_local keyword as well as the serialization/
; deserialization of the dso_local value on GlobalValues.

@local_global = dso_local global i32 0
; CHECK: @local_global = dso_local global i32 0

@weak_local_global = weak dso_local global i32 0
; CHECK: @weak_local_global = weak dso_local global i32 0

@external_local_global = external dso_local global i32
; CHECK: @external_local_global = external dso_local global i32

@default_local_global = dso_local default global i32 0
; CHECK: @default_local_global = dso_local global i32 0

@hidden_local_global = hidden global i32 0
; CHECK: @hidden_local_global = hidden global i32 0

@protected_local_global = protected global i32 0
; CHECK: @protected_local_global = protected global i32 0

@local_alias = dso_local alias i32, i32* @local_global
; CHECK-DAG: @local_alias = dso_local alias i32, i32* @local_global

@preemptable_alias = dso_preemptable alias i32, i32* @hidden_local_global
; CHECK-DAG: @preemptable_alias = alias i32, i32* @hidden_local_global

@preemptable_ifunc = dso_preemptable ifunc void (), i8* ()* @ifunc_resolver
; CHECK-DAG: @preemptable_ifunc = ifunc void (), i8* ()* @ifunc_resolver
declare dso_local default void @default_local()
; CHECK: declare dso_local void @default_local()

declare hidden void @hidden_local()
; CHECK: declare hidden void @hidden_local()

define protected void @protected_local() {
; CHECK: define protected void @protected_local()
entry:
  ret void
}

define i8* @ifunc_resolver() {
entry:
  ret i8* null
}
