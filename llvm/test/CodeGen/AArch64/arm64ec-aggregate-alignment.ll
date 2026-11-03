; RUN: llc -mtriple=arm64ec-pc-windows-msvc -O0 -stop-after=Arm64ECCallLowering %s -o - | FileCheck %s --check-prefix=IR
; RUN: llc -mtriple=arm64ec-pc-windows-msvc -O2 -stop-after=Arm64ECCallLowering %s -o - | FileCheck %s --check-prefix=IR
; RUN: llc -mtriple=arm64ec-pc-windows-msvc -O2 %s -o - | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=arm64ec-pc-windows-msvc -O0 -verify-machineinstrs -filetype=obj %s -o %t.O0.obj
; RUN: llc -mtriple=arm64ec-pc-windows-msvc -O2 -verify-machineinstrs -filetype=obj %s -o %t.O2.obj

; These 16-byte aggregates need different native argument registers after a
; pointer. Their entry and exit thunks must not share size-only names.
define void @aligned_store(ptr %out, i128 %value) {
  store i128 %value, ptr %out, align 16
  ret void
}

define void @pair_store(ptr %out, [2 x i64] %value) {
  store [2 x i64] %value, ptr %out, align 8
  ret void
}

define void @call_aligned(ptr %callee, ptr %out, i128 %value) {
  call void %callee(ptr %out, i128 %value)
  ret void
}

define void @call_pair(ptr %callee, ptr %out, [2 x i64] %value) {
  call void %callee(ptr %out, [2 x i64] %value)
  ret void
}

; IR-LABEL: define linkonce_odr {{.*}}void @"$iexit_thunk$cdecl$v$i8m16a16"(
; IR: store i128 {{.*}}, ptr %[[ALIGNED_SLOT:[^ ,]+]], align 16
; IR-NEXT: call cc108 void {{.*}}(ptr {{.*}}, i64 {{.*}}, ptr %[[ALIGNED_SLOT]])
; IR-NEXT: ret void
; IR-LABEL: define linkonce_odr {{.*}}void @"$iexit_thunk$cdecl$v$i8m16"(
; IR: store [2 x i64] {{.*}}, ptr %[[PAIR_SLOT:[^ ,]+]], align 8
; IR-NEXT: call cc108 void {{.*}}(ptr {{.*}}, i64 {{.*}}, ptr %[[PAIR_SLOT]])
; IR-NEXT: ret void
; IR-LABEL: define linkonce_odr {{.*}}void @"$ientry_thunk$cdecl$v$i8m16a16"(
; IR: %[[ALIGNED_ARG:[^ ]+]] = load i128, ptr {{.*}}, align 16
; IR-NEXT: call void {{.*}}(i64 {{.*}}, i128 %[[ALIGNED_ARG]])
; IR-NEXT: ret void
; IR-LABEL: define linkonce_odr {{.*}}void @"$ientry_thunk$cdecl$v$i8m16"(
; IR: %[[PAIR_ARG:[^ ]+]] = load [2 x i64], ptr {{.*}}, align 8
; IR-NEXT: call void {{.*}}(i64 {{.*}}, [2 x i64] %[[PAIR_ARG]])
; IR-NEXT: ret void

; ASM-LABEL: $ientry_thunk$cdecl$v$i8m16a16:
; ASM: ldp x2, x3, [x1]
; ASM-NEXT: blr x9
; ASM-LABEL: $ientry_thunk$cdecl$v$i8m16:
; ASM: ldp x1, x2, [x1]
; ASM-NEXT: blr x9
