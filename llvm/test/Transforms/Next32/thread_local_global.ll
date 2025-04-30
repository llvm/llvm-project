; RUN: opt -S -passes=next-silicon-ir-fixup %s | FileCheck %s

$second_tls_access = comdat any

@i = thread_local global i32 0

define void @_tls_access() {
  %1 = call ptr @llvm.threadlocal.address.p0(ptr @i)
  store i32 9, ptr %1
  ret void
}

declare nonnull ptr @llvm.threadlocal.address.p0(ptr nonnull)

define weak_odr hidden noundef ptr @second_tls_access() comdat {
  %1 = call ptr @llvm.threadlocal.address.p0(ptr @i)
  ret ptr %1
}

; CHECK-LABEL: @_tls_access(
; CHECK: call ptr @__next_tls_var_location_i()
; CHECK-NOT: call ptr @llvm.threadlocal.address.p0(ptr @i)

; CHECK-LABEL: @second_tls_access(
; CHECK-NEXT: call ptr @__next_tls_var_location_i()
; CHECK-NOT: call ptr @llvm.threadlocal.address.p0(ptr @i)

; CHECK-LABEL: define ptr @__next_tls_var_location_i(
; CHECK-NEXT: entry:
; CHECK-NEXT: call ptr @llvm.threadlocal.address.p0(ptr @i)
; CHECK-NEXT: ret ptr
