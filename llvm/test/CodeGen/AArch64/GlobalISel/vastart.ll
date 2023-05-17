; RUN: llc -O0 -stop-after=irtranslator -global-isel -verify-machineinstrs %s -o -  -mtriple=aarch64-apple-ios7.0 | FileCheck --check-prefix=CHECK --check-prefix=CHECK-IOS %s
; RUN: llc -O0 -stop-after=irtranslator -global-isel -verify-machineinstrs %s -o -  -mtriple=aarch64-linux-gnu | FileCheck --check-prefix=CHECK --check-prefix=CHECK-LINUX %s


declare void @llvm.va_start(ptr)
define void @test_va_start(ptr %list) {
; CHECK-LABEL: name: test_va_start
; CHECK: [[LIST:%[0-9]+]]:_(p0) = COPY $x0
; CHECK-IOS: G_VASTART [[LIST]](p0) :: (store (s64) into %ir.list, align 1)
; CHECK-LINUX: G_VASTART [[LIST]](p0) :: (store (s256) into %ir.list, align 1)
  call void @llvm.va_start(ptr %list)
  ret void
}
