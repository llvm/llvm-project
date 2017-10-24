; RUN: llc -O0 -stop-after=irtranslator -global-isel -verify-machineinstrs %s -o -  -mtriple=aarch64-apple-ios7.0 | FileCheck --check-prefix=CHECK --check-prefix=CHECK-IOS %s
; RUN: llc -O0 -stop-after=irtranslator -global-isel -verify-machineinstrs %s -o -  -mtriple=aarch64-linux-gnu | FileCheck --check-prefix=CHECK --check-prefix=CHECK-LINUX %s


declare void @llvm.va_start(i8*)
define void @test_va_start(i8* %list) {
; CHECK-LABEL: name: test_va_start
; CHECK: [[LIST:%[0-9]+]]:_(p0) = COPY %x0
; CHECK-IOS: G_VASTART [[LIST]](p0) :: (store 8 into %ir.list, align 0)
; CHECK-LINUX: G_VASTART [[LIST]](p0) :: (store 32 into %ir.list, align 0)
  call void @llvm.va_start(i8* %list)
  ret void
}
