; RUN: llc -mtriple=aarch64 -mattr=+reserve-x18 -fast-isel=0 -global-isel=false \
; RUN:     -stop-after=finalize-isel < %s | FileCheck %s

define i64 @read_reserved_x18() {
; CHECK-LABEL: name: read_reserved_x18
; CHECK: COPY $x18
; CHECK-NOT: READ_REGISTER_GPR64
entry:
  %0 = call i64 @llvm.read_volatile_register.i64(metadata !0)
  ret i64 %0
}

declare i64 @llvm.read_volatile_register.i64(metadata)

!0 = !{!"x18"}
