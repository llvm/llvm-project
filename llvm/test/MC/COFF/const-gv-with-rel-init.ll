; RUN: llc -mtriple x86_64-pc-windows-msvc < %s | FileCheck %s

define void @f() {
  ret void
}

@ptr = constant ptr @f, section ".CRT$XLB", align 8
; CHECK:  .section  .CRT$XLB,"dr"

@weak_array = weak_odr unnamed_addr constant [1 x ptr] [ptr @f]
; CHECK:  .section  .rdata,"dr"
