; RUN: llc -mtriple x86_64-pc-windows-msvc < %s | FileCheck %s

define void @f() {
  ret void
}

@ptr = constant void ()* @f, section ".CRT$XLB", align 8
; CHECK:  .section  .CRT$XLB,"dr"

@weak_array = weak_odr unnamed_addr constant [1 x i8*] [i8* bitcast (void ()* @f to i8*)]
; CHECK:  .section  .rdata,"dr",discard,weak_array
