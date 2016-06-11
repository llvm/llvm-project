; RUN: llc -enable-ipra -print-regusage -o /dev/null 2>&1 < %s | FileCheck %s

target triple = "x86_64-unknown-unknown"
declare void @bar1()
define preserve_allcc void @foo()#0 {
; CHECK: foo Clobbered Registers: EFLAGS YMM0 YMM1 YMM2 YMM3 YMM4 YMM5 YMM6 YMM7 YMM8 YMM9 YMM10 YMM11 YMM12 YMM13 YMM14 YMM15 ZMM0 ZMM1 ZMM2 ZMM3 ZMM4 ZMM5 ZMM6 ZMM7 ZMM8 ZMM9 ZMM10 ZMM11 ZMM12 ZMM13 ZMM14 ZMM15
  call void @bar1()
  call void @bar2()
  ret void
}
declare void @bar2()
attributes #0 = {nounwind}
