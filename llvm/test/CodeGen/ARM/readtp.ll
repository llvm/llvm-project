; RUN: llc -mtriple=armeb-linux-gnueabihf -O2 -mattr=+read-tp-tpidrurw %s -o - | FileCheck %s -check-prefix=CHECK-TPIDRURW
; RUN: llc -mtriple=armeb-linux-gnueabihf -O2 -mattr=+read-tp-tpidruro %s -o - | FileCheck %s -check-prefix=CHECK-TPIDRURO
; RUN: llc -mtriple=armeb-linux-gnueabihf -O2 -mattr=+read-tp-tpidrprw %s -o - | FileCheck %s -check-prefix=CHECK-TPIDRPRW
; RUN: llc -mtriple=armeb-linux-gnueabihf -O2 %s -o - | FileCheck %s -check-prefix=CHECK-SOFT
; RUN: llc -mtriple=thumbv7-linux-gnueabihf -O2 -mattr=+read-tp-tpidrurw %s -o - | FileCheck %s -check-prefix=CHECK-TPIDRURW
; RUN: llc -mtriple=thumbv7-linux-gnueabihf -O2 -mattr=+read-tp-tpidruro %s -o - | FileCheck %s -check-prefix=CHECK-TPIDRURO
; RUN: llc -mtriple=thumbv7-linux-gnueabihf -O2 -mattr=+read-tp-tpidrprw %s -o - | FileCheck %s -check-prefix=CHECK-TPIDRPRW
; RUN: llc -mtriple=thumbv7-linux-gnueabihf -O2 %s -o - | FileCheck %s -check-prefix=CHECK-SOFT


; __thread int counter;
;  void foo() {
;    counter = 5;
;  }


@counter = thread_local local_unnamed_addr global i32 0, align 4

define void @foo() local_unnamed_addr #0 {
entry:
  store i32 5, ptr @counter, align 4
  ret void
}


; CHECK-LABEL: foo:
; CHECK-TPIDRURW:     mrc	p15, #0, {{r[0-9]+}}, c13, c0, #2
; CHECK-TPIDRURO:     mrc	p15, #0, {{r[0-9]+}}, c13, c0, #3
; CHECK-TPIDRPRW:     mrc	p15, #0, {{r[0-9]+}}, c13, c0, #4
; CHECK-SOFT:         bl	__aeabi_read_tp
