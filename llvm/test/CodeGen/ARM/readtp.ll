; RUN: llc -mtriple=armeb-linux-gnueabihf -O2 -mattr=+read-tp-hard %s -o - | FileCheck %s -check-prefix=CHECK-HARD
; RUN: llc -mtriple=armeb-linux-gnueabihf -O2 %s -o - | FileCheck %s -check-prefix=CHECK-SOFT


; __thread int counter;
;  void foo() {
;    counter = 5;
;  }


@counter = thread_local local_unnamed_addr global i32 0, align 4

define void @foo() local_unnamed_addr #0 {
entry:
  store i32 5, i32* @counter, align 4
  ret void
}


; CHECK-LABEL: foo:
; CHECK-HARD:    mrc	p15, #0, {{r[0-9]+}}, c13, c0, #3
; CHECK-SOFT:    bl	__aeabi_read_tp
