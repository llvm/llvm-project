; RUN: llvm-as -o %t.bc %s
; RUN: llvm-lto -exported-symbol=foo -exported-symbol=bar -j2 -o %t.o %t.bc
; RUN: llvm-nm %t.o.0 | FileCheck --check-prefix=CHECK0 %s
; RUN: llvm-nm %t.o.1 | FileCheck --check-prefix=CHECK1 %s

; FIXME: Investigate test failures on these architectures.
; UNSUPPORTED: target={{(mips|mipsel)-.*}}, target=aarch64{{.*}}, target=powerpc64{{.*}}

target triple = "x86_64-unknown-linux-gnu"

; CHECK0-NOT: bar
; CHECK0: T foo
; CHECK0-NOT: bar
define void @foo() mustprogress {
  call void @bar()
  ret void
}

; CHECK1-NOT: foo
; CHECK1: T bar
; CHECK1-NOT: foo
define void @bar() mustprogress {
  call void @foo()
  ret void
}
