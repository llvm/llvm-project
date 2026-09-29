; All of the functions in this module must end up
; in the same partition.

; RUN: llvm-split -j=2 -preserve-locals -o %t %s
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK1 %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=CHECK0 %s

; CHECK0: declare dso_local i32 @foo
; CHECK0: declare dso_local i32 @baz
; CHECK0: declare i32 @bar
; CHECK0: declare i32 @bar2

; CHECK1: @bla
; CHECK1: @bla2
; CHECK1: define internal i32 @foo
; CHECK1: define internal i32 @baz
; CHECK1: define i32 @bar
; CHECK1: define i32 @bar2

%struct.anon = type { i64, i64 }

@bla = internal global %struct.anon { i64 1, i64 2 }, align 8
@bla2 = internal global %struct.anon { i64 1, i64 2 }, align 8

define internal i32 @foo() {
entry:
  store i64 5, ptr @bla, align 8
  store i32 -1, ptr getelementptr inbounds (%struct.anon, ptr @bla2, i32 0, i32 1), align 8
  ret i32 0
}

define internal i32 @baz() {
entry:
  store i64 5, ptr @bla, align 8
  store i32 -1, ptr getelementptr inbounds (%struct.anon, ptr @bla2, i32 0, i32 1), align 8
  ret i32 0
}

define i32 @bar() {
  %call = call i32 @foo()
  ret i32 0
}

define i32 @bar2() {
  %call = call i32 @baz()
  ret i32 0
}

