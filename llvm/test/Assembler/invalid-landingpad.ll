; RUN: not llvm-as --disable-output %s 2>&1 | FileCheck -DFILE=%s %s


define void @test(i32 %in) personality ptr null {
; CHECK: [[FILE]]:[[@LINE+1]]:24: error: 'filter' clause has an invalid type
  landingpad {} filter i32 %in
}
