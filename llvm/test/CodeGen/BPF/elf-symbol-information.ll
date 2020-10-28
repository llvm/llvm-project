; RUN: llc -march=bpfel -filetype=obj -o - %s | llvm-readelf --symbols - | FileCheck %s

; CHECK: 0 FILE    LOCAL  DEFAULT  ABS elf-symbol-information.ll
; CHECK: 8 FUNC    GLOBAL DEFAULT    2 test_func
define void @test_func() {
entry:
  ret void
}
