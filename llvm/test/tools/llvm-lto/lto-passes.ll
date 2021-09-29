; Test that -debug-pass-manager works in LTOCodeGenerator. The actual list of
; passes that is run during LTO is tested in:
;    llvm/test/Other/new-pm-lto-defaults.ll
;
; RUN: llvm-as < %s > %t.bc
; RUN: llvm-lto %t.bc -O0 --debug-pass-manager 2>&1 | FileCheck %s
; CHECK: Running pass: WholeProgramDevirtPass

define i32 @main() {
entry:
  ret i32 42
}
