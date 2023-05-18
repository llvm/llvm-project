; RUN: opt -passes='no-op-module' -debug-pass-manager -thinlto-bc -thin-link-bitcode-file=%t2 -o %t %s 2>&1 | FileCheck %s --check-prefix=DEBUG_PM
; RUN: llvm-bcanalyzer -dump %t2 | FileCheck %s --check-prefix=BITCODE

; DEBUG_PM: ThinLTOBitcodeWriterPass
; BITCODE: Foo

define void @Foo() {
  ret void
}
