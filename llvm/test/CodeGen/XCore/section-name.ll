; RUN: not llc < %s -march=xcore 2>&1 | FileCheck %s

@bar = internal global i32 zeroinitializer

define void @".dp.bss"() {
  ret void
}

; CHECK: LLVM ERROR: invalid symbol redefinition
