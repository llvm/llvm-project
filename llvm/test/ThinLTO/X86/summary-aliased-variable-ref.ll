; RUN: opt -module-summary %s -o %t.bc
; RUN: llvm-dis < %t.bc | FileCheck %s

; CHECK: name: "set"{{.*}}notEligibleToImport: 1

@g_real = global i16 7
@g = alias i16, ptr @g_real

define void @set(i16 %a) {
  store i16 %a, ptr @g_real
  ret void
}
