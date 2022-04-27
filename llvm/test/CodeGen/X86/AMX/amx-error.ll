; RUN: not llc < %s -mtriple=x86_64-unknown-unknown -mattr=+amx-tile -o /dev/null 2>&1 | FileCheck %s

@row = dso_local global i16 8, align 2
@col = dso_local global i16 8, align 2

define dso_local void @add() {
entry:
  ; CHECK: Failed to config tile register
  %t0 = load i16, ptr @row, align 2
  %t1 = call x86_amx @llvm.x86.tilezero.internal(i16 %t0, i16 64)
  %t2 = load i16, ptr @col, align 2
  %t3 = call x86_amx @llvm.x86.tilezero.internal(i16 16, i16 %t2)
  ret void
}

declare x86_amx @llvm.x86.tilezero.internal(i16, i16)
