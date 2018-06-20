; RUN: not llc -o /dev/null %s 2>&1 | FileCheck %s
target triple = "x86_64--"

@a = global i32 0, align 4

; CHECK: error: couldn't allocate input reg for constraint 'x'
define i32 @main() {
entry:
  %0 = load i32, i32* @a, align 4
  %cmp = icmp ne i32 %0, 0
  %1 = call { i32, i32 } asm "", "={ax},={dx},x,~{dirflag},~{fpsr},~{flags}"(i1 %cmp)
  %asmresult = extractvalue { i32, i32 } %1, 0
  %asmresult1 = extractvalue { i32, i32 } %1, 1
  store i32 %asmresult, i32* @a, align 4
  store i32 %asmresult1, i32* @a, align 4
  ret i32 0
}
