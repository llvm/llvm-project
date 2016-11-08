; RUN: opt -basicaa -print-alias-sets -S -o - < %s 2>&1 | FileCheck %s

; CHECK: Alias sets for function 'test1':
; CHECK: Alias Set Tracker: 2 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %a, 1)
; CHECK-NOT: 1 Unknown instruction
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %b, 1)
define void @test1(i32 %c) {
entry:
  %a = alloca i8, align 1
  %b = alloca i8, align 1
  store i8 1, i8* %a, align 1
  %cond1 = icmp ne i32 %c, 0
  call void @llvm.assume(i1 %cond1)
  store i8 1, i8* %b, align 1
  ret void
}

declare void @llvm.assume(i1)
