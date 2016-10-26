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

; CHECK: Alias sets for function 'test2':
; CHECK: Alias Set Tracker: 2 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %a, 1)
; CHECK-NOT:  2 Unknown instructions
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %b, 1)
; CHECK-NOT:  2 Unknown instructions
define void @test2(i8* %ptr) {
entry:
  %a = alloca i8, align 1
  call void @llvm.lifetime.start(i64 1, i8* %a)
  %b = alloca i8, align 1
  call void @llvm.lifetime.start(i64 1, i8* %b)
  store i8 1, i8* %a, align 1
  call void @llvm.lifetime.end(i64 1, i8* %a)
  store i8 1, i8* %b, align 1
  call void @llvm.lifetime.end(i64 1, i8* %b)
  ret void
}

; CHECK: Alias sets for function 'test3':
; CHECK: Alias Set Tracker: 1 alias sets for 1 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %ptr, 1)
; CHECK-NOT:  2 Unknown instructions
define void @test3(i8* %ptr) {
  store i8 5, i8* %ptr
  %i = call {}* @llvm.invariant.start.p0i8(i64 1, i8* %ptr)
  call void @llvm.invariant.end.p0i8({}* %i, i64 1, i8* %ptr)
  store i8 6, i8* %ptr
  ret void
}

declare void @llvm.assume(i1)
declare void @llvm.lifetime.start(i64, i8* nocapture)
declare void @llvm.lifetime.end(i64, i8* nocapture)
declare {}* @llvm.invariant.start.p0i8(i64, i8* nocapture) nounwind readonly
declare void @llvm.invariant.end.p0i8({}*, i64, i8* nocapture) nounwind
