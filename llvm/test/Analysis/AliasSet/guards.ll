; RUN: opt -aa-pipeline=basic-aa -passes=print-alias-sets -S -o - < %s 2>&1 | FileCheck %s
declare void @llvm.experimental.guard(i1, ...)

; CHECK: Alias sets for function 'test0':
; CHECK: Alias Set Tracker: 3 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Ref       Pointers: (i8* %a, LocationSize::precise(1))
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] may alias, Ref
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Ref       Pointers: (i8* %b, LocationSize::precise(1))
define void @test0(i1 %cond_b) {
entry:
  %a = alloca i8, align 1
  %b = alloca i8, align 1
  %0 = load i8, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  %1 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test1':
; CHECK: Alias Set Tracker: 3 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Ref       Pointers: (i8* %a, LocationSize::precise(1))
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] may alias, Ref
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %b, LocationSize::precise(1))
define void @test1(i1 %cond_b) {
entry:
  %a = alloca i8, align 1
  %b = alloca i8, align 1
  %0 = load i8, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test2':
; CHECK: Alias Set Tracker: 3 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %a, LocationSize::precise(1))
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] may alias, Ref
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Ref       Pointers: (i8* %b, LocationSize::precise(1))
define void @test2(i1 %cond_b) {
entry:
  %a = alloca i8, align 1
  %b = alloca i8, align 1
  store i8 0, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  %0 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test3':
; CHECK: Alias Set Tracker: 3 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %a, LocationSize::precise(1))
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] may alias, Ref
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %b, LocationSize::precise(1))
define void @test3(i1 %cond_b) {
entry:
  %a = alloca i8, align 1
  %b = alloca i8, align 1
  store i8 0, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test4':
; CHECK: Alias Set Tracker: 3 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] may alias, Ref
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Ref       Pointers: (i8* %a, LocationSize::precise(1))
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Ref       Pointers: (i8* %b, LocationSize::precise(1))
define void @test4(i1 %cond_a) {
entry:
  %a = alloca i8, align 1
  %b = alloca i8, align 1
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  %0 = load i8, i8* %a
  %1 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test5':
; CHECK: Alias Set Tracker: 3 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] may alias, Ref
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Ref       Pointers: (i8* %a, LocationSize::precise(1))
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %b, LocationSize::precise(1))
define void @test5(i1 %cond_a) {
entry:
  %a = alloca i8, align 1
  %b = alloca i8, align 1
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  %0 = load i8, i8* %a
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test6':
; CHECK: Alias Set Tracker: 3 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] may alias, Ref
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %a, LocationSize::precise(1))
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Ref       Pointers: (i8* %b, LocationSize::precise(1))
define void @test6(i1 %cond_a) {
entry:
  %a = alloca i8, align 1
  %b = alloca i8, align 1
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  store i8 0, i8* %a
  %0 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test7':
; CHECK: Alias Set Tracker: 3 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] may alias, Ref
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %a, LocationSize::precise(1))
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %b, LocationSize::precise(1))
define void @test7(i1 %cond_a) {
entry:
  %a = alloca i8, align 1
  %b = alloca i8, align 1
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  store i8 0, i8* %a
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test8':
; CHECK: Alias Set Tracker: 3 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] may alias, Ref
; CHECK:     2 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ],   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Ref       Pointers: (i8* %a, LocationSize::precise(1))
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Ref       Pointers: (i8* %b, LocationSize::precise(1))
define void @test8(i1 %cond_a, i1 %cond_b) {
entry:
  %a = alloca i8, align 1
  %b = alloca i8, align 1
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  %0 = load i8, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  %1 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test9':
; CHECK: Alias Set Tracker: 3 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] may alias, Ref
; CHECK:     2 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ],   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Ref       Pointers: (i8* %a, LocationSize::precise(1))
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %b, LocationSize::precise(1))
define void @test9(i1 %cond_a, i1 %cond_b) {
entry:
  %a = alloca i8, align 1
  %b = alloca i8, align 1
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  %0 = load i8, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test10':
; CHECK: Alias Set Tracker: 3 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] may alias, Ref
; CHECK:     2 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ],   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %a, LocationSize::precise(1))
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Ref       Pointers: (i8* %b, LocationSize::precise(1))
define void @test10(i1 %cond_a, i1 %cond_b) {
entry:
  %a = alloca i8, align 1
  %b = alloca i8, align 1
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  store i8 0, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  %0 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test11':
; CHECK: Alias Set Tracker: 3 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] may alias, Ref
; CHECK:     2 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ],   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %a, LocationSize::precise(1))
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %b, LocationSize::precise(1))
define void @test11(i1 %cond_a, i1 %cond_b) {
entry:
  %a = alloca i8, align 1
  %b = alloca i8, align 1
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  store i8 0, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test12':
; CHECK: Alias Set Tracker: 2 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Ref       Pointers: (i8* %a, LocationSize::precise(1))
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 2] may alias, Ref       Pointers: (i8* %b, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
define void @test12(i8* %b, i1 %cond_b) {
entry:
  %a = alloca i8, align 1
  %0 = load i8, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  %1 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test13':
; CHECK: Alias Set Tracker: 2 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Ref       Pointers: (i8* %a, LocationSize::precise(1))
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 2] may alias, Mod/Ref   Pointers: (i8* %b, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
define void @test13(i8* %b, i1 %cond_b) {
entry:
  %a = alloca i8, align 1
  %0 = load i8, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test14':
; CHECK: Alias Set Tracker: 2 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %a, LocationSize::precise(1))
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 2] may alias, Ref       Pointers: (i8* %b, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
define void @test14(i8* %b, i1 %cond_b) {
entry:
  %a = alloca i8, align 1
  store i8 0, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  %0 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test15':
; CHECK: Alias Set Tracker: 2 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %a, LocationSize::precise(1))
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 2] may alias, Mod/Ref   Pointers: (i8* %b, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
define void @test15(i8* %b, i1 %cond_b) {
entry:
  %a = alloca i8, align 1
  store i8 0, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test16':
; CHECK: Alias Set Tracker: 2 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 2] may alias, Ref       Pointers: (i8* %b, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Ref       Pointers: (i8* %a, LocationSize::precise(1))
define void @test16(i1 %cond_a, i8* %b) {
entry:
  %a = alloca i8, align 1
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  %0 = load i8, i8* %a
  %1 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test17':
; CHECK: Alias Set Tracker: 2 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 2] may alias, Mod/Ref   Pointers: (i8* %b, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Ref       Pointers: (i8* %a, LocationSize::precise(1))
define void @test17(i1 %cond_a, i8* %b) {
entry:
  %a = alloca i8, align 1
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  %0 = load i8, i8* %a
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test18':
; CHECK: Alias Set Tracker: 2 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 2] may alias, Ref       Pointers: (i8* %b, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %a, LocationSize::precise(1))
define void @test18(i1 %cond_a, i8* %b) {
entry:
  %a = alloca i8, align 1
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  store i8 0, i8* %a
  %0 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test19':
; CHECK: Alias Set Tracker: 2 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 2] may alias, Mod/Ref   Pointers: (i8* %b, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %a, LocationSize::precise(1))
define void @test19(i1 %cond_a, i8* %b) {
entry:
  %a = alloca i8, align 1
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  store i8 0, i8* %a
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test20':
; CHECK: Alias Set Tracker: 2 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 2] may alias, Ref       Pointers: (i8* %b, LocationSize::precise(1))
; CHECK:     2 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ],   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Ref       Pointers: (i8* %a, LocationSize::precise(1))
define void @test20(i1 %cond_a, i8* %b, i1 %cond_b) {
entry:
  %a = alloca i8, align 1
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  %0 = load i8, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  %1 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test21':
; CHECK: Alias Set Tracker: 2 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 2] may alias, Mod/Ref   Pointers: (i8* %b, LocationSize::precise(1))
; CHECK:     2 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ],   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Ref       Pointers: (i8* %a, LocationSize::precise(1))
define void @test21(i1 %cond_a, i8* %b, i1 %cond_b) {
entry:
  %a = alloca i8, align 1
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  %0 = load i8, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test22':
; CHECK: Alias Set Tracker: 2 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 2] may alias, Ref       Pointers: (i8* %b, LocationSize::precise(1))
; CHECK:     2 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ],   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %a, LocationSize::precise(1))
define void @test22(i1 %cond_a, i8* %b, i1 %cond_b) {
entry:
  %a = alloca i8, align 1
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  store i8 0, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  %0 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test23':
; CHECK: Alias Set Tracker: 2 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 2] may alias, Mod/Ref   Pointers: (i8* %b, LocationSize::precise(1))
; CHECK:     2 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ],   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %a, LocationSize::precise(1))
define void @test23(i1 %cond_a, i8* %b, i1 %cond_b) {
entry:
  %a = alloca i8, align 1
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  store i8 0, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test24':
; CHECK: Alias Set Tracker: 2 alias sets for 3 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 3] may alias, Ref       Pointers: (i8** %ptr_b, LocationSize::precise(8)), (i8* %b, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Ref       Pointers: (i8* %a, LocationSize::precise(1))
define void @test24(i8** %ptr_b, i1 %cond_b) {
entry:
  %a = alloca i8, align 1
  %b = load i8*, i8** %ptr_b
  %0 = load i8, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  %1 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test25':
; CHECK: Alias Set Tracker: 2 alias sets for 3 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 3] may alias, Mod/Ref   Pointers: (i8** %ptr_b, LocationSize::precise(8)), (i8* %b, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Ref       Pointers: (i8* %a, LocationSize::precise(1))
define void @test25(i8** %ptr_b, i1 %cond_b) {
entry:
  %a = alloca i8, align 1
  %b = load i8*, i8** %ptr_b
  %0 = load i8, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test26':
; CHECK: Alias Set Tracker: 2 alias sets for 3 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 3] may alias, Ref       Pointers: (i8** %ptr_b, LocationSize::precise(8)), (i8* %b, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %a, LocationSize::precise(1))
define void @test26(i8** %ptr_b, i1 %cond_b) {
entry:
  %a = alloca i8, align 1
  %b = load i8*, i8** %ptr_b
  store i8 0, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  %0 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test27':
; CHECK: Alias Set Tracker: 2 alias sets for 3 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 3] may alias, Mod/Ref   Pointers: (i8** %ptr_b, LocationSize::precise(8)), (i8* %b, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %a, LocationSize::precise(1))
define void @test27(i8** %ptr_b, i1 %cond_b) {
entry:
  %a = alloca i8, align 1
  %b = load i8*, i8** %ptr_b
  store i8 0, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test28':
; CHECK: Alias Set Tracker: 2 alias sets for 3 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 3] may alias, Ref       Pointers: (i8** %ptr_b, LocationSize::precise(8)), (i8* %b, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Ref       Pointers: (i8* %a, LocationSize::precise(1))
define void @test28(i1 %cond_a, i8** %ptr_b) {
entry:
  %a = alloca i8, align 1
  %b = load i8*, i8** %ptr_b
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  %0 = load i8, i8* %a
  %1 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test29':
; CHECK: Alias Set Tracker: 2 alias sets for 3 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 3] may alias, Mod/Ref   Pointers: (i8** %ptr_b, LocationSize::precise(8)), (i8* %b, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Ref       Pointers: (i8* %a, LocationSize::precise(1))
define void @test29(i1 %cond_a, i8** %ptr_b) {
entry:
  %a = alloca i8, align 1
  %b = load i8*, i8** %ptr_b
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  %0 = load i8, i8* %a
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test30':
; CHECK: Alias Set Tracker: 2 alias sets for 3 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 3] may alias, Ref       Pointers: (i8** %ptr_b, LocationSize::precise(8)), (i8* %b, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %a, LocationSize::precise(1))
define void @test30(i1 %cond_a, i8** %ptr_b) {
entry:
  %a = alloca i8, align 1
  %b = load i8*, i8** %ptr_b
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  store i8 0, i8* %a
  %0 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test31':
; CHECK: Alias Set Tracker: 2 alias sets for 3 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 3] may alias, Mod/Ref   Pointers: (i8** %ptr_b, LocationSize::precise(8)), (i8* %b, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %a, LocationSize::precise(1))
define void @test31(i1 %cond_a, i8** %ptr_b) {
entry:
  %a = alloca i8, align 1
  %b = load i8*, i8** %ptr_b
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  store i8 0, i8* %a
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test32':
; CHECK: Alias Set Tracker: 2 alias sets for 3 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 3] may alias, Ref       Pointers: (i8** %ptr_b, LocationSize::precise(8)), (i8* %b, LocationSize::precise(1))
; CHECK:     2 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ],   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Ref       Pointers: (i8* %a, LocationSize::precise(1))
define void @test32(i1 %cond_a, i8** %ptr_b, i1 %cond_b) {
entry:
  %a = alloca i8, align 1
  %b = load i8*, i8** %ptr_b
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  %0 = load i8, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  %1 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test33':
; CHECK: Alias Set Tracker: 2 alias sets for 3 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 3] may alias, Mod/Ref   Pointers: (i8** %ptr_b, LocationSize::precise(8)), (i8* %b, LocationSize::precise(1))
; CHECK:     2 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ],   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Ref       Pointers: (i8* %a, LocationSize::precise(1))
define void @test33(i1 %cond_a, i8** %ptr_b, i1 %cond_b) {
entry:
  %a = alloca i8, align 1
  %b = load i8*, i8** %ptr_b
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  %0 = load i8, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test34':
; CHECK: Alias Set Tracker: 2 alias sets for 3 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 3] may alias, Ref       Pointers: (i8** %ptr_b, LocationSize::precise(8)), (i8* %b, LocationSize::precise(1))
; CHECK:     2 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ],   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %a, LocationSize::precise(1))
define void @test34(i1 %cond_a, i8** %ptr_b, i1 %cond_b) {
entry:
  %a = alloca i8, align 1
  %b = load i8*, i8** %ptr_b
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  store i8 0, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  %0 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test35':
; CHECK: Alias Set Tracker: 2 alias sets for 3 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 3] may alias, Mod/Ref   Pointers: (i8** %ptr_b, LocationSize::precise(8)), (i8* %b, LocationSize::precise(1))
; CHECK:     2 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ],   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %a, LocationSize::precise(1))
define void @test35(i1 %cond_a, i8** %ptr_b, i1 %cond_b) {
entry:
  %a = alloca i8, align 1
  %b = load i8*, i8** %ptr_b
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  store i8 0, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test36':
; CHECK: Alias Set Tracker: 2 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 2] may alias, Ref       Pointers: (i8* %a, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Ref       Pointers: (i8* %b, LocationSize::precise(1))
define void @test36(i8* %a, i1 %cond_b) {
entry:
  %b = alloca i8, align 1
  %0 = load i8, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  %1 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test37':
; CHECK: Alias Set Tracker: 2 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 2] may alias, Ref       Pointers: (i8* %a, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %b, LocationSize::precise(1))
define void @test37(i8* %a, i1 %cond_b) {
entry:
  %b = alloca i8, align 1
  %0 = load i8, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test38':
; CHECK: Alias Set Tracker: 2 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 2] may alias, Mod/Ref   Pointers: (i8* %a, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Ref       Pointers: (i8* %b, LocationSize::precise(1))
define void @test38(i8* %a, i1 %cond_b) {
entry:
  %b = alloca i8, align 1
  store i8 0, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  %0 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test39':
; CHECK: Alias Set Tracker: 2 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 2] may alias, Mod/Ref   Pointers: (i8* %a, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %b, LocationSize::precise(1))
define void @test39(i8* %a, i1 %cond_b) {
entry:
  %b = alloca i8, align 1
  store i8 0, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test40':
; CHECK: Alias Set Tracker: 2 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 2] may alias, Ref       Pointers: (i8* %a, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Ref       Pointers: (i8* %b, LocationSize::precise(1))
define void @test40(i8* %a, i1 %cond_a) {
entry:
  %b = alloca i8, align 1
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  %0 = load i8, i8* %a
  %1 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test41':
; CHECK: Alias Set Tracker: 2 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 2] may alias, Ref       Pointers: (i8* %a, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %b, LocationSize::precise(1))
define void @test41(i8* %a, i1 %cond_a) {
entry:
  %b = alloca i8, align 1
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  %0 = load i8, i8* %a
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test42':
; CHECK: Alias Set Tracker: 2 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 2] may alias, Mod/Ref   Pointers: (i8* %a, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Ref       Pointers: (i8* %b, LocationSize::precise(1))
define void @test42(i8* %a, i1 %cond_a) {
entry:
  %b = alloca i8, align 1
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  store i8 0, i8* %a
  %0 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test43':
; CHECK: Alias Set Tracker: 2 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 2] may alias, Mod/Ref   Pointers: (i8* %a, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %b, LocationSize::precise(1))
define void @test43(i8* %a, i1 %cond_a) {
entry:
  %b = alloca i8, align 1
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  store i8 0, i8* %a
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test44':
; CHECK: Alias Set Tracker: 2 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 2] may alias, Ref       Pointers: (i8* %a, LocationSize::precise(1))
; CHECK:     2 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ],   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Ref       Pointers: (i8* %b, LocationSize::precise(1))
define void @test44(i8* %a, i1 %cond_a, i1 %cond_b) {
entry:
  %b = alloca i8, align 1
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  %0 = load i8, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  %1 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test45':
; CHECK: Alias Set Tracker: 2 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 2] may alias, Ref       Pointers: (i8* %a, LocationSize::precise(1))
; CHECK:     2 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ],   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %b, LocationSize::precise(1))
define void @test45(i8* %a, i1 %cond_a, i1 %cond_b) {
entry:
  %b = alloca i8, align 1
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  %0 = load i8, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test46':
; CHECK: Alias Set Tracker: 2 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 2] may alias, Mod/Ref   Pointers: (i8* %a, LocationSize::precise(1))
; CHECK:     2 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ],   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Ref       Pointers: (i8* %b, LocationSize::precise(1))
define void @test46(i8* %a, i1 %cond_a, i1 %cond_b) {
entry:
  %b = alloca i8, align 1
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  store i8 0, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  %0 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test47':
; CHECK: Alias Set Tracker: 2 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 2] may alias, Mod/Ref   Pointers: (i8* %a, LocationSize::precise(1))
; CHECK:     2 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ],   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %b, LocationSize::precise(1))
define void @test47(i8* %a, i1 %cond_a, i1 %cond_b) {
entry:
  %b = alloca i8, align 1
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  store i8 0, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test48':
; CHECK: Alias Set Tracker: 1 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 3] may alias, Ref       Pointers: (i8* %a, LocationSize::precise(1)), (i8* %b, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
define void @test48(i8* %a, i8* %b, i1 %cond_b) {
entry:
  %0 = load i8, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  %1 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test49':
; CHECK: Alias Set Tracker: 1 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 3] may alias, Mod/Ref   Pointers: (i8* %a, LocationSize::precise(1)), (i8* %b, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
define void @test49(i8* %a, i8* %b, i1 %cond_b) {
entry:
  %0 = load i8, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test50':
; CHECK: Alias Set Tracker: 1 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 3] may alias, Mod/Ref   Pointers: (i8* %a, LocationSize::precise(1)), (i8* %b, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
define void @test50(i8* %a, i8* %b, i1 %cond_b) {
entry:
  store i8 0, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  %0 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test51':
; CHECK: Alias Set Tracker: 1 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 3] may alias, Mod/Ref   Pointers: (i8* %a, LocationSize::precise(1)), (i8* %b, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
define void @test51(i8* %a, i8* %b, i1 %cond_b) {
entry:
  store i8 0, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test52':
; CHECK: Alias Set Tracker: 1 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 3] may alias, Ref       Pointers: (i8* %a, LocationSize::precise(1)), (i8* %b, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
define void @test52(i8* %a, i1 %cond_a, i8* %b) {
entry:
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  %0 = load i8, i8* %a
  %1 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test53':
; CHECK: Alias Set Tracker: 1 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 3] may alias, Mod/Ref   Pointers: (i8* %a, LocationSize::precise(1)), (i8* %b, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
define void @test53(i8* %a, i1 %cond_a, i8* %b) {
entry:
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  %0 = load i8, i8* %a
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test54':
; CHECK: Alias Set Tracker: 1 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 3] may alias, Mod/Ref   Pointers: (i8* %a, LocationSize::precise(1)), (i8* %b, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
define void @test54(i8* %a, i1 %cond_a, i8* %b) {
entry:
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  store i8 0, i8* %a
  %0 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test55':
; CHECK: Alias Set Tracker: 1 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 3] may alias, Mod/Ref   Pointers: (i8* %a, LocationSize::precise(1)), (i8* %b, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
define void @test55(i8* %a, i1 %cond_a, i8* %b) {
entry:
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  store i8 0, i8* %a
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test56':
; CHECK: Alias Set Tracker: 1 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 3] may alias, Ref       Pointers: (i8* %a, LocationSize::precise(1)), (i8* %b, LocationSize::precise(1))
; CHECK:     2 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ],   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
define void @test56(i8* %a, i1 %cond_a, i8* %b, i1 %cond_b) {
entry:
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  %0 = load i8, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  %1 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test57':
; CHECK: Alias Set Tracker: 1 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 3] may alias, Mod/Ref   Pointers: (i8* %a, LocationSize::precise(1)), (i8* %b, LocationSize::precise(1))
; CHECK:     2 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ],   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
define void @test57(i8* %a, i1 %cond_a, i8* %b, i1 %cond_b) {
entry:
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  %0 = load i8, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test58':
; CHECK: Alias Set Tracker: 1 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 3] may alias, Mod/Ref   Pointers: (i8* %a, LocationSize::precise(1)), (i8* %b, LocationSize::precise(1))
; CHECK:     2 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ],   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
define void @test58(i8* %a, i1 %cond_a, i8* %b, i1 %cond_b) {
entry:
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  store i8 0, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  %0 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test59':
; CHECK: Alias Set Tracker: 1 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 3] may alias, Mod/Ref   Pointers: (i8* %a, LocationSize::precise(1)), (i8* %b, LocationSize::precise(1))
; CHECK:     2 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ],   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
define void @test59(i8* %a, i1 %cond_a, i8* %b, i1 %cond_b) {
entry:
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  store i8 0, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test60':
; CHECK: Alias Set Tracker: 1 alias sets for 3 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 4] may alias, Ref       Pointers: (i8** %ptr_b, LocationSize::precise(8)), (i8* %a, LocationSize::precise(1)), (i8* %b, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
define void @test60(i8* %a, i8** %ptr_b, i1 %cond_b) {
entry:
  %b = load i8*, i8** %ptr_b
  %0 = load i8, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  %1 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test61':
; CHECK: Alias Set Tracker: 1 alias sets for 3 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 4] may alias, Mod/Ref   Pointers: (i8** %ptr_b, LocationSize::precise(8)), (i8* %a, LocationSize::precise(1)), (i8* %b, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
define void @test61(i8* %a, i8** %ptr_b, i1 %cond_b) {
entry:
  %b = load i8*, i8** %ptr_b
  %0 = load i8, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test62':
; CHECK: Alias Set Tracker: 1 alias sets for 3 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 4] may alias, Mod/Ref   Pointers: (i8** %ptr_b, LocationSize::precise(8)), (i8* %a, LocationSize::precise(1)), (i8* %b, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
define void @test62(i8* %a, i8** %ptr_b, i1 %cond_b) {
entry:
  %b = load i8*, i8** %ptr_b
  store i8 0, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  %0 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test63':
; CHECK: Alias Set Tracker: 1 alias sets for 3 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 4] may alias, Mod/Ref   Pointers: (i8** %ptr_b, LocationSize::precise(8)), (i8* %a, LocationSize::precise(1)), (i8* %b, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
define void @test63(i8* %a, i8** %ptr_b, i1 %cond_b) {
entry:
  %b = load i8*, i8** %ptr_b
  store i8 0, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test64':
; CHECK: Alias Set Tracker: 1 alias sets for 3 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 4] may alias, Ref       Pointers: (i8** %ptr_b, LocationSize::precise(8)), (i8* %a, LocationSize::precise(1)), (i8* %b, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
define void @test64(i8* %a, i1 %cond_a, i8** %ptr_b) {
entry:
  %b = load i8*, i8** %ptr_b
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  %0 = load i8, i8* %a
  %1 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test65':
; CHECK: Alias Set Tracker: 1 alias sets for 3 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 4] may alias, Mod/Ref   Pointers: (i8** %ptr_b, LocationSize::precise(8)), (i8* %a, LocationSize::precise(1)), (i8* %b, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
define void @test65(i8* %a, i1 %cond_a, i8** %ptr_b) {
entry:
  %b = load i8*, i8** %ptr_b
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  %0 = load i8, i8* %a
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test66':
; CHECK: Alias Set Tracker: 1 alias sets for 3 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 4] may alias, Mod/Ref   Pointers: (i8** %ptr_b, LocationSize::precise(8)), (i8* %a, LocationSize::precise(1)), (i8* %b, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
define void @test66(i8* %a, i1 %cond_a, i8** %ptr_b) {
entry:
  %b = load i8*, i8** %ptr_b
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  store i8 0, i8* %a
  %0 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test67':
; CHECK: Alias Set Tracker: 1 alias sets for 3 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 4] may alias, Mod/Ref   Pointers: (i8** %ptr_b, LocationSize::precise(8)), (i8* %a, LocationSize::precise(1)), (i8* %b, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
define void @test67(i8* %a, i1 %cond_a, i8** %ptr_b) {
entry:
  %b = load i8*, i8** %ptr_b
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  store i8 0, i8* %a
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test68':
; CHECK: Alias Set Tracker: 1 alias sets for 3 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 4] may alias, Ref       Pointers: (i8** %ptr_b, LocationSize::precise(8)), (i8* %a, LocationSize::precise(1)), (i8* %b, LocationSize::precise(1))
; CHECK:     2 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ],   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
define void @test68(i8* %a, i1 %cond_a, i8** %ptr_b, i1 %cond_b) {
entry:
  %b = load i8*, i8** %ptr_b
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  %0 = load i8, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  %1 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test69':
; CHECK: Alias Set Tracker: 1 alias sets for 3 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 4] may alias, Mod/Ref   Pointers: (i8** %ptr_b, LocationSize::precise(8)), (i8* %a, LocationSize::precise(1)), (i8* %b, LocationSize::precise(1))
; CHECK:     2 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ],   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
define void @test69(i8* %a, i1 %cond_a, i8** %ptr_b, i1 %cond_b) {
entry:
  %b = load i8*, i8** %ptr_b
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  %0 = load i8, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test70':
; CHECK: Alias Set Tracker: 1 alias sets for 3 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 4] may alias, Mod/Ref   Pointers: (i8** %ptr_b, LocationSize::precise(8)), (i8* %a, LocationSize::precise(1)), (i8* %b, LocationSize::precise(1))
; CHECK:     2 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ],   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
define void @test70(i8* %a, i1 %cond_a, i8** %ptr_b, i1 %cond_b) {
entry:
  %b = load i8*, i8** %ptr_b
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  store i8 0, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  %0 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test71':
; CHECK: Alias Set Tracker: 1 alias sets for 3 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 4] may alias, Mod/Ref   Pointers: (i8** %ptr_b, LocationSize::precise(8)), (i8* %a, LocationSize::precise(1)), (i8* %b, LocationSize::precise(1))
; CHECK:     2 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ],   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
define void @test71(i8* %a, i1 %cond_a, i8** %ptr_b, i1 %cond_b) {
entry:
  %b = load i8*, i8** %ptr_b
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  store i8 0, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test72':
; CHECK: Alias Set Tracker: 2 alias sets for 3 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 3] may alias, Ref       Pointers: (i8** %ptr_a, LocationSize::precise(8)), (i8* %a, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Ref       Pointers: (i8* %b, LocationSize::precise(1))
define void @test72(i8** %ptr_a, i1 %cond_b) {
entry:
  %a = load i8*, i8** %ptr_a
  %b = alloca i8, align 1
  %0 = load i8, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  %1 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test73':
; CHECK: Alias Set Tracker: 2 alias sets for 3 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 3] may alias, Ref       Pointers: (i8** %ptr_a, LocationSize::precise(8)), (i8* %a, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %b, LocationSize::precise(1))
define void @test73(i8** %ptr_a, i1 %cond_b) {
entry:
  %a = load i8*, i8** %ptr_a
  %b = alloca i8, align 1
  %0 = load i8, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test74':
; CHECK: Alias Set Tracker: 2 alias sets for 3 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 3] may alias, Mod/Ref   Pointers: (i8** %ptr_a, LocationSize::precise(8)), (i8* %a, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Ref       Pointers: (i8* %b, LocationSize::precise(1))
define void @test74(i8** %ptr_a, i1 %cond_b) {
entry:
  %a = load i8*, i8** %ptr_a
  %b = alloca i8, align 1
  store i8 0, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  %0 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test75':
; CHECK: Alias Set Tracker: 2 alias sets for 3 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 3] may alias, Mod/Ref   Pointers: (i8** %ptr_a, LocationSize::precise(8)), (i8* %a, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %b, LocationSize::precise(1))
define void @test75(i8** %ptr_a, i1 %cond_b) {
entry:
  %a = load i8*, i8** %ptr_a
  %b = alloca i8, align 1
  store i8 0, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test76':
; CHECK: Alias Set Tracker: 2 alias sets for 3 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 3] may alias, Ref       Pointers: (i8** %ptr_a, LocationSize::precise(8)), (i8* %a, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Ref       Pointers: (i8* %b, LocationSize::precise(1))
define void @test76(i8** %ptr_a, i1 %cond_a) {
entry:
  %a = load i8*, i8** %ptr_a
  %b = alloca i8, align 1
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  %0 = load i8, i8* %a
  %1 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test77':
; CHECK: Alias Set Tracker: 2 alias sets for 3 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 3] may alias, Ref       Pointers: (i8** %ptr_a, LocationSize::precise(8)), (i8* %a, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %b, LocationSize::precise(1))
define void @test77(i8** %ptr_a, i1 %cond_a) {
entry:
  %a = load i8*, i8** %ptr_a
  %b = alloca i8, align 1
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  %0 = load i8, i8* %a
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test78':
; CHECK: Alias Set Tracker: 2 alias sets for 3 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 3] may alias, Mod/Ref   Pointers: (i8** %ptr_a, LocationSize::precise(8)), (i8* %a, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Ref       Pointers: (i8* %b, LocationSize::precise(1))
define void @test78(i8** %ptr_a, i1 %cond_a) {
entry:
  %a = load i8*, i8** %ptr_a
  %b = alloca i8, align 1
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  store i8 0, i8* %a
  %0 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test79':
; CHECK: Alias Set Tracker: 2 alias sets for 3 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 3] may alias, Mod/Ref   Pointers: (i8** %ptr_a, LocationSize::precise(8)), (i8* %a, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %b, LocationSize::precise(1))
define void @test79(i8** %ptr_a, i1 %cond_a) {
entry:
  %a = load i8*, i8** %ptr_a
  %b = alloca i8, align 1
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  store i8 0, i8* %a
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test80':
; CHECK: Alias Set Tracker: 2 alias sets for 3 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 3] may alias, Ref       Pointers: (i8** %ptr_a, LocationSize::precise(8)), (i8* %a, LocationSize::precise(1))
; CHECK:     2 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ],   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Ref       Pointers: (i8* %b, LocationSize::precise(1))
define void @test80(i8** %ptr_a, i1 %cond_a, i1 %cond_b) {
entry:
  %a = load i8*, i8** %ptr_a
  %b = alloca i8, align 1
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  %0 = load i8, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  %1 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test81':
; CHECK: Alias Set Tracker: 2 alias sets for 3 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 3] may alias, Ref       Pointers: (i8** %ptr_a, LocationSize::precise(8)), (i8* %a, LocationSize::precise(1))
; CHECK:     2 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ],   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %b, LocationSize::precise(1))
define void @test81(i8** %ptr_a, i1 %cond_a, i1 %cond_b) {
entry:
  %a = load i8*, i8** %ptr_a
  %b = alloca i8, align 1
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  %0 = load i8, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test82':
; CHECK: Alias Set Tracker: 2 alias sets for 3 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 3] may alias, Mod/Ref   Pointers: (i8** %ptr_a, LocationSize::precise(8)), (i8* %a, LocationSize::precise(1))
; CHECK:     2 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ],   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Ref       Pointers: (i8* %b, LocationSize::precise(1))
define void @test82(i8** %ptr_a, i1 %cond_a, i1 %cond_b) {
entry:
  %a = load i8*, i8** %ptr_a
  %b = alloca i8, align 1
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  store i8 0, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  %0 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test83':
; CHECK: Alias Set Tracker: 2 alias sets for 3 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 3] may alias, Mod/Ref   Pointers: (i8** %ptr_a, LocationSize::precise(8)), (i8* %a, LocationSize::precise(1))
; CHECK:     2 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ],   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %b, LocationSize::precise(1))
define void @test83(i8** %ptr_a, i1 %cond_a, i1 %cond_b) {
entry:
  %a = load i8*, i8** %ptr_a
  %b = alloca i8, align 1
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  store i8 0, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test84':
; CHECK: Alias Set Tracker: 1 alias sets for 3 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 4] may alias, Ref       Pointers: (i8** %ptr_a, LocationSize::precise(8)), (i8* %a, LocationSize::precise(1)), (i8* %b, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
define void @test84(i8** %ptr_a, i8* %b, i1 %cond_b) {
entry:
  %a = load i8*, i8** %ptr_a
  %0 = load i8, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  %1 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test85':
; CHECK: Alias Set Tracker: 1 alias sets for 3 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 4] may alias, Mod/Ref   Pointers: (i8** %ptr_a, LocationSize::precise(8)), (i8* %a, LocationSize::precise(1)), (i8* %b, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
define void @test85(i8** %ptr_a, i8* %b, i1 %cond_b) {
entry:
  %a = load i8*, i8** %ptr_a
  %0 = load i8, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test86':
; CHECK: Alias Set Tracker: 1 alias sets for 3 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 4] may alias, Mod/Ref   Pointers: (i8** %ptr_a, LocationSize::precise(8)), (i8* %a, LocationSize::precise(1)), (i8* %b, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
define void @test86(i8** %ptr_a, i8* %b, i1 %cond_b) {
entry:
  %a = load i8*, i8** %ptr_a
  store i8 0, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  %0 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test87':
; CHECK: Alias Set Tracker: 1 alias sets for 3 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 4] may alias, Mod/Ref   Pointers: (i8** %ptr_a, LocationSize::precise(8)), (i8* %a, LocationSize::precise(1)), (i8* %b, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
define void @test87(i8** %ptr_a, i8* %b, i1 %cond_b) {
entry:
  %a = load i8*, i8** %ptr_a
  store i8 0, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test88':
; CHECK: Alias Set Tracker: 1 alias sets for 3 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 4] may alias, Ref       Pointers: (i8** %ptr_a, LocationSize::precise(8)), (i8* %a, LocationSize::precise(1)), (i8* %b, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
define void @test88(i8** %ptr_a, i1 %cond_a, i8* %b) {
entry:
  %a = load i8*, i8** %ptr_a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  %0 = load i8, i8* %a
  %1 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test89':
; CHECK: Alias Set Tracker: 1 alias sets for 3 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 4] may alias, Mod/Ref   Pointers: (i8** %ptr_a, LocationSize::precise(8)), (i8* %a, LocationSize::precise(1)), (i8* %b, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
define void @test89(i8** %ptr_a, i1 %cond_a, i8* %b) {
entry:
  %a = load i8*, i8** %ptr_a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  %0 = load i8, i8* %a
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test90':
; CHECK: Alias Set Tracker: 1 alias sets for 3 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 4] may alias, Mod/Ref   Pointers: (i8** %ptr_a, LocationSize::precise(8)), (i8* %a, LocationSize::precise(1)), (i8* %b, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
define void @test90(i8** %ptr_a, i1 %cond_a, i8* %b) {
entry:
  %a = load i8*, i8** %ptr_a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  store i8 0, i8* %a
  %0 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test91':
; CHECK: Alias Set Tracker: 1 alias sets for 3 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 4] may alias, Mod/Ref   Pointers: (i8** %ptr_a, LocationSize::precise(8)), (i8* %a, LocationSize::precise(1)), (i8* %b, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
define void @test91(i8** %ptr_a, i1 %cond_a, i8* %b) {
entry:
  %a = load i8*, i8** %ptr_a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  store i8 0, i8* %a
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test92':
; CHECK: Alias Set Tracker: 1 alias sets for 3 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 4] may alias, Ref       Pointers: (i8** %ptr_a, LocationSize::precise(8)), (i8* %a, LocationSize::precise(1)), (i8* %b, LocationSize::precise(1))
; CHECK:     2 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ],   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
define void @test92(i8** %ptr_a, i1 %cond_a, i8* %b, i1 %cond_b) {
entry:
  %a = load i8*, i8** %ptr_a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  %0 = load i8, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  %1 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test93':
; CHECK: Alias Set Tracker: 1 alias sets for 3 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 4] may alias, Mod/Ref   Pointers: (i8** %ptr_a, LocationSize::precise(8)), (i8* %a, LocationSize::precise(1)), (i8* %b, LocationSize::precise(1))
; CHECK:     2 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ],   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
define void @test93(i8** %ptr_a, i1 %cond_a, i8* %b, i1 %cond_b) {
entry:
  %a = load i8*, i8** %ptr_a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  %0 = load i8, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test94':
; CHECK: Alias Set Tracker: 1 alias sets for 3 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 4] may alias, Mod/Ref   Pointers: (i8** %ptr_a, LocationSize::precise(8)), (i8* %a, LocationSize::precise(1)), (i8* %b, LocationSize::precise(1))
; CHECK:     2 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ],   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
define void @test94(i8** %ptr_a, i1 %cond_a, i8* %b, i1 %cond_b) {
entry:
  %a = load i8*, i8** %ptr_a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  store i8 0, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  %0 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test95':
; CHECK: Alias Set Tracker: 1 alias sets for 3 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 4] may alias, Mod/Ref   Pointers: (i8** %ptr_a, LocationSize::precise(8)), (i8* %a, LocationSize::precise(1)), (i8* %b, LocationSize::precise(1))
; CHECK:     2 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ],   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
define void @test95(i8** %ptr_a, i1 %cond_a, i8* %b, i1 %cond_b) {
entry:
  %a = load i8*, i8** %ptr_a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  store i8 0, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test96':
; CHECK: Alias Set Tracker: 1 alias sets for 4 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 5] may alias, Ref       Pointers: (i8** %ptr_a, LocationSize::precise(8)), (i8** %ptr_b, LocationSize::precise(8)), (i8* %a, LocationSize::precise(1)), (i8* %b, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
define void @test96(i8** %ptr_a, i8** %ptr_b, i1 %cond_b) {
entry:
  %a = load i8*, i8** %ptr_a
  %b = load i8*, i8** %ptr_b
  %0 = load i8, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  %1 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test97':
; CHECK: Alias Set Tracker: 1 alias sets for 4 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 5] may alias, Mod/Ref   Pointers: (i8** %ptr_a, LocationSize::precise(8)), (i8** %ptr_b, LocationSize::precise(8)), (i8* %a, LocationSize::precise(1)), (i8* %b, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
define void @test97(i8** %ptr_a, i8** %ptr_b, i1 %cond_b) {
entry:
  %a = load i8*, i8** %ptr_a
  %b = load i8*, i8** %ptr_b
  %0 = load i8, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test98':
; CHECK: Alias Set Tracker: 1 alias sets for 4 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 5] may alias, Mod/Ref   Pointers: (i8** %ptr_a, LocationSize::precise(8)), (i8** %ptr_b, LocationSize::precise(8)), (i8* %a, LocationSize::precise(1)), (i8* %b, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
define void @test98(i8** %ptr_a, i8** %ptr_b, i1 %cond_b) {
entry:
  %a = load i8*, i8** %ptr_a
  %b = load i8*, i8** %ptr_b
  store i8 0, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  %0 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test99':
; CHECK: Alias Set Tracker: 1 alias sets for 4 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 5] may alias, Mod/Ref   Pointers: (i8** %ptr_a, LocationSize::precise(8)), (i8** %ptr_b, LocationSize::precise(8)), (i8* %a, LocationSize::precise(1)), (i8* %b, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
define void @test99(i8** %ptr_a, i8** %ptr_b, i1 %cond_b) {
entry:
  %a = load i8*, i8** %ptr_a
  %b = load i8*, i8** %ptr_b
  store i8 0, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test100':
; CHECK: Alias Set Tracker: 1 alias sets for 4 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 5] may alias, Ref       Pointers: (i8** %ptr_a, LocationSize::precise(8)), (i8** %ptr_b, LocationSize::precise(8)), (i8* %a, LocationSize::precise(1)), (i8* %b, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
define void @test100(i8** %ptr_a, i1 %cond_a, i8** %ptr_b) {
entry:
  %a = load i8*, i8** %ptr_a
  %b = load i8*, i8** %ptr_b
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  %0 = load i8, i8* %a
  %1 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test101':
; CHECK: Alias Set Tracker: 1 alias sets for 4 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 5] may alias, Mod/Ref   Pointers: (i8** %ptr_a, LocationSize::precise(8)), (i8** %ptr_b, LocationSize::precise(8)), (i8* %a, LocationSize::precise(1)), (i8* %b, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
define void @test101(i8** %ptr_a, i1 %cond_a, i8** %ptr_b) {
entry:
  %a = load i8*, i8** %ptr_a
  %b = load i8*, i8** %ptr_b
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  %0 = load i8, i8* %a
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test102':
; CHECK: Alias Set Tracker: 1 alias sets for 4 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 5] may alias, Mod/Ref   Pointers: (i8** %ptr_a, LocationSize::precise(8)), (i8** %ptr_b, LocationSize::precise(8)), (i8* %a, LocationSize::precise(1)), (i8* %b, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
define void @test102(i8** %ptr_a, i1 %cond_a, i8** %ptr_b) {
entry:
  %a = load i8*, i8** %ptr_a
  %b = load i8*, i8** %ptr_b
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  store i8 0, i8* %a
  %0 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test103':
; CHECK: Alias Set Tracker: 1 alias sets for 4 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 5] may alias, Mod/Ref   Pointers: (i8** %ptr_a, LocationSize::precise(8)), (i8** %ptr_b, LocationSize::precise(8)), (i8* %a, LocationSize::precise(1)), (i8* %b, LocationSize::precise(1))
; CHECK:     1 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
define void @test103(i8** %ptr_a, i1 %cond_a, i8** %ptr_b) {
entry:
  %a = load i8*, i8** %ptr_a
  %b = load i8*, i8** %ptr_b
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  store i8 0, i8* %a
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test104':
; CHECK: Alias Set Tracker: 1 alias sets for 4 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 5] may alias, Ref       Pointers: (i8** %ptr_a, LocationSize::precise(8)), (i8** %ptr_b, LocationSize::precise(8)), (i8* %a, LocationSize::precise(1)), (i8* %b, LocationSize::precise(1))
; CHECK:     2 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ],   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
define void @test104(i8** %ptr_a, i1 %cond_a, i8** %ptr_b, i1 %cond_b) {
entry:
  %a = load i8*, i8** %ptr_a
  %b = load i8*, i8** %ptr_b
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  %0 = load i8, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  %1 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test105':
; CHECK: Alias Set Tracker: 1 alias sets for 4 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 5] may alias, Mod/Ref   Pointers: (i8** %ptr_a, LocationSize::precise(8)), (i8** %ptr_b, LocationSize::precise(8)), (i8* %a, LocationSize::precise(1)), (i8* %b, LocationSize::precise(1))
; CHECK:     2 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ],   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
define void @test105(i8** %ptr_a, i1 %cond_a, i8** %ptr_b, i1 %cond_b) {
entry:
  %a = load i8*, i8** %ptr_a
  %b = load i8*, i8** %ptr_b
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  %0 = load i8, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  store i8 1, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test106':
; CHECK: Alias Set Tracker: 1 alias sets for 4 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 5] may alias, Mod/Ref   Pointers: (i8** %ptr_a, LocationSize::precise(8)), (i8** %ptr_b, LocationSize::precise(8)), (i8* %a, LocationSize::precise(1)), (i8* %b, LocationSize::precise(1))
; CHECK:     2 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ],   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
define void @test106(i8** %ptr_a, i1 %cond_a, i8** %ptr_b, i1 %cond_b) {
entry:
  %a = load i8*, i8** %ptr_a
  %b = load i8*, i8** %ptr_b
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  store i8 0, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  %0 = load i8, i8* %b
  ret void
}

; CHECK: Alias sets for function 'test107':
; CHECK: Alias Set Tracker: 1 alias sets for 4 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 5] may alias, Mod/Ref   Pointers: (i8** %ptr_a, LocationSize::precise(8)), (i8** %ptr_b, LocationSize::precise(8)), (i8* %a, LocationSize::precise(1)), (i8* %b, LocationSize::precise(1))
; CHECK:     2 Unknown instructions:   call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ],   call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
define void @test107(i8** %ptr_a, i1 %cond_a, i8** %ptr_b, i1 %cond_b) {
entry:
  %a = load i8*, i8** %ptr_a
  %b = load i8*, i8** %ptr_b
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_a) [ "deopt"() ]
  store i8 0, i8* %a
  call void (i1, ...) @llvm.experimental.guard(i1 %cond_b) [ "deopt"() ]
  store i8 1, i8* %b
  ret void
}
