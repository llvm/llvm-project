; RUN: opt -basicaa -print-alias-sets -S -o - < %s 2>&1 | FileCheck %s

@s = global i8 1, align 1
@d = global i8 2, align 1

; CHECK: Alias sets for function 'test1':
; CHECK: Alias Set Tracker: 3 alias sets for 4 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %a, 1)
; CHECK-NOT:    1 Unknown instructions
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 2] may alias, Mod/Ref   Pointers: (i8* %s, 1), (i8* %d, 1)
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %b, 1)
define void @test1(i8* %s, i8* %d) {
entry:
  %a = alloca i8, align 1
  %b = alloca i8, align 1
  store i8 1, i8* %a, align 1
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %d, i8* %s, i64 1, i1 false)
  store i8 1, i8* %b, align 1
  ret void
}

; CHECK: Alias sets for function 'test2':
; CHECK: Alias Set Tracker: 3 alias sets for 4 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %a, 1)
; CHECK-NOT:    1 Unknown instructions
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 2] may alias, Mod/Ref   [volatile] Pointers: (i8* %s, 1), (i8* %d, 1)
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %b, 1)
define void @test2(i8* %s, i8* %d) {
entry:
  %a = alloca i8, align 1
  %b = alloca i8, align 1
  store i8 1, i8* %a, align 1
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %d, i8* %s, i64 1, i1 true)
  store i8 1, i8* %b, align 1
  ret void
}

; CHECK: Alias sets for function 'test3':
; CHECK: Alias Set Tracker: 3 alias sets for 4 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %a, 1)
; CHECK-NOT:    1 Unknown instructions
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 2] may alias, Mod/Ref   Pointers: (i8* %s, 1), (i8* %d, 1)
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %b, 1)
define void @test3(i8* %s, i8* %d) {
entry:
  %a = alloca i8, align 1
  %b = alloca i8, align 1
  store i8 1, i8* %a, align 1
  call void @llvm.memmove.p0i8.p0i8.i64(i8* %d, i8* %s, i64 1, i1 false)
  store i8 1, i8* %b, align 1
  ret void
}

; CHECK: Alias sets for function 'test4':
; CHECK: Alias Set Tracker: 3 alias sets for 4 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %a, 1)
; CHECK-NOT:    1 Unknown instructions
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 2] may alias, Mod/Ref   [volatile] Pointers: (i8* %s, 1), (i8* %d, 1)
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %b, 1)
define void @test4(i8* %s, i8* %d) {
entry:
  %a = alloca i8, align 1
  %b = alloca i8, align 1
  store i8 1, i8* %a, align 1
  call void @llvm.memmove.p0i8.p0i8.i64(i8* %d, i8* %s, i64 1, i1 true)
  store i8 1, i8* %b, align 1
  ret void
}

; CHECK: Alias sets for function 'test5':
; CHECK: Alias Set Tracker: 2 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod/Ref   Pointers: (i8* %a, 1)
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %b, 1)
define void @test5() {
entry:
  %a = alloca i8, align 1
  %b = alloca i8, align 1
  store i8 1, i8* %a, align 1
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %b, i8* %a, i64 1, i1 false)
  store i8 1, i8* %b, align 1
  ret void
}

; CHECK: Alias sets for function 'test6':
; CHECK: Alias Set Tracker: 2 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod/Ref   Pointers: (i8* %a, 1)
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %b, 1)
define void @test6() {
entry:
  %a = alloca i8, align 1
  %b = alloca i8, align 1
  store i8 1, i8* %a, align 1
  call void @llvm.memmove.p0i8.p0i8.i64(i8* %b, i8* %a, i64 1, i1 false)
  store i8 1, i8* %b, align 1
  ret void
}

; CHECK: Alias sets for function 'test7':
; CHECK: Alias Set Tracker: 2 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod/Ref   Pointers: (i8* %a, 1)
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod/Ref   Pointers: (i8* %b, 1)
define void @test7() {
entry:
  %a = alloca i8, align 1
  %b = alloca i8, align 1
  store i8 1, i8* %a, align 1
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %b, i8* %a, i64 1, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %a, i8* %b, i64 1, i1 false)
  store i8 1, i8* %b, align 1
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1)
declare void @llvm.memmove.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i1)
