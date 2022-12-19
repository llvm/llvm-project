; RUN: opt -passes=print-alias-sets -S -o - < %s 2>&1 | FileCheck %s

@s = global i8 1, align 1
@d = global i8 2, align 1


; CHECK: Alias sets for function 'test_known_size':
; CHECK: Alias Set Tracker: 2 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (ptr %d, LocationSize::precise(1))
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Ref       Pointers: (ptr %s, LocationSize::precise(1))
define void @test_known_size(ptr noalias %s, ptr noalias %d) {
entry:
  call void @llvm.memcpy.p0.p0.i64(ptr %d, ptr %s, i64 1, i1 false)
  ret void
}

; CHECK: Alias sets for function 'test_unknown_size':
; CHECK: Alias Set Tracker: 2 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (ptr %d, unknown after)
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Ref       Pointers: (ptr %s, unknown after)
define void @test_unknown_size(ptr noalias %s, ptr noalias %d, i64 %len) {
entry:
  call void @llvm.memcpy.p0.p0.i64(ptr %d, ptr %s, i64 %len, i1 false)
  ret void
}


; CHECK: Alias sets for function 'test1':
; CHECK: Alias Set Tracker: 3 alias sets for 4 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (ptr %a, LocationSize::precise(1))
; CHECK-NOT:    1 Unknown instructions
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 2] may alias, Mod/Ref   Pointers: (ptr %d, LocationSize::precise(1)), (ptr %s, LocationSize::precise(1))
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (ptr %b, LocationSize::precise(1))
define void @test1(ptr %s, ptr %d) {
entry:
  %a = alloca i8, align 1
  %b = alloca i8, align 1
  store i8 1, ptr %a, align 1
  call void @llvm.memcpy.p0.p0.i64(ptr %d, ptr %s, i64 1, i1 false)
  store i8 1, ptr %b, align 1
  ret void
}

; CHECK: Alias sets for function 'test1_atomic':
; CHECK: Alias Set Tracker: 3 alias sets for 4 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (ptr %a, LocationSize::precise(1))
; CHECK-NOT:    1 Unknown instructions
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 2] may alias, Mod/Ref   Pointers: (ptr %d, LocationSize::precise(1)), (ptr %s, LocationSize::precise(1))
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (ptr %b, LocationSize::precise(1))
define void @test1_atomic(ptr %s, ptr %d) {
entry:
  %a = alloca i8, align 1
  %b = alloca i8, align 1
  store atomic i8 1, ptr %a unordered, align 1
  call void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr align 1 %d, ptr align 1 %s, i64 1, i32 1)
  store atomic i8 1, ptr %b unordered, align 1
  ret void
}

; CHECK: Alias sets for function 'test2':
; CHECK: Alias Set Tracker: 3 alias sets for 4 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (ptr %a, LocationSize::precise(1))
; CHECK-NOT:    1 Unknown instructions
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 2] may alias, Mod/Ref    Pointers: (ptr %d, LocationSize::precise(1)), (ptr %s, LocationSize::precise(1))
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (ptr %b, LocationSize::precise(1))
define void @test2(ptr %s, ptr %d) {
entry:
  %a = alloca i8, align 1
  %b = alloca i8, align 1
  store i8 1, ptr %a, align 1
  call void @llvm.memcpy.p0.p0.i64(ptr %d, ptr %s, i64 1, i1 true)
  store i8 1, ptr %b, align 1
  ret void
}

; CHECK: Alias sets for function 'test3':
; CHECK: Alias Set Tracker: 3 alias sets for 4 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (ptr %a, LocationSize::precise(1))
; CHECK-NOT:    1 Unknown instructions
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 2] may alias, Mod/Ref   Pointers: (ptr %d, LocationSize::precise(1)), (ptr %s, LocationSize::precise(1))
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (ptr %b, LocationSize::precise(1))
define void @test3(ptr %s, ptr %d) {
entry:
  %a = alloca i8, align 1
  %b = alloca i8, align 1
  store i8 1, ptr %a, align 1
  call void @llvm.memmove.p0.p0.i64(ptr %d, ptr %s, i64 1, i1 false)
  store i8 1, ptr %b, align 1
  ret void
}

; CHECK: Alias sets for function 'test3_atomic':
; CHECK: Alias Set Tracker: 3 alias sets for 4 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (ptr %a, LocationSize::precise(1))
; CHECK-NOT:    1 Unknown instructions
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 2] may alias, Mod/Ref   Pointers: (ptr %d, LocationSize::precise(1)), (ptr %s, LocationSize::precise(1))
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (ptr %b, LocationSize::precise(1))
define void @test3_atomic(ptr %s, ptr %d) {
entry:
  %a = alloca i8, align 1
  %b = alloca i8, align 1
  store atomic i8 1, ptr %a unordered, align 1
  call void @llvm.memmove.element.unordered.atomic.p0.p0.i64(ptr align 1 %d, ptr align 1 %s, i64 1, i32 1)
  store atomic i8 1, ptr %b unordered, align 1
  ret void
}

; CHECK: Alias sets for function 'test4':
; CHECK: Alias Set Tracker: 3 alias sets for 4 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (ptr %a, LocationSize::precise(1))
; CHECK-NOT:    1 Unknown instructions
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 2] may alias, Mod/Ref    Pointers: (ptr %d, LocationSize::precise(1)), (ptr %s, LocationSize::precise(1))
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (ptr %b, LocationSize::precise(1))
define void @test4(ptr %s, ptr %d) {
entry:
  %a = alloca i8, align 1
  %b = alloca i8, align 1
  store i8 1, ptr %a, align 1
  call void @llvm.memmove.p0.p0.i64(ptr %d, ptr %s, i64 1, i1 true)
  store i8 1, ptr %b, align 1
  ret void
}

; CHECK: Alias sets for function 'test5':
; CHECK: Alias Set Tracker: 2 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod/Ref   Pointers: (ptr %a, LocationSize::precise(1))
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (ptr %b, LocationSize::precise(1))
define void @test5() {
entry:
  %a = alloca i8, align 1
  %b = alloca i8, align 1
  store i8 1, ptr %a, align 1
  call void @llvm.memcpy.p0.p0.i64(ptr %b, ptr %a, i64 1, i1 false)
  store i8 1, ptr %b, align 1
  ret void
}

; CHECK: Alias sets for function 'test5_atomic':
; CHECK: Alias Set Tracker: 2 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod/Ref   Pointers: (ptr %a, LocationSize::precise(1))
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (ptr %b, LocationSize::precise(1))
define void @test5_atomic() {
entry:
  %a = alloca i8, align 1
  %b = alloca i8, align 1
  store atomic i8 1, ptr %a unordered, align 1
  call void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr align 1 %b, ptr align 1 %a, i64 1, i32 1)
  store atomic i8 1, ptr %b unordered, align 1
  ret void
}

; CHECK: Alias sets for function 'test6':
; CHECK: Alias Set Tracker: 2 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod/Ref   Pointers: (ptr %a, LocationSize::precise(1))
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (ptr %b, LocationSize::precise(1))
define void @test6() {
entry:
  %a = alloca i8, align 1
  %b = alloca i8, align 1
  store i8 1, ptr %a, align 1
  call void @llvm.memmove.p0.p0.i64(ptr %b, ptr %a, i64 1, i1 false)
  store i8 1, ptr %b, align 1
  ret void
}

; CHECK: Alias sets for function 'test6_atomic':
; CHECK: Alias Set Tracker: 2 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod/Ref   Pointers: (ptr %a, LocationSize::precise(1))
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (ptr %b, LocationSize::precise(1))
define void @test6_atomic() {
entry:
  %a = alloca i8, align 1
  %b = alloca i8, align 1
  store atomic i8 1, ptr %a unordered, align 1
  call void @llvm.memmove.element.unordered.atomic.p0.p0.i64(ptr align 1 %b, ptr align 1 %a, i64 1, i32 1)
  store atomic i8 1, ptr %b unordered, align 1
  ret void
}

; CHECK: Alias sets for function 'test7':
; CHECK: Alias Set Tracker: 2 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod/Ref   Pointers: (ptr %a, LocationSize::precise(1))
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod/Ref   Pointers: (ptr %b, LocationSize::precise(1))
define void @test7() {
entry:
  %a = alloca i8, align 1
  %b = alloca i8, align 1
  store i8 1, ptr %a, align 1
  call void @llvm.memcpy.p0.p0.i64(ptr %b, ptr %a, i64 1, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr %a, ptr %b, i64 1, i1 false)
  store i8 1, ptr %b, align 1
  ret void
}

; CHECK: Alias sets for function 'test7_atomic':
; CHECK: Alias Set Tracker: 2 alias sets for 2 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod/Ref   Pointers: (ptr %a, LocationSize::precise(1))
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod/Ref   Pointers: (ptr %b, LocationSize::precise(1))
define void @test7_atomic() {
entry:
  %a = alloca i8, align 1
  %b = alloca i8, align 1
  store atomic i8 1, ptr %a unordered, align 1
  call void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr align 1 %b, ptr align 1 %a, i64 1, i32 1)
  call void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr align 1 %a, ptr align 1 %b, i64 1, i32 1)
  store atomic i8 1, ptr %b unordered, align 1
  ret void
}

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1)
declare void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i32)
declare void @llvm.memmove.p0.p0.i64(ptr nocapture, ptr nocapture readonly, i64, i1)
declare void @llvm.memmove.element.unordered.atomic.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i32)
