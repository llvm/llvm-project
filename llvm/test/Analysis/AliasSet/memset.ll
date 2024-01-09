; RUN: opt -passes=print-alias-sets -S -o - < %s 2>&1 | FileCheck %s

@s = global i8 1, align 1
@d = global i8 2, align 1

; CHECK: Alias sets for function 'test_known_size':
; CHECK: Alias Set Tracker: 1 alias sets for 1 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (ptr %d, LocationSize::precise(1))
define void @test_known_size(ptr noalias %d) {
entry:
  call void @llvm.memset.p0.i64(ptr align 1 %d, i8 0, i64 1, i1 false)
  ret void
}

; CHECK: Alias sets for function 'test_unknown_size':
; CHECK: Alias Set Tracker: 1 alias sets for 1 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (ptr %d, unknown after)
define void @test_unknown_size(ptr noalias %d, i64 %len) {
entry:
  call void @llvm.memset.p0.i64(ptr align 1 %d, i8 0, i64 %len, i1 false)
  ret void
}


; CHECK: Alias sets for function 'test_atomic_known_size':
; CHECK: Alias Set Tracker: 1 alias sets for 1 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (ptr %d, LocationSize::precise(1))
define void @test_atomic_known_size(ptr noalias %d) {
entry:
  call void @llvm.memset.element.unordered.atomic.p0.i32(ptr align 1 %d, i8 0, i64 1, i32 1)
  ret void
}

; CHECK: Alias sets for function 'test_atomic_unknown_size':
; CHECK: Alias Set Tracker: 1 alias sets for 1 pointer values.
; CHECK:   AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (ptr %d, unknown after)
define void @test_atomic_unknown_size(ptr noalias %d, i64 %len) {
entry:
  call void @llvm.memset.element.unordered.atomic.p0.i32(ptr align 1 %d, i8 0, i64 %len, i32 1)
  ret void
}

declare void @llvm.memset.p0.i64(ptr %dest, i8 %val,
                                   i64 %len, i1 %isvolatile)
declare void @llvm.memset.element.unordered.atomic.p0.i32(ptr %dest,
                                                            i8 %value,
                                                            i64 %len,
                                                            i32 %element_size)
