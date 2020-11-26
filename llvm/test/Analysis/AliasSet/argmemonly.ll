; RUN: opt -basic-aa -print-alias-sets -S -o - < %s 2>&1 | FileCheck %s

@s = global i8 1, align 1
@d = global i8 2, align 1

; CHECK: Alias sets for function 'test_alloca_argmemonly':
; CHECK-NEXT: Alias Set Tracker: 2 alias sets for 3 pointer values.
; CHECK-NEXT: AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %a, LocationSize::precise(1))
; CHECK-NEXT: AliasSet[0x{{[0-9a-f]+}}, 2] may alias, Mod/Ref    Pointers: (i8* %d, unknown before-or-after), (i8* %s, unknown before-or-after)
define void @test_alloca_argmemonly(i8* %s, i8* %d) {
entry:
  %a = alloca i8, align 1
  store i8 1, i8* %a, align 1
  call void @my_memcpy(i8* %d, i8* %s, i64 1)
  ret void
}

; CHECK: Alias sets for function 'test_readonly_arg'
; CHECK-NEXT: Alias Set Tracker: 2 alias sets for 2 pointer values.
; CHECK-NEXT: AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %d, unknown before-or-after)
; CHECK-NEXT: AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Ref       Pointers: (i8* %s, unknown before-or-after)
define i8 @test_readonly_arg(i8* noalias %s, i8* noalias %d) {
entry:
  call void @my_memcpy(i8* %d, i8* %s, i64 1)
  %ret = load i8, i8* %s
  ret i8 %ret
}

; CHECK: Alias sets for function 'test_noalias_argmemonly':
; CHECK-NEXT: Alias Set Tracker: 2 alias sets for 3 pointer values.
; CHECK-NEXT: AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Pointers: (i8* %a, LocationSize::precise(1))
; CHECK-NEXT: AliasSet[0x{{[0-9a-f]+}}, 2] may alias, Mod/Ref    Pointers: (i8* %d, unknown before-or-after), (i8* %s, unknown before-or-after)
define void @test_noalias_argmemonly(i8* noalias %a, i8* %s, i8* %d) {
entry:
  store i8 1, i8* %a, align 1
  call void @my_memmove(i8* %d, i8* %s, i64 1)
  ret void
}

; CHECK: Alias sets for function 'test5':
; CHECK-NEXT: Alias Set Tracker: 2 alias sets for 2 pointer values.
; CHECK-NEXT: AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod/Ref Pointers: (i8* %a, unknown before-or-after)
; CHECK-NEXT: AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod Pointers: (i8* %b, unknown before-or-after)
define void @test5(i8* noalias %a, i8* noalias %b) {
entry:
  store i8 1, i8* %a, align 1
  call void @my_memcpy(i8* %b, i8* %a, i64 1)
  store i8 1, i8* %b, align 1
  ret void
}

; CHECK: Alias sets for function 'test_argcollapse':
; CHECK-NEXT: Alias Set Tracker: 2 alias sets for 2 pointer values.
; CHECK-NEXT: AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod/Ref Pointers: (i8* %a, unknown before-or-after)
; CHECK-NEXT: AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod/Ref Pointers: (i8* %b, unknown before-or-after)
define void @test_argcollapse(i8* noalias %a, i8* noalias %b) {
entry:
  store i8 1, i8* %a, align 1
  call void @my_memmove(i8* %b, i8* %a, i64 1)
  store i8 1, i8* %b, align 1
  ret void
}

; CHECK: Alias sets for function 'test_memcpy1':
; CHECK-NEXT: Alias Set Tracker: 2 alias sets for 2 pointer values.
; CHECK-NEXT: AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod/Ref Pointers: (i8* %b, unknown before-or-after)
; CHECK-NEXT: AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod/Ref Pointers: (i8* %a, unknown before-or-after)
define void @test_memcpy1(i8* noalias %a, i8* noalias %b) {
entry:
  call void @my_memcpy(i8* %b, i8* %a, i64 1)
  call void @my_memcpy(i8* %a, i8* %b, i64 1)
  ret void
}

; CHECK: Alias sets for function 'test_memset1':
; CHECK-NEXT: Alias Set Tracker: 1 alias sets for 1 pointer values.
; CHECK-NEXT: AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod Pointers: (i8* %a, unknown before-or-after)
define void @test_memset1() {
entry:
  %a = alloca i8, align 1
  call void @my_memset(i8* %a, i8 0, i64 1)
  ret void
}

; CHECK: Alias sets for function 'test_memset2':
; CHECK-NEXT: Alias Set Tracker: 1 alias sets for 1 pointer values.
; CHECK-NEXT: AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod Pointers: (i8* %a, unknown before-or-after)
define void @test_memset2(i8* %a) {
entry:
  call void @my_memset(i8* %a, i8 0, i64 1)
  ret void
}

; CHECK: Alias sets for function 'test_memset3':
; CHECK-NEXT: Alias Set Tracker: 1 alias sets for 2 pointer values.
; CHECK-NEXT: AliasSet[0x{{[0-9a-f]+}}, 2] may alias, Mod Pointers: (i8* %a, unknown before-or-after), (i8* %b, unknown before-or-after)
define void @test_memset3(i8* %a, i8* %b) {
entry:
  call void @my_memset(i8* %a, i8 0, i64 1)
  call void @my_memset(i8* %b, i8 0, i64 1)
  ret void
}

;; PICKUP HERE

; CHECK: Alias sets for function 'test_memset4':
; CHECK-NEXT: Alias Set Tracker: 2 alias sets for 2 pointer values.
; CHECK-NEXT: AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod Pointers: (i8* %a, unknown before-or-after)
; CHECK-NEXT: AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod Pointers: (i8* %b, unknown before-or-after)
define void @test_memset4(i8* noalias %a, i8* noalias %b) {
entry:
  call void @my_memset(i8* %a, i8 0, i64 1)
  call void @my_memset(i8* %b, i8 0, i64 1)
  ret void
}

declare void @my_memset(i8* nocapture writeonly, i8, i64) argmemonly
declare void @my_memcpy(i8* nocapture writeonly, i8* nocapture readonly, i64) argmemonly
declare void @my_memmove(i8* nocapture, i8* nocapture readonly, i64) argmemonly


; CHECK: Alias sets for function 'test_attribute_intersect':
; CHECK-NEXT: Alias Set Tracker: 1 alias sets for 1 pointer values.
; CHECK-NEXT: AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Ref       Pointers: (i8* %a, LocationSize::precise(1))
define i8 @test_attribute_intersect(i8* noalias %a) {
entry:
  ;; This call is effectively readnone since the argument is readonly
  ;; and the function is declared writeonly.  
  call void @attribute_intersect(i8* %a)
  %val = load i8, i8* %a
  ret i8 %val
}

declare void @attribute_intersect(i8* readonly) argmemonly writeonly

