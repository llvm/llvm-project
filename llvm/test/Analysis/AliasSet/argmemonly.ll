; RUN: opt -passes=print-alias-sets -S -o - < %s 2>&1 | FileCheck %s

@s = global i8 1, align 1
@d = global i8 2, align 1

; CHECK: Alias sets for function 'test_alloca_argmemonly':
; CHECK-NEXT: Alias Set Tracker: 2 alias sets for 3 pointer values.
; CHECK-NEXT: AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Memory locations: (ptr %a, LocationSize::precise(1))
; CHECK-NEXT: AliasSet[0x{{[0-9a-f]+}}, 2] may alias, Mod/Ref    Memory locations: (ptr %d, unknown before-or-after), (ptr %s, unknown before-or-after)
define void @test_alloca_argmemonly(ptr %s, ptr %d) {
entry:
  %a = alloca i8, align 1
  store i8 1, ptr %a, align 1
  call void @my_memcpy(ptr %d, ptr %s, i64 1)
  ret void
}

; CHECK: Alias sets for function 'test_readonly_arg'
; CHECK-NEXT: Alias Set Tracker: 2 alias sets for 2 pointer values.
; CHECK-NEXT: AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Memory locations: (ptr %d, unknown before-or-after)
; CHECK-NEXT: AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Ref       Memory locations: (ptr %s, unknown before-or-after), (ptr %s, LocationSize::precise(1))
define i8 @test_readonly_arg(ptr noalias %s, ptr noalias %d) {
entry:
  call void @my_memcpy(ptr %d, ptr %s, i64 1)
  %ret = load i8, ptr %s
  ret i8 %ret
}

; CHECK: Alias sets for function 'test_noalias_argmemonly':
; CHECK-NEXT: Alias Set Tracker: 2 alias sets for 3 pointer values.
; CHECK-NEXT: AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod       Memory locations: (ptr %a, LocationSize::precise(1))
; CHECK-NEXT: AliasSet[0x{{[0-9a-f]+}}, 2] may alias, Mod/Ref    Memory locations: (ptr %d, unknown before-or-after), (ptr %s, unknown before-or-after)
define void @test_noalias_argmemonly(ptr noalias %a, ptr %s, ptr %d) {
entry:
  store i8 1, ptr %a, align 1
  call void @my_memmove(ptr %d, ptr %s, i64 1)
  ret void
}

; CHECK: Alias sets for function 'test5':
; CHECK-NEXT: Alias Set Tracker: 2 alias sets for 2 pointer values.
; CHECK-NEXT: AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod/Ref Memory locations: (ptr %a, LocationSize::precise(1)), (ptr %a, unknown before-or-after)
; CHECK-NEXT: AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod Memory locations: (ptr %b, unknown before-or-after), (ptr %b, LocationSize::precise(1))
define void @test5(ptr noalias %a, ptr noalias %b) {
entry:
  store i8 1, ptr %a, align 1
  call void @my_memcpy(ptr %b, ptr %a, i64 1)
  store i8 1, ptr %b, align 1
  ret void
}

; CHECK: Alias sets for function 'test_argcollapse':
; CHECK-NEXT: Alias Set Tracker: 2 alias sets for 2 pointer values.
; CHECK-NEXT: AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod/Ref Memory locations: (ptr %a, LocationSize::precise(1)), (ptr %a, unknown before-or-after)
; CHECK-NEXT: AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod/Ref Memory locations: (ptr %b, unknown before-or-after), (ptr %b, LocationSize::precise(1))
define void @test_argcollapse(ptr noalias %a, ptr noalias %b) {
entry:
  store i8 1, ptr %a, align 1
  call void @my_memmove(ptr %b, ptr %a, i64 1)
  store i8 1, ptr %b, align 1
  ret void
}

; CHECK: Alias sets for function 'test_memcpy1':
; CHECK-NEXT: Alias Set Tracker: 2 alias sets for 2 pointer values.
; CHECK-NEXT: AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod/Ref Memory locations: (ptr %b, unknown before-or-after)
; CHECK-NEXT: AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod/Ref Memory locations: (ptr %a, unknown before-or-after)
define void @test_memcpy1(ptr noalias %a, ptr noalias %b) {
entry:
  call void @my_memcpy(ptr %b, ptr %a, i64 1)
  call void @my_memcpy(ptr %a, ptr %b, i64 1)
  ret void
}

; CHECK: Alias sets for function 'test_memset1':
; CHECK-NEXT: Alias Set Tracker: 1 alias sets for 1 pointer values.
; CHECK-NEXT: AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod Memory locations: (ptr %a, unknown before-or-after)
define void @test_memset1() {
entry:
  %a = alloca i8, align 1
  call void @my_memset(ptr %a, i8 0, i64 1)
  ret void
}

; CHECK: Alias sets for function 'test_memset2':
; CHECK-NEXT: Alias Set Tracker: 1 alias sets for 1 pointer values.
; CHECK-NEXT: AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod Memory locations: (ptr %a, unknown before-or-after)
define void @test_memset2(ptr %a) {
entry:
  call void @my_memset(ptr %a, i8 0, i64 1)
  ret void
}

; CHECK: Alias sets for function 'test_memset3':
; CHECK-NEXT: Alias Set Tracker: 1 alias sets for 2 pointer values.
; CHECK-NEXT: AliasSet[0x{{[0-9a-f]+}}, 2] may alias, Mod Memory locations: (ptr %a, unknown before-or-after), (ptr %b, unknown before-or-after)
define void @test_memset3(ptr %a, ptr %b) {
entry:
  call void @my_memset(ptr %a, i8 0, i64 1)
  call void @my_memset(ptr %b, i8 0, i64 1)
  ret void
}

;; PICKUP HERE

; CHECK: Alias sets for function 'test_memset4':
; CHECK-NEXT: Alias Set Tracker: 2 alias sets for 2 pointer values.
; CHECK-NEXT: AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod Memory locations: (ptr %a, unknown before-or-after)
; CHECK-NEXT: AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Mod Memory locations: (ptr %b, unknown before-or-after)
define void @test_memset4(ptr noalias %a, ptr noalias %b) {
entry:
  call void @my_memset(ptr %a, i8 0, i64 1)
  call void @my_memset(ptr %b, i8 0, i64 1)
  ret void
}

declare void @my_memset(ptr nocapture writeonly, i8, i64) argmemonly
declare void @my_memcpy(ptr nocapture writeonly, ptr nocapture readonly, i64) argmemonly
declare void @my_memmove(ptr nocapture, ptr nocapture readonly, i64) argmemonly


; CHECK: Alias sets for function 'test_attribute_intersect':
; CHECK-NEXT: Alias Set Tracker: 1 alias sets for 1 pointer values.
; CHECK-NEXT: AliasSet[0x{{[0-9a-f]+}}, 1] must alias, Ref       Memory locations: (ptr %a, LocationSize::precise(1))
define i8 @test_attribute_intersect(ptr noalias %a) {
entry:
  ;; This call is effectively readnone since the argument is readonly
  ;; and the function is declared writeonly.  
  call void @attribute_intersect(ptr %a)
  %val = load i8, ptr %a
  ret i8 %val
}

declare void @attribute_intersect(ptr readonly) argmemonly writeonly

