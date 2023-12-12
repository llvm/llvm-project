; RUN: opt -passes=print-alias-sets -alias-set-saturation-threshold=4 -S -o - < %s 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=NOSAT
; RUN: opt -passes=print-alias-sets -alias-set-saturation-threshold=3 -S -o - < %s 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=SAT

; CHECK-LABEL: 'nomerge'
; CHECK: AliasSet[{{.*}}, 1] must alias, Mod Pointers: (ptr %a, LocationSize::precise(4))
; CHECK: AliasSet[{{.*}}, 2] may alias, Mod Pointers: (ptr %b, LocationSize::precise(4)), (ptr %b1, LocationSize::precise(4))
define void @nomerge(i32 %k) {
  %a = alloca i32
  %b = alloca [10 x i32]
  store i32 1, ptr %a
  store i32 2, ptr %b
  %b1 = getelementptr i32, ptr %b, i32 %k
  store i32 3, ptr %b1
  ret void
}

; CHECK-LABEL: 'mergemay'
; NOSAT: AliasSet[{{.*}}, 3] may alias, Mod Pointers: (ptr %a, LocationSize::precise(4)), (ptr %a1, LocationSize::precise(4)), (ptr %a2, LocationSize::precise(4))
; NOSAT: AliasSet[{{.*}}, 1] must alias, Mod Pointers: (ptr %b, LocationSize::precise(4))
; SAT: AliasSet[{{.*}}, 3] may alias, Mod forwarding to 0x[[FWD:[0-9a-f]*]]
; SAT: AliasSet[{{.*}}, 1] must alias, Mod forwarding to 0x[[FWD]]
; SAT: AliasSet[0x[[FWD]], 2] may alias, Mod/Ref Pointers: (ptr %a, LocationSize::precise(4)), (ptr %a1, LocationSize::precise(4)), (ptr %a2, LocationSize::precise(4)), (ptr %b, LocationSize::precise(4))
define void @mergemay(i32 %k, i32 %l) {
  %a = alloca i32
  %b = alloca i32
  store i32 1, ptr %a
  store i32 2, ptr %b
  %a1 = getelementptr i32, ptr %a, i32 %k
  store i32 2, ptr %a1
  %a2 = getelementptr i32, ptr %a, i32 %l
  store i32 2, ptr %a2
  ret void
}

; CHECK-LABEL: 'mergemust'
; NOSAT: AliasSet[{{.*}}, 1] must alias, Mod Pointers: (ptr %a, LocationSize::precise(4))
; NOSAT: AliasSet[{{.*}}, 1] must alias, Mod Pointers: (ptr %b, LocationSize::precise(4))
; NOSAT: AliasSet[{{.*}}, 2] may alias,  Mod Pointers: (ptr %c, LocationSize::precise(4)), (ptr %d, LocationSize::precise(4))
; SAT: AliasSet[{{.*}}, 1] must alias, Mod forwarding to 0x[[FWD:[0-9a-f]*]]
; SAT: AliasSet[{{.*}}, 1] must alias, Mod forwarding to 0x[[FWD]]
; SAT: AliasSet[{{.*}}, 2] may alias,  Mod forwarding to 0x[[FWD]]
; SAT: AliasSet[0x[[FWD]], 3] may alias, Mod/Ref Pointers: (ptr %a, LocationSize::precise(4)), (ptr %b, LocationSize::precise(4)), (ptr %c, LocationSize::precise(4)), (ptr %d, LocationSize::precise(4))
define void @mergemust(ptr %c, ptr %d) {
  %a = alloca i32
  %b = alloca i32
  store i32 1, ptr %a
  store i32 2, ptr %b
  store i32 3, ptr %c
  store i32 4, ptr %d
  ret void
}
