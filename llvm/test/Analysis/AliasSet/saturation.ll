; RUN: opt -print-alias-sets -alias-set-saturation-threshold=2 -S -o - < %s 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=NOSAT
; RUN: opt -print-alias-sets -alias-set-saturation-threshold=1 -S -o - < %s 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=SAT

; CHECK-LABEL: 'allmust'
; CHECK: AliasSet[{{.*}}, 1] must alias, Mod Pointers: (i32* %a, LocationSize::precise(4))
; CHECK: AliasSet[{{.*}}, 1] must alias, Mod Pointers: (i32* %b, LocationSize::precise(4))
; CHECK: AliasSet[{{.*}}, 1] must alias, Mod Pointers: (i32* %c, LocationSize::precise(4))
; CHECK: AliasSet[{{.*}}, 1] must alias, Mod Pointers: (i32* %d, LocationSize::precise(4))
define void @allmust() {
  %a = alloca i32
  %b = alloca i32
  %c = alloca i32
  %d = alloca i32
  store i32 1, i32* %a
  store i32 2, i32* %b
  store i32 3, i32* %c
  store i32 4, i32* %d
  ret void
}

; CHECK-LABEL: 'mergemay'
; NOSAT: AliasSet[{{.*}}, 2] may alias, Mod Pointers: (i32* %a, LocationSize::precise(4)), (i32* %a1, LocationSize::precise(4))
; NOSAT: AliasSet[{{.*}}, 1] must alias, Mod Pointers: (i32* %b, LocationSize::precise(4))
; SAT: AliasSet[{{.*}}, 2] may alias, Mod forwarding to 0x[[FWD:[0-9a-f]*]]
; SAT: AliasSet[{{.*}}, 1] must alias, Mod forwarding to 0x[[FWD]]
; SAT: AliasSet[0x[[FWD]], 2] may alias, Mod/Ref Pointers: (i32* %a, LocationSize::precise(4)), (i32* %a1, LocationSize::precise(4)), (i32* %b, LocationSize::precise(4))
define void @mergemay(i32 %k) {
  %a = alloca i32
  %b = alloca i32
  store i32 1, i32* %a
  store i32 2, i32* %b
  %a1 = getelementptr i32, i32 *%a, i32 %k
  store i32 2, i32* %a1  
  ret void
}

; CHECK-LABEL: 'mergemust'
; NOSAT: AliasSet[{{.*}}, 1] must alias, Mod Pointers: (i32* %a, LocationSize::precise(4))
; NOSAT: AliasSet[{{.*}}, 1] must alias, Mod Pointers: (i32* %b, LocationSize::precise(4))
; NOSAT: AliasSet[{{.*}}, 2] may alias,  Mod Pointers: (i32* %c, LocationSize::precise(4)), (i32* %d, LocationSize::precise(4))
; SAT: AliasSet[{{.*}}, 1] must alias, Mod forwarding to 0x[[FWD:[0-9a-f]*]]
; SAT: AliasSet[{{.*}}, 1] must alias, Mod forwarding to 0x[[FWD]]
; SAT: AliasSet[{{.*}}, 2] may alias,  Mod forwarding to 0x[[FWD]]
; SAT: AliasSet[0x[[FWD]], 3] may alias, Mod/Ref Pointers: (i32* %a, LocationSize::precise(4)), (i32* %b, LocationSize::precise(4)), (i32* %c, LocationSize::precise(4)), (i32* %d, LocationSize::precise(4))
define void @mergemust(i32* %c, i32* %d) {
  %a = alloca i32
  %b = alloca i32
  store i32 1, i32* %a
  store i32 2, i32* %b
  store i32 3, i32* %c
  store i32 4, i32* %d
  ret void
}
