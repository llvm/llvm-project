; RUN: llc < %s -asm-verbose=false -fast-isel -fast-isel-abort=1 -verify-machineinstrs | FileCheck %s

target triple = "wasm32-unknown-unknown"

; FastISel should not fold one of the add/sub operands into a load/store's
; offset when 'nuw' (no unsigned wrap) is not present, because the address
; calculation does not wrap. When there is an add/sub and nuw is not present, we
; bail out of FastISel.

@mylabel = external global ptr

; CHECK-LABEL: dont_fold_non_nuw_add_load:
; CHECK:       local.get  0
; CHECK-NEXT:  i32.const  2147483644
; CHECK-NEXT:  i32.add
; CHECK-NEXT:  i32.load  0
define i32 @dont_fold_non_nuw_add_load(ptr %p) {
  %q = ptrtoint ptr %p to i32
  %r = add i32 %q, 2147483644
  %s = inttoptr i32 %r to ptr
  %t = load i32, ptr %s
  ret i32 %t
}

; CHECK-LABEL: dont_fold_non_nuw_add_store:
; CHECK:       local.get  0
; CHECK-NEXT:  i32.const  2147483644
; CHECK-NEXT:  i32.add
; CHECK-NEXT:  i32.const  5
; CHECK-NEXT:  i32.store  0
define void @dont_fold_non_nuw_add_store(ptr %p) {
  %q = ptrtoint ptr %p to i32
  %r = add i32 %q, 2147483644
  %s = inttoptr i32 %r to ptr
  store i32 5, ptr %s
  ret void
}

; CHECK-LABEL: dont_fold_non_nuw_add_load_2:
; CHECK:       i32.const  mylabel
; CHECK-NEXT:  i32.const  -4
; CHECK-NEXT:  i32.add
; CHECK-NEXT:  i32.load  0
define i32 @dont_fold_non_nuw_add_load_2() {
  %t = load i32, ptr inttoptr (i32 add (i32 ptrtoint (ptr @mylabel to i32), i32 -4) to ptr), align 4
  ret i32 %t
}

; CHECK-LABEL: dont_fold_non_nuw_add_store_2:
; CHECK:       i32.const  mylabel
; CHECK-NEXT:  i32.const  -4
; CHECK-NEXT:  i32.add
; CHECK-NEXT:  i32.const  5
; CHECK-NEXT:  i32.store  0
define void @dont_fold_non_nuw_add_store_2() {
  store i32 5, ptr inttoptr (i32 add (i32 ptrtoint (ptr @mylabel to i32), i32 -4) to ptr), align 4
  ret void
}

; CHECK-LABEL: dont_fold_non_nuw_sub_load:
; CHECK:       local.get  0
; CHECK-NEXT:  i32.const  -2147483644
; CHECK-NEXT:  i32.sub
; CHECK-NEXT:  i32.load  0
define i32 @dont_fold_non_nuw_sub_load(ptr %p) {
  %q = ptrtoint ptr %p to i32
  %r = sub i32 %q, -2147483644
  %s = inttoptr i32 %r to ptr
  %t = load i32, ptr %s
  ret i32 %t
}

; CHECK-LABEL: dont_fold_non_nuw_sub_store:
; CHECK:       local.get  0
; CHECK-NEXT:  i32.const  -2147483644
; CHECK-NEXT:  i32.sub
; CHECK-NEXT:  i32.const  5
; CHECK-NEXT:  i32.store  0
define void @dont_fold_non_nuw_sub_store(ptr %p) {
  %q = ptrtoint ptr %p to i32
  %r = sub i32 %q, -2147483644
  %s = inttoptr i32 %r to ptr
  store i32 5, ptr %s
  ret void
}

; CHECK-LABEL: dont_fold_non_nuw_sub_load_2:
; CHECK:       i32.const  mylabel
; CHECK-NEXT:  i32.const  4
; CHECK-NEXT:  i32.sub
; CHECK-NEXT:  i32.load  0
define i32 @dont_fold_non_nuw_sub_load_2() {
  %t = load i32, ptr inttoptr (i32 sub (i32 ptrtoint (ptr @mylabel to i32), i32 4) to ptr), align 4
 ret i32 %t
}

; CHECK-LABEL: dont_fold_non_nuw_sub_store_2:
; CHECK:       i32.const  mylabel
; CHECK-NEXT:  i32.const  4
; CHECK-NEXT:  i32.sub
; CHECK-NEXT:  i32.const  5
; CHECK-NEXT:  i32.store  0
define void @dont_fold_non_nuw_sub_store_2() {
  store i32 5, ptr inttoptr (i32 sub (i32 ptrtoint (ptr @mylabel to i32), i32 4) to ptr), align 4
  ret void
}
