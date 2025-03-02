; RUN: opt -aa-pipeline=basic-aa -print-all-alias-modref-info -passes=aa-eval < %s 2>&1 | FileCheck %s
; RUN: opt -aa-pipeline=basic-aa -print-all-alias-modref-info -passes=aa-eval -use-dereferenceable-at-point-semantics < %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@G = global i32 0, align 4

define i64 @global_and_deref_arg_1(ptr dereferenceable(8) %arg) nofree nosync {
; CHECK:     Function: global_and_deref_arg_1: 2 pointers, 0 call sites
; CHECK-NEXT:  NoAlias:	i64* %arg, i32* @G
bb:
  store i64 1, ptr %arg, align 8
  store i32 0, ptr @G, align 4
  %tmp = load i64, ptr %arg, align 8
  ret i64 %tmp
}

define i32 @global_and_deref_arg_2(ptr dereferenceable(8) %arg) nofree nosync {
; CHECK:     Function: global_and_deref_arg_2: 2 pointers, 0 call sites
; CHECK-NEXT:  NoAlias:	i32* %arg, i32* @G
bb:
  store i32 1, ptr %arg, align 8
  store i32 0, ptr @G, align 4
  %tmp = load i32, ptr %arg, align 8
  ret i32 %tmp
}

define i32 @byval_and_deref_arg_1(ptr byval(i32) %obj, ptr dereferenceable(8) %arg) nofree nosync {
; CHECK:     Function: byval_and_deref_arg_1: 2 pointers, 0 call sites
; CHECK-NEXT:  NoAlias:	i64* %arg, i32* %obj
bb:
  store i32 1, ptr %obj, align 4
  store i64 0, ptr %arg, align 8
  %tmp = load i32, ptr %obj, align 4
  ret i32 %tmp
}

define i32 @byval_and_deref_arg_2(ptr byval(i32) %obj, ptr dereferenceable(8) %arg) nofree nosync {
; CHECK:     Function: byval_and_deref_arg_2: 2 pointers, 0 call sites
; CHECK-NEXT:  NoAlias:	i32* %arg, i32* %obj
bb:
  store i32 1, ptr %obj, align 4
  store i32 0, ptr %arg, align 8
  %tmp = load i32, ptr %obj, align 4
  ret i32 %tmp
}

declare dereferenceable(8) ptr @get_i32_deref8()
declare dereferenceable(8) ptr @get_i64_deref8()
declare void @unknown(ptr)

define i32 @local_and_deref_ret_1() {
; CHECK:     Function: local_and_deref_ret_1: 2 pointers, 2 call sites
; CHECK-NEXT:  NoAlias:	i32* %obj, i64* %ret
bb:
  %obj = alloca i32
  call void @unknown(ptr %obj)
  %ret = call dereferenceable(8) ptr @get_i64_deref8()
  store i32 1, ptr %obj, align 4
  store i64 0, ptr %ret, align 8
  %tmp = load i32, ptr %obj, align 4
  ret i32 %tmp
}

define i32 @local_and_deref_ret_2() {
; CHECK:     Function: local_and_deref_ret_2: 2 pointers, 2 call sites
; CHECK-NEXT:  NoAlias:	i32* %obj, i32* %ret
bb:
  %obj = alloca i32
  call void @unknown(ptr %obj)
  %ret = call dereferenceable(8) ptr @get_i32_deref8()
  store i32 1, ptr %obj, align 4
  store i32 0, ptr %ret, align 8
  %tmp = load i32, ptr %obj, align 4
  ret i32 %tmp
}


; Baseline tests, same as above but with 2 instead of 8 dereferenceable bytes.

define i64 @global_and_deref_arg_non_deref_1(ptr dereferenceable(2) %arg) nofree nosync {
; CHECK:     Function: global_and_deref_arg_non_deref_1: 2 pointers, 0 call sites
; CHECK-NEXT:  NoAlias:	i64* %arg, i32* @G
bb:
  store i64 1, ptr %arg, align 8
  store i32 0, ptr @G, align 4
  %tmp = load i64, ptr %arg, align 8
  ret i64 %tmp
}

define i32 @global_and_deref_arg_non_deref_2(ptr dereferenceable(2) %arg) nofree nosync {
; CHECK:     Function: global_and_deref_arg_non_deref_2: 2 pointers, 0 call sites
; Different result than above (see @global_and_deref_arg_2).
; CHECK-NEXT:  MayAlias:	i32* %arg, i32* @G
bb:
  store i32 1, ptr %arg, align 8
  store i32 0, ptr @G, align 4
  %tmp = load i32, ptr %arg, align 8
  ret i32 %tmp
}

define i32 @byval_and_deref_arg_non_deref_1(ptr byval(i32) %obj, ptr dereferenceable(2) %arg) nofree nosync {
; CHECK:     Function: byval_and_deref_arg_non_deref_1: 2 pointers, 0 call sites
; CHECK-NEXT:  NoAlias:	i64* %arg, i32* %obj
bb:
  store i32 1, ptr %obj, align 4
  store i64 0, ptr %arg, align 8
  %tmp = load i32, ptr %obj, align 4
  ret i32 %tmp
}

define i32 @byval_and_deref_arg_non_deref_2(ptr byval(i32) %obj, ptr dereferenceable(2) %arg) nofree nosync {
; CHECK:     Function: byval_and_deref_arg_non_deref_2: 2 pointers, 0 call sites
; CHECK-NEXT:  NoAlias:	i32* %arg, i32* %obj
bb:
  store i32 1, ptr %obj, align 4
  store i32 0, ptr %arg, align 8
  %tmp = load i32, ptr %obj, align 4
  ret i32 %tmp
}

declare dereferenceable(2) ptr @get_i32_deref2()
declare dereferenceable(2) ptr @get_i64_deref2()

define i32 @local_and_deref_ret_non_deref_1() {
; CHECK:     Function: local_and_deref_ret_non_deref_1: 2 pointers, 2 call sites
; CHECK-NEXT:  NoAlias:	i32* %obj, i64* %ret
bb:
  %obj = alloca i32
  call void @unknown(ptr %obj)
  %ret = call dereferenceable(2) ptr @get_i64_deref2()
  store i32 1, ptr %obj, align 4
  store i64 0, ptr %ret, align 8
  %tmp = load i32, ptr %obj, align 4
  ret i32 %tmp
}

define i32 @local_and_deref_ret_non_deref_2() {
; CHECK:     Function: local_and_deref_ret_non_deref_2: 2 pointers, 2 call sites
; Different result than above (see @local_and_deref_ret_2).
; CHECK-NEXT:  MayAlias:	i32* %obj, i32* %ret
bb:
  %obj = alloca i32
  call void @unknown(ptr %obj)
  %ret = call dereferenceable(2) ptr @get_i32_deref2()
  store i32 1, ptr %obj, align 4
  store i32 0, ptr %ret, align 8
  %tmp = load i32, ptr %obj, align 4
  ret i32 %tmp
}
