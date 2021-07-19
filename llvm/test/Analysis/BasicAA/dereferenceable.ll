; RUN: opt -basic-aa -print-all-alias-modref-info -aa-eval < %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@G = global i32 0, align 4

define i64 @global_and_deref_arg_1(i64* dereferenceable(8) %arg) nofree nosync {
; CHECK:     Function: global_and_deref_arg_1: 2 pointers, 0 call sites
; CHECK-NEXT:  NoAlias:	i32* @G, i64* %arg
bb:
  store i64 1, i64* %arg, align 8
  store i32 0, i32* @G, align 4
  %tmp = load i64, i64* %arg, align 8
  ret i64 %tmp
}

define i32 @global_and_deref_arg_2(i32* dereferenceable(8) %arg) nofree nosync {
; CHECK:     Function: global_and_deref_arg_2: 2 pointers, 0 call sites
; CHECK-NEXT:  NoAlias:	i32* %arg, i32* @G
bb:
  store i32 1, i32* %arg, align 8
  store i32 0, i32* @G, align 4
  %tmp = load i32, i32* %arg, align 8
  ret i32 %tmp
}

define i32 @byval_and_deref_arg_1(i32* byval(i32) %obj, i64* dereferenceable(8) %arg) nofree nosync {
; CHECK:     Function: byval_and_deref_arg_1: 2 pointers, 0 call sites
; CHECK-NEXT:  NoAlias:	i32* %obj, i64* %arg
bb:
  store i32 1, i32* %obj, align 4
  store i64 0, i64* %arg, align 8
  %tmp = load i32, i32* %obj, align 4
  ret i32 %tmp
}

define i32 @byval_and_deref_arg_2(i32* byval(i32) %obj, i32* dereferenceable(8) %arg) nofree nosync {
; CHECK:     Function: byval_and_deref_arg_2: 2 pointers, 0 call sites
; CHECK-NEXT:  NoAlias:	i32* %arg, i32* %obj
bb:
  store i32 1, i32* %obj, align 4
  store i32 0, i32* %arg, align 8
  %tmp = load i32, i32* %obj, align 4
  ret i32 %tmp
}

declare dereferenceable(8) i32* @get_i32_deref8()
declare dereferenceable(8) i64* @get_i64_deref8()
declare void @unknown(i32*)

define i32 @local_and_deref_ret_1() {
; CHECK:     Function: local_and_deref_ret_1: 2 pointers, 2 call sites
; CHECK-NEXT:  NoAlias:	i32* %obj, i64* %ret
bb:
  %obj = alloca i32
  call void @unknown(i32* %obj)
  %ret = call dereferenceable(8) i64* @get_i64_deref8()
  store i32 1, i32* %obj, align 4
  store i64 0, i64* %ret, align 8
  %tmp = load i32, i32* %obj, align 4
  ret i32 %tmp
}

define i32 @local_and_deref_ret_2() {
; CHECK:     Function: local_and_deref_ret_2: 2 pointers, 2 call sites
; CHECK-NEXT:  NoAlias:	i32* %obj, i32* %ret
bb:
  %obj = alloca i32
  call void @unknown(i32* %obj)
  %ret = call dereferenceable(8) i32* @get_i32_deref8()
  store i32 1, i32* %obj, align 4
  store i32 0, i32* %ret, align 8
  %tmp = load i32, i32* %obj, align 4
  ret i32 %tmp
}


; Baseline tests, same as above but with 2 instead of 8 dereferenceable bytes.

define i64 @global_and_deref_arg_non_deref_1(i64* dereferenceable(2) %arg) nofree nosync {
; CHECK:     Function: global_and_deref_arg_non_deref_1: 2 pointers, 0 call sites
; CHECK-NEXT:  NoAlias:	i32* @G, i64* %arg
bb:
  store i64 1, i64* %arg, align 8
  store i32 0, i32* @G, align 4
  %tmp = load i64, i64* %arg, align 8
  ret i64 %tmp
}

define i32 @global_and_deref_arg_non_deref_2(i32* dereferenceable(2) %arg) nofree nosync {
; CHECK:     Function: global_and_deref_arg_non_deref_2: 2 pointers, 0 call sites
; Different result than above (see @global_and_deref_arg_2).
; CHECK-NEXT:  MayAlias:	i32* %arg, i32* @G
bb:
  store i32 1, i32* %arg, align 8
  store i32 0, i32* @G, align 4
  %tmp = load i32, i32* %arg, align 8
  ret i32 %tmp
}

define i32 @byval_and_deref_arg_non_deref_1(i32* byval(i32) %obj, i64* dereferenceable(2) %arg) nofree nosync {
; CHECK:     Function: byval_and_deref_arg_non_deref_1: 2 pointers, 0 call sites
; CHECK-NEXT:  NoAlias:	i32* %obj, i64* %arg
bb:
  store i32 1, i32* %obj, align 4
  store i64 0, i64* %arg, align 8
  %tmp = load i32, i32* %obj, align 4
  ret i32 %tmp
}

define i32 @byval_and_deref_arg_non_deref_2(i32* byval(i32) %obj, i32* dereferenceable(2) %arg) nofree nosync {
; CHECK:     Function: byval_and_deref_arg_non_deref_2: 2 pointers, 0 call sites
; CHECK-NEXT:  NoAlias:	i32* %arg, i32* %obj
bb:
  store i32 1, i32* %obj, align 4
  store i32 0, i32* %arg, align 8
  %tmp = load i32, i32* %obj, align 4
  ret i32 %tmp
}

declare dereferenceable(2) i32* @get_i32_deref2()
declare dereferenceable(2) i64* @get_i64_deref2()

define i32 @local_and_deref_ret_non_deref_1() {
; CHECK:     Function: local_and_deref_ret_non_deref_1: 2 pointers, 2 call sites
; CHECK-NEXT:  NoAlias:	i32* %obj, i64* %ret
bb:
  %obj = alloca i32
  call void @unknown(i32* %obj)
  %ret = call dereferenceable(2) i64* @get_i64_deref2()
  store i32 1, i32* %obj, align 4
  store i64 0, i64* %ret, align 8
  %tmp = load i32, i32* %obj, align 4
  ret i32 %tmp
}

define i32 @local_and_deref_ret_non_deref_2() {
; CHECK:     Function: local_and_deref_ret_non_deref_2: 2 pointers, 2 call sites
; Different result than above (see @local_and_deref_ret_2).
; CHECK-NEXT:  MayAlias:	i32* %obj, i32* %ret
bb:
  %obj = alloca i32
  call void @unknown(i32* %obj)
  %ret = call dereferenceable(2) i32* @get_i32_deref2()
  store i32 1, i32* %obj, align 4
  store i32 0, i32* %ret, align 8
  %tmp = load i32, i32* %obj, align 4
  ret i32 %tmp
}
