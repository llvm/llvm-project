; RUN: opt -passes=debugify -S -o - < %s | FileCheck %s
; RUN: opt -passes=debugify,check-debugify -S -o - < %s | \
; RUN:   FileCheck %s -implicit-check-not="CheckModuleDebugify: FAIL"

; Debugify used to abort on instructions producing scalable vectors: the
; synthetic DIBasicType size was computed by implicitly converting a scalable
; TypeSize to a scalar, which is a fatal error.

define <vscale x 1 x i32> @scalable_vec(<vscale x 1 x i32> %x) {
; CHECK-LABEL: define <vscale x 1 x i32> @scalable_vec
; CHECK: #dbg_value(<vscale x 1 x i32> %add
  %add = add <vscale x 1 x i32> %x, %x
  ret <vscale x 1 x i32> %add
}

; A struct type containing a scalable vector also has a scalable size.
define { <vscale x 2 x i16>, i32 } @scalable_struct(<vscale x 2 x i16> %v) {
; CHECK-LABEL: define { <vscale x 2 x i16>, i32 } @scalable_struct
; CHECK: #dbg_value({ <vscale x 2 x i16>, i32 } %s
  %s = insertvalue { <vscale x 2 x i16>, i32 } poison, <vscale x 2 x i16> %v, 0
  ret { <vscale x 2 x i16>, i32 } %s
}

; Scalable types are given a zero-sized sentinel type, like unsized types.
; CHECK-DAG: DIBasicType(name: "ty0"
