; RUN: opt -consthoist -S %s -o - | FileCheck %s --check-prefix=OPT
; RUN: opt -consthoist -S -consthoist-min-num-to-rebase=1 %s -o - | FileCheck %s --check-prefix=OPT --check-prefix=OPT-1
; RUN: opt -consthoist -S -consthoist-min-num-to-rebase=2 %s -o - | FileCheck %s --check-prefix=OPT --check-prefix=OPT-2
; RUN: opt -consthoist -S -consthoist-min-num-to-rebase=3 %s -o - | FileCheck %s --check-prefix=OPT --check-prefix=OPT-3

; RUN: llc -consthoist-min-num-to-rebase=2 %s -o - | FileCheck %s --check-prefix=LLC

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv6m-none-unknown-musleabi"

; Test that constant 0 and 1 of i1 type is NOT hoisted due low
; materializing cost.

; OPT-LABEL: avalon
; OPT: bb1:
; OPT: store i1 true
; OPT: bb2:
; OPT: store i1 false
; OPT: bb3:
; OPT: store i1 false
; OPT: store i1 false
; OPT-NOT: add

; LLC-LABEL: avalon
; LLC-DAG: movs r{{[0-9]+}}, #0
; LLC-DAG: movs r{{[0-9]+}}, #1
; LLC-NOT: add

@global = local_unnamed_addr global i1 undef, align 1
@global.0 = local_unnamed_addr global i1 undef, align 1

define void @avalon() #0 {
bb:
  switch i8 undef, label %bb5 [
    i8 0, label %bb1
    i8 -1, label %bb2
    i8 1, label %bb3
  ]

bb1:
  store i1 1, i1* @global, align 1
  unreachable

bb2:
  store i1 0, i1* @global, align 1
  unreachable

bb3:
  store i1 0, i1* @global.0, align 1
  store i1 0, i1* @global, align 1
  unreachable

bb5:
  ret void
}

; Test that for i8 type, constant -1 is not rebased since it's the only
; dependent of base constant -2.
; This test is also covered by r342898, see
; test/CodeGen/Thumb/consthoist-imm8-costs-1.ll

; OPT-2-LABEL: barney
; OPT-2: bb1:
; OPT-2: store i8 -1
; OPT-2: bb2:
; OPT-2: store i8 -2
; OPT-2: bb3:
; OPT-2: store i8 -2
; OPT-2: store i8 -2
; OPT-2-NOT: add

; LLC-LABEL: barney
; LLC-DAG: movs r{{[0-9]+}}, #254
; LLC-DAG: movs r{{[0-9]+}}, #255
; LLC-NOT: mvn
; LLC-NOT: add

@global.1 = local_unnamed_addr global i8 undef, align 1
@global.2 = local_unnamed_addr global i8 undef, align 1

define void @barney() #0 {
bb:
  switch i8 undef, label %bb5 [
    i8 0, label %bb1
    i8 -1, label %bb2
    i8 1, label %bb3
  ]

bb1:                                              ; preds = %bb
  store i8 -1, i8* @global.1, align 1
  unreachable

bb2:                                              ; preds = %bb
  store i8 -2, i8* @global.1, align 1
  unreachable

bb3:                                              ; preds = %bb
  store i8 -2, i8* @global.2, align 1
  store i8 -2, i8* @global.1, align 1
  unreachable

bb5:                                              ; preds = %bb
  ret void
}

; Test that for i16 type constant 65532 is not rebased if it's the only
; dependent of base constant 65531. Cost would be the same if rebased.
; If rebased, 3 two-byte instructions:
;   movs    r0, #4
;   mvns    r0, r0
;   adds    r0, r0, #1
; If NOT rebased, 1 two-byte instruction plus 1 four-byte CP entry:
;           ldr     r1, .LCPI2_3
;           ...
;   .LCPI2_3:
;           .long   65532

; OPT-LABEL: carla

; -consthoist-min-num-to-rebase=1, check that 65532 and single use of 65531
; in bb2 is rebased
; OPT-1: bb1:
; OPT-1: %[[C1:const[0-9]?]] = bitcast i16 -5 to i16
; OPT-1-NEXT: %const_mat = add i16 %[[C1]], 1
; OPT-1-NEXT: store i16 %const_mat, i16* @global.3, align 1
; OPT-1: bb2:
; OPT-1-NEXT: %[[C2:const[0-9]?]] = bitcast i16 -5 to i16
; OPT-1-NEXT: store i16 %[[C2]], i16* @global.3, align 1
; OPT-1: bb3:
; OPT-1-NEXT: %[[C3:const[0-9]?]] = bitcast i16 -5 to i16
; OPT-1-NEXT: store i16 %[[C3]], i16* @global.4, align 1
; OPT-1-NEXT: store i16 %[[C3]], i16* @global.3, align 1

; -consthoist-min-num-to-rebase=2, check that 65532 and single use of 65531
; in bb2 is not rebased
; OPT-2: bb1:
; OPT-2-NEXT: store i16 -4, i16* @global.3, align 1
; OPT-2: bb2:
; OPT-2-NEXT: store i16 -5, i16* @global.3, align 1
; OPT-2: bb3:
; OPT-2-NEXT:   %[[C4:const[0-9]?]] = bitcast i16 -5 to i16
; OPT-2-NEXT:   store i16 %[[C4]], i16* @global.4, align 1
; OPT-2-NEXT:   store i16 %[[C4]], i16* @global.3, align 1
; OPT-2-NOT: add

; -consthoist-min-num-to-rebase=3, check that dual uses of 65531 in bb3 are
; not rebase
; OPT-3: bb1:
; OPT-3-NEXT: store i16 -4, i16* @global.3, align 1
; OPT-3: bb2:
; OPT-3-NEXT: store i16 -5, i16* @global.3, align 1
; OPT-3: bb3:
; OPT-3-NEXT:   store i16 -5, i16* @global.4, align 1
; OPT-3-NEXT:   store i16 -5, i16* @global.3, align 1
; OPT-3-NOT: add
; OPT-3-NOT: bitcast

; LLC-LABEL: carla
; LLC-DAG: ldr r{{[0-9]+}}, .LCPI2_1
; LLC-DAG: ldr r{{[0-9]+}}, .LCPI2_3
; LLC-NOT: mvn
; LLC-NOT: add

@global.3 = local_unnamed_addr global i16 undef, align 2
@global.4 = local_unnamed_addr global i16 undef, align 2

define void @carla() {
bb:
  switch i8 undef, label %bb5 [
    i8 0, label %bb1
    i8 -1, label %bb2
    i8 1, label %bb3
  ]

bb1:                                              ; preds = %bb
  store i16 65532, i16* @global.3, align 1
  unreachable

bb2:                                              ; preds = %bb
  store i16 65531, i16* @global.3, align 1
  unreachable

bb3:                                              ; preds = %bb
  store i16 65531, i16* @global.4, align 1
  store i16 65531, i16* @global.3, align 1
  unreachable

bb5:                                              ; preds = %bb
  ret void
}
