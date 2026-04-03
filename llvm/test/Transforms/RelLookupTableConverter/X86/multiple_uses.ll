; RUN: opt < %s -passes=rel-lookup-table-converter -relocation-model=pic -S | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [5 x i8] c"zero\00", align 1
@.str.1 = private unnamed_addr constant [4 x i8] c"one\00", align 1
@.str.2 = private unnamed_addr constant [4 x i8] c"two\00", align 1

@switch.table.multiple_uses = private unnamed_addr constant [3 x ptr]
  [
    ptr @.str,
    ptr @.str.1,
    ptr @.str.2
  ], align 8

@switch.table.mixed_uses = private unnamed_addr constant [3 x ptr]
  [
    ptr @.str,
    ptr @.str.1,
    ptr @.str.2
  ], align 8

@switch.table.bitcast = private unnamed_addr constant [3 x ptr]
  [
    ptr @.str,
    ptr @.str.1,
    ptr @.str.2
  ], align 8

; CHECK:      @switch.table.multiple_uses.rel = private unnamed_addr constant [3 x i32] [
; CHECK-SAME:   i32 trunc (i64 sub (i64 ptrtoint (ptr @.str to i64), i64 ptrtoint (ptr @switch.table.multiple_uses.rel to i64)) to i32),
; CHECK-SAME:   i32 trunc (i64 sub (i64 ptrtoint (ptr @.str.1 to i64), i64 ptrtoint (ptr @switch.table.multiple_uses.rel to i64)) to i32),
; CHECK-SAME:   i32 trunc (i64 sub (i64 ptrtoint (ptr @.str.2 to i64), i64 ptrtoint (ptr @switch.table.multiple_uses.rel to i64)) to i32)
; CHECK-SAME: ], align 4

; CHECK: @switch.table.mixed_uses = private unnamed_addr constant [3 x ptr]
; CHECK: @switch.table.bitcast = private unnamed_addr constant [3 x ptr]

;; Here to allow loads to be used.
declare void @escape(ptr %p)

define void @test(i32 %idx) {
; CHECK-LABEL: @test(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[SHIFT:%.*]] = shl i32 %idx, 2
; CHECK-NEXT:    [[RES:%.*]] = call ptr @llvm.load.relative.i32(ptr @switch.table.multiple_uses.rel, i32 [[SHIFT]])
; CHECK-NEXT:    [[SHIFT2:%.*]] = shl i32 %idx, 2
; CHECK-NEXT:    [[RES2:%.*]] = call ptr @llvm.load.relative.i32(ptr @switch.table.multiple_uses.rel, i32 [[SHIFT2]])
;
entry:
  %gep1 = getelementptr inbounds [3 x ptr], ptr @switch.table.multiple_uses, i32 0, i32 %idx
  %load1 = load ptr, ptr %gep1, align 8
  %gep2 = getelementptr inbounds [3 x ptr], ptr @switch.table.multiple_uses, i32 0, i32 %idx
  %load2 = load ptr, ptr %gep2, align 8
  call void @escape(ptr %load1)
  call void @escape(ptr %load2)
  ret void
}

define void @accross_functions(i32 %idx) {
; CHECK-LABEL: @accross_functions(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[SHIFT:%.*]] = shl i32 %idx, 2
; CHECK-NEXT:    [[RES:%.*]] = call ptr @llvm.load.relative.i32(ptr @switch.table.multiple_uses.rel, i32 [[SHIFT]])
entry:
  %gep1 = getelementptr inbounds [3 x ptr], ptr @switch.table.multiple_uses, i32 0, i32 %idx
  %load1 = load ptr, ptr %gep1, align 8
  call void @escape(ptr %load1)
  ret void
}

;; Both the tables referenced below will not be transformed despite having
;; more than once use simply because their uses are more than GEPs+LOADs.

define void @test_mixed(i32 %idx, ptr %ptr) {
; CHECK-LABEL: @test_mixed(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[GEP:%.*]] = getelementptr inbounds [3 x ptr], ptr @switch.table.mixed_uses, i32 0, i32 %idx
; CHECK-NEXT:    [[LOAD:%.*]] = load ptr, ptr [[GEP]], align 8
; CHECK-NEXT:    call void @escape(ptr [[LOAD]])
; CHECK-NEXT:    store ptr @switch.table.mixed_uses, ptr %ptr, align 8
entry:
  %gep = getelementptr inbounds [3 x ptr], ptr @switch.table.mixed_uses, i32 0, i32 %idx
  %load = load ptr, ptr %gep, align 8
  call void @escape(ptr %load)
  store ptr @switch.table.mixed_uses, ptr %ptr, align 8
  ret void
}

define void @test_bitcast(i32 %idx) {
; CHECK-LABEL: @test_bitcast(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[GEP:%.*]] = getelementptr inbounds [3 x ptr], ptr @switch.table.bitcast, i32 0, i32 %idx
; CHECK-NEXT:    [[LOAD:%.*]] = load ptr, ptr [[GEP]], align 8
; CHECK-NEXT:    call void @escape(ptr [[LOAD]])
; CHECK-NEXT:    [[CAST:%.*]] = bitcast ptr @switch.table.bitcast to ptr
entry:
  %gep = getelementptr inbounds [3 x ptr], ptr @switch.table.bitcast, i32 0, i32 %idx
  %load = load ptr, ptr %gep, align 8
  call void @escape(ptr %load)
  %cast = bitcast ptr @switch.table.bitcast to ptr
  ret void
}
