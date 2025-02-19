; RUN: mlir-translate --import-llvm %s -split-input-file | FileCheck %s

@foo_alias = alias ptr, ptr @callee

; CHECK: llvm.mlir.alias external @foo_alias : !llvm.ptr {
; CHECK:   %[[ADDR:.*]] = llvm.mlir.addressof @callee : !llvm.ptr
; CHECK:   llvm.return %[[ADDR]] : !llvm.ptr
; CHECK: }

define internal ptr @callee() {
entry:
  ret ptr null
}

; // -----

@zed = global i32 42
@foo = alias i32, ptr @zed
@foo2 = alias i16, ptr @zed

; CHECK: llvm.mlir.alias external @foo : i32 {
; CHECK:   %[[ADDR:.*]] = llvm.mlir.addressof @zed : !llvm.ptr
; CHECK:   llvm.return %[[ADDR]] : !llvm.ptr
; CHECK: }
; CHECK: llvm.mlir.alias external @foo2 : i16 {
; CHECK:   %[[ADDR:.*]] = llvm.mlir.addressof @zed : !llvm.ptr
; CHECK:   llvm.return %[[ADDR]] : !llvm.ptr
; CHECK: }

; // -----

@v1 = global i32 0
@a3 = alias i32, addrspacecast (ptr @v1 to ptr addrspace(2))
; CHECK: llvm.mlir.alias external @a3 : i32 {
; CHECK:   %[[ADDR:.*]] = llvm.mlir.addressof @v1 : !llvm.ptr
; CHECK:   %[[CASTED_ADDR:.*]] = llvm.addrspacecast %[[ADDR]] : !llvm.ptr to !llvm.ptr<2>
; CHECK:   llvm.return %[[CASTED_ADDR]] : !llvm.ptr<2>
; CHECK: }

; // -----

@some_name = constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr null, ptr null] }
@vtable = alias { [3 x ptr] }, ptr @some_name

; CHECK: llvm.mlir.alias external @vtable : !llvm.struct<(array<3 x ptr>)> {
; CHECK:   %[[ADDR:.*]] = llvm.mlir.addressof @some_name : !llvm.ptr
; CHECK:   llvm.return %[[ADDR]] : !llvm.ptr
; CHECK: }

; // -----

@glob.private = private constant [32 x i32] zeroinitializer
@glob = linkonce_odr hidden alias [32 x i32], inttoptr (i64 add (i64 ptrtoint (ptr @glob.private to i64), i64 1234) to ptr)

; CHECK: llvm.mlir.alias linkonce_odr hidden @glob {dso_local} : !llvm.array<32 x i32> {
; CHECK: %[[CST:.*]] = llvm.mlir.constant(1234 : i64) : i64
; CHECK: %[[ADDR:.*]] = llvm.mlir.addressof @glob.private : !llvm.ptr
; CHECK: %[[PTRTOINT:.*]] = llvm.ptrtoint %[[ADDR]] : !llvm.ptr to i64
; CHECK: %[[INTTOPTR:.*]] = llvm.add %[[PTRTOINT]], %[[CST]] : i64
; CHECK: %[[RET:.*]] = llvm.inttoptr %[[INTTOPTR]] : i64 to !llvm.ptr
; CHECK: llvm.return %[[RET]] : !llvm.ptr

; // -----

@g1 = private global i32 0
@g2 = internal constant ptr @a1
@g3 = internal constant ptr @a2
@a1 = private alias i32, ptr @g1
@a2 = private alias ptr, ptr @a1


; CHECK: llvm.mlir.global internal constant @g2() {addr_space = 0 : i32, dso_local} : !llvm.ptr {
; CHECK-NEXT:   %[[ADDR:.*]] = llvm.mlir.addressof @a1 : !llvm.ptr
; CHECK-NEXT:   llvm.return %[[ADDR]] : !llvm.ptr
; CHECK-NEXT: }

; CHECK: llvm.mlir.global internal constant @g3() {addr_space = 0 : i32, dso_local} : !llvm.ptr {
; CHECK-NEXT:   %[[ADDR:.*]] = llvm.mlir.addressof @a2 : !llvm.ptr
; CHECK-NEXT:   llvm.return %[[ADDR]] : !llvm.ptr
; CHECK-NEXT: }

; CHECK: llvm.mlir.alias private @a1 {dso_local} : i32 {
; CHECK-NEXT:   %[[ADDR:.*]] = llvm.mlir.addressof @g1 : !llvm.ptr
; CHECK-NEXT:   llvm.return %[[ADDR]] : !llvm.ptr
; CHECK-NEXT: }

; CHECK: llvm.mlir.alias private @a2 {dso_local} : !llvm.ptr {
; CHECK-NEXT:   %[[ADDR:.*]] = llvm.mlir.addressof @a1 : !llvm.ptr
; CHECK-NEXT:   llvm.return %[[ADDR]] : !llvm.ptr
; CHECK-NEXT: }
