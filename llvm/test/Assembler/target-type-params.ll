; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; Check support for basic target extension type properties

declare void @g1(target("atype", void))
declare void @g2(target("atype", i32))
declare void @g3(target("atype", i32, 0))
declare void @g4(target("atype", 0))
declare void @g5(target("atype", 0, 1, 2))
declare void @g6(target("atype", void, i32, float, {i32, bfloat}, 0, 1, 2))

;CHECK: declare void @g1(target("atype", void))
;CHECK: declare void @g2(target("atype", i32))
;CHECK: declare void @g3(target("atype", i32, 0))
;CHECK: declare void @g4(target("atype", 0))
;CHECK: declare void @g5(target("atype", 0, 1, 2))
;CHECK: declare void @g6(target("atype", void, i32, float, { i32, bfloat }, 0, 1, 2))
