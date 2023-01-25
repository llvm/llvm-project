; RUN: llvm-dis -expand-constant-exprs < %s.bc | FileCheck %s

@g = extern_weak global i32
@g2 = extern_weak global i32

define i64 @test_cast() {
; CHECK-LABEL: define i64 @test_cast() {
; CHECK-NEXT: %constexpr = ptrtoint ptr @g to i64
; CHECK-NEXT: ret i64 %constexpr
  ret i64 ptrtoint (ptr @g to i64)
}

define i1 @test_icmp() {
; CHECK-LABEL: define i1 @test_icmp() {
; CHECK-NEXT: %constexpr = ptrtoint ptr @g to i64
; CHECK-NEXT: %constexpr1 = icmp ne i64 %constexpr, 0
; CHECK-NEXT: ret i1 %constexpr1
  ret i1 icmp ne (i64 ptrtoint (ptr @g to i64), i64 0)
}

define i32 @test_select() {
; CHECK-LABEL: define i32 @test_select() {
; CHECK-NEXT: %constexpr = ptrtoint ptr @g to i64
; CHECK-NEXT: %constexpr1 = icmp ne i64 %constexpr, 0
; CHECK-NEXT: %constexpr2 = select i1 %constexpr1, i32 1, i32 2
; CHECK-NEXT: ret i32 %constexpr2
  ret i32 select (i1 icmp ne (i64 ptrtoint (ptr @g to i64), i64 0), i32 1, i32 2)
}

define i8 @test_extractelement() {
; CHECK-LABEL: define i8 @test_extractelement() {
; CHECK-NEXT: %constexpr = ptrtoint ptr @g to i64
; CHECK-NEXT: %constexpr1 = icmp ne i64 %constexpr, 0
; CHECK-NEXT: %constexpr2 = select i1 %constexpr1, <2 x i8> zeroinitializer, <2 x i8> <i8 0, i8 1>
; CHECK-NEXT: %constexpr3 = extractelement <2 x i8> %constexpr2, i32 0
; CHECK-NEXT: ret i8 %constexpr3
  ret i8 extractelement (<2 x i8> select (i1 icmp ne (i64 ptrtoint (ptr @g to i64), i64 0), <2 x i8> zeroinitializer, <2 x i8> <i8 0, i8 1>), i32 0)
}

define <2 x i8> @test_insertelement() {
; CHECK-LABEL: define <2 x i8> @test_insertelement() {
; CHECK-NEXT: %constexpr = ptrtoint ptr @g to i32
; CHECK-NEXT: %constexpr1 = insertelement <2 x i8> poison, i8 42, i32 %constexpr
; CHECK-NEXT: ret <2 x i8> %constexpr1
  ret <2 x i8> insertelement (<2 x i8> poison, i8 42, i32 ptrtoint (ptr @g to i32))
}

define double @test_fneg() {
; CHECK-LABEL: define double @test_fneg() {
; CHECK-NEXT: %constexpr = ptrtoint ptr @g to i64
; CHECK-NEXT: %constexpr1 = bitcast i64 %constexpr to double
; CHECK-NEXT: %constexpr2 = fneg double %constexpr1
  ret double fneg (double bitcast (i64 ptrtoint (ptr @g to i64) to double))
}

define i64 @test_flags() {
; CHECK-LABEL: define i64 @test_flags() {
; CHECK-NEXT: %constexpr = ptrtoint ptr @g to i64
; CHECK-NEXT: %constexpr1 = add nuw i64 %constexpr, 1
; CHECK-NEXT: ret i64 %constexpr1
  ret i64 add nuw (i64 ptrtoint (ptr @g to i64), i64 1)
}

define <3 x i64> @test_vector() {
; CHECK-LABEL: define <3 x i64> @test_vector() {
; CHECK-NEXT: %constexpr = ptrtoint ptr @g to i64
; CHECK-NEXT: %constexpr.ins = insertelement <3 x i64> poison, i64 5, i32 0
; CHECK-NEXT: %constexpr.ins1 = insertelement <3 x i64> %constexpr.ins, i64 %constexpr, i32 1
; CHECK-NEXT: %constexpr.ins2 = insertelement <3 x i64> %constexpr.ins1, i64 7, i32 2
  ret <3 x i64> <i64 5, i64 ptrtoint (ptr @g to i64), i64 7>
}

define [3 x i64] @test_array() {
; CHECK-LABEL: define [3 x i64] @test_array() {
; CHECK-NEXT: %constexpr = ptrtoint ptr @g to i64
; CHECK-NEXT: %constexpr.ins = insertvalue [3 x i64] poison, i64 5, 0
; CHECK-NEXT: %constexpr.ins1 = insertvalue [3 x i64] %constexpr.ins, i64 %constexpr, 1
; CHECK-NEXT: %constexpr.ins2 = insertvalue [3 x i64] %constexpr.ins1, i64 7, 2
  ret [3 x i64] [i64 5, i64 ptrtoint (ptr @g to i64), i64 7]
}

define { i64, i64, i64 } @test_struct() {
; CHECK-LABEL: define { i64, i64, i64 } @test_struct() {
; CHECK-NEXT: %constexpr = ptrtoint ptr @g to i64
; CHECK-NEXT: %constexpr.ins = insertvalue { i64, i64, i64 } poison, i64 5, 0
; CHECK-NEXT: %constexpr.ins1 = insertvalue { i64, i64, i64 } %constexpr.ins, i64 %constexpr, 1
; CHECK-NEXT: %constexpr.ins2 = insertvalue { i64, i64, i64 } %constexpr.ins1, i64 7, 2
  ret { i64, i64, i64 } {i64 5, i64 ptrtoint (ptr @g to i64), i64 7}
}

define i64 @test_reused_expr() {
; CHECK-LABEL: define i64 @test_reused_expr() {
; CHECK-NEXT: %constexpr = ptrtoint ptr @g to i64
; CHECK-NEXT: %constexpr1 = add i64 %constexpr, %constexpr
; CHECK-NEXT: ret i64 %constexpr1
  ret i64 add (i64 ptrtoint (ptr @g to i64), i64 ptrtoint (ptr @g to i64))
}

define i64 @test_multiple_expanded_operands() {
; CHECK-LABEL: define i64 @test_multiple_expanded_operands() {
; CHECK-NEXT: %constexpr = ptrtoint ptr @g to i64
; CHECK-NEXT: %constexpr1 = ptrtoint ptr @g2 to i64
; CHECK-NEXT: %constexpr2 = add i64 %constexpr, %constexpr1
; CHECK-NEXT: ret i64 %constexpr2
  ret i64 add (i64 ptrtoint (ptr @g to i64), i64 ptrtoint (ptr @g2 to i64))
}

define i64 @test_mid_block(i64 %arg) {
; CHECK-LABEL: define i64 @test_mid_block(i64 %arg) {
; CHECK-NEXT: %x = mul i64 %arg, 3
; CHECK-NEXT: %constexpr = ptrtoint ptr @g to i64
; CHECK-NEXT: %add = add i64 %x, %constexpr
; CHECK-NEXT: ret i64 %add
  %x = mul i64 %arg, 3
  %add = add i64 %x, ptrtoint (ptr @g to i64)
  ret i64 %add
}

define i64 @test_phi_non_critical_edge_block_before(i1 %c) {
; CHECK-LABEL: define i64 @test_phi_non_critical_edge_block_before(i1 %c) {
; CHECK: entry:
; CHECK-NEXT: br i1 %c, label %if, label %join
; CHECK: if:
; CHECK-NEXT: br label %phi.constexpr
; CHECK: phi.constexpr:
; CHECK-NEXT: %constexpr = ptrtoint ptr @g to i64
; CHECK-NEXT: br label %join
; CHECK: join:
; CHECK-NEXT: %phi = phi i64 [ 0, %entry ], [ %constexpr, %phi.constexpr ]
; CHECK-NEXT: ret i64 %phi
entry:
  br i1 %c, label %if, label %join

if:
  br label %join

join:
  %phi = phi i64 [ 0, %entry ], [ ptrtoint (ptr @g to i64), %if ]
  ret i64 %phi
}

define i64 @test_phi_non_critical_edge_block_after(i1 %c) {
; CHECK-LABEL: define i64 @test_phi_non_critical_edge_block_after(i1 %c) {
; CHECK: entry:
; CHECK-NEXT: br i1 %c, label %if, label %join
; CHECK: phi.constexpr:
; CHECK-NEXT: %constexpr = ptrtoint ptr @g to i64
; CHECK-NEXT: br label %join
; CHECK: join:
; CHECK-NEXT: %phi = phi i64 [ 0, %entry ], [ %constexpr, %phi.constexpr ]
; CHECK-NEXT: ret i64 %phi
; CHECK: if:
; CHECK-NEXT: br label %phi.constexpr
entry:
  br i1 %c, label %if, label %join

join:
  %phi = phi i64 [ 0, %entry ], [ ptrtoint (ptr @g to i64), %if ]
  ret i64 %phi

if:
  br label %join
}

define i64 @test_phi_critical_edge(i1 %c) {
; CHECK-LABEL: define i64 @test_phi_critical_edge(i1 %c) {
; CHECK: entry:
; CHECK-NEXT: br i1 %c, label %if, label %phi.constexpr
; CHECK: if:
; CHECK-NEXT: br label %join
; CHECK: phi.constexpr:
; CHECK-NEXT: %constexpr = ptrtoint ptr @g to i64
; CHECK-NEXT: br label %join
; CHECK: join:
; CHECK-NEXT: %phi = phi i64 [ %constexpr, %phi.constexpr ], [ 0, %if ]
; CHECK-NEXT: ret i64 %phi
entry:
  br i1 %c, label %if, label %join

if:
  br label %join

join:
  %phi = phi i64 [ ptrtoint (ptr @g to i64), %entry ], [ 0, %if ]
  ret i64 %phi
}

define i64 @test_phi_multiple_nodes(i1 %c) {
; CHECK-LABEL: define i64 @test_phi_multiple_nodes(i1 %c) {
; CHECK: entry:
; CHECK-NEXT: br i1 %c, label %if, label %join
; CHECK: if:
; CHECK-NEXT: br label %phi.constexpr
; CHECK: phi.constexpr:
; CHECK-NEXT: %constexpr = ptrtoint ptr @g to i64
; CHECK-NEXT: %constexpr2 = ptrtoint ptr @g2 to i64
; CHECK-NEXT: br label %join
; CHECK: join:
; CHECK-NEXT: %phi = phi i64 [ 0, %entry ], [ %constexpr, %phi.constexpr ]
; CHECK-NEXT: %phi2 = phi i64 [ 0, %entry ], [ %constexpr2, %phi.constexpr ]
; CHECK-NEXT: ret i64 %phi
entry:
  br i1 %c, label %if, label %join

if:
  br label %join

join:
  %phi = phi i64 [ 0, %entry ], [ ptrtoint (ptr @g to i64), %if ]
  %phi2 = phi i64 [ 0, %entry ], [ ptrtoint (ptr @g2 to i64), %if ]
  ret i64 %phi
}


define i64 @test_phi_multiple_identical_predecessors(i32 %x) {
; CHECK-LABEL: define i64 @test_phi_multiple_identical_predecessors(i32 %x) {
; CHECK: entry:
; CHECK-NEXT: switch i32 %x, label %default [
; CHECK-NEXT:   i32 0, label %phi.constexpr
; CHECK-NEXT:   i32 1, label %phi.constexpr
; CHECK-NEXT: ]
; CHECK: default:
; CHECK-NEXT: br label %join
; CHECK: phi.constexpr:
; CHECK-NEXT: %constexpr = ptrtoint ptr @g to i64
; CHECK-NEXT: br label %join
; CHECK: join:
; CHECK-NEXT: %phi = phi i64 [ %constexpr, %phi.constexpr ], [ %constexpr, %phi.constexpr ], [ 0, %default ]
; CHECK-NEXT: ret i64 %phi
entry:
  switch i32 %x, label %default [
    i32 0, label %join
    i32 1, label %join
  ]

default:
  br label %join

join:
  %phi = phi i64 [ ptrtoint (ptr @g to i64), %entry ], [ ptrtoint (ptr @g to i64), %entry ], [ 0, %default ]
  ret i64 %phi
}
