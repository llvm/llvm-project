// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK-LABEL: func @ops
// CHECK-SAME: (%[[I32:.*]]: !llvm.i32, %[[FLOAT:.*]]: !llvm.float, %[[I8PTR1:.*]]: !llvm<"i8*">, %[[I8PTR2:.*]]: !llvm<"i8*">, %[[BOOL:.*]]: !llvm.i1)
func @ops(%arg0: !llvm.i32, %arg1: !llvm.float,
          %arg2: !llvm<"i8*">, %arg3: !llvm<"i8*">,
          %arg4: !llvm.i1) {
// Integer arithmetic binary operations.
//
// CHECK: {{.*}} = llvm.add %[[I32]], %[[I32]] : !llvm.i32
// CHECK: {{.*}} = llvm.sub %[[I32]], %[[I32]] : !llvm.i32
// CHECK: {{.*}} = llvm.mul %[[I32]], %[[I32]] : !llvm.i32
// CHECK: {{.*}} = llvm.udiv %[[I32]], %[[I32]] : !llvm.i32
// CHECK: {{.*}} = llvm.sdiv %[[I32]], %[[I32]] : !llvm.i32
// CHECK: {{.*}} = llvm.urem %[[I32]], %[[I32]] : !llvm.i32
// CHECK: {{.*}} = llvm.srem %[[I32]], %[[I32]] : !llvm.i32
// CHECK: {{.*}} = llvm.icmp "ne" %[[I32]], %[[I32]] : !llvm.i32
  %0 = llvm.add %arg0, %arg0 : !llvm.i32
  %1 = llvm.sub %arg0, %arg0 : !llvm.i32
  %2 = llvm.mul %arg0, %arg0 : !llvm.i32
  %3 = llvm.udiv %arg0, %arg0 : !llvm.i32
  %4 = llvm.sdiv %arg0, %arg0 : !llvm.i32
  %5 = llvm.urem %arg0, %arg0 : !llvm.i32
  %6 = llvm.srem %arg0, %arg0 : !llvm.i32
  %7 = llvm.icmp "ne" %arg0, %arg0 : !llvm.i32

// Floating point binary operations.
//
// CHECK: {{.*}} = llvm.fadd %[[FLOAT]], %[[FLOAT]] : !llvm.float
// CHECK: {{.*}} = llvm.fsub %[[FLOAT]], %[[FLOAT]] : !llvm.float
// CHECK: {{.*}} = llvm.fmul %[[FLOAT]], %[[FLOAT]] : !llvm.float
// CHECK: {{.*}} = llvm.fdiv %[[FLOAT]], %[[FLOAT]] : !llvm.float
// CHECK: {{.*}} = llvm.frem %[[FLOAT]], %[[FLOAT]] : !llvm.float
  %8 = llvm.fadd %arg1, %arg1 : !llvm.float
  %9 = llvm.fsub %arg1, %arg1 : !llvm.float
  %10 = llvm.fmul %arg1, %arg1 : !llvm.float
  %11 = llvm.fdiv %arg1, %arg1 : !llvm.float
  %12 = llvm.frem %arg1, %arg1 : !llvm.float

// Memory-related operations.
//
// CHECK-NEXT:  %[[ALLOCA:.*]] = llvm.alloca %[[I32]] x !llvm.double : (!llvm.i32) -> !llvm<"double*">
// CHECK-NEXT:  %[[GEP:.*]] = llvm.getelementptr %[[ALLOCA]][%[[I32]], %[[I32]]] : (!llvm<"double*">, !llvm.i32, !llvm.i32) -> !llvm<"double*">
// CHECK-NEXT:  %[[VALUE:.*]] = llvm.load %[[GEP]] : !llvm<"double*">
// CHECK-NEXT:  llvm.store %[[VALUE]], %[[ALLOCA]] : !llvm<"double*">
// CHECK-NEXT:  %{{.*}} = llvm.bitcast %[[ALLOCA]] : !llvm<"double*"> to !llvm<"i64*">
  %13 = llvm.alloca %arg0 x !llvm.double : (!llvm.i32) -> !llvm<"double*">
  %14 = llvm.getelementptr %13[%arg0, %arg0] : (!llvm<"double*">, !llvm.i32, !llvm.i32) -> !llvm<"double*">
  %15 = llvm.load %14 : !llvm<"double*">
  llvm.store %15, %13 : !llvm<"double*">
  %16 = llvm.bitcast %13 : !llvm<"double*"> to !llvm<"i64*">

// Function call-related operations.
//
// CHECK: %[[STRUCT:.*]] = llvm.call @foo(%[[I32]]) : (!llvm.i32) -> !llvm<"{ i32, double, i32 }">
// CHECK: %[[VALUE:.*]] = llvm.extractvalue %[[STRUCT]][0] : !llvm<"{ i32, double, i32 }">
// CHECK: %[[NEW_STRUCT:.*]] = llvm.insertvalue %[[VALUE]], %[[STRUCT]][2] : !llvm<"{ i32, double, i32 }">
// CHECK: %[[FUNC:.*]] = llvm.mlir.constant(@foo) : !llvm<"{ i32, double, i32 } (i32)*">
// CHECK: %{{.*}} = llvm.call %[[FUNC]](%[[I32]]) : (!llvm.i32) -> !llvm<"{ i32, double, i32 }">
  %17 = llvm.call @foo(%arg0) : (!llvm.i32) -> !llvm<"{ i32, double, i32 }">
  %18 = llvm.extractvalue %17[0] : !llvm<"{ i32, double, i32 }">
  %19 = llvm.insertvalue %18, %17[2] : !llvm<"{ i32, double, i32 }">
  %20 = llvm.mlir.constant(@foo) : !llvm<"{ i32, double, i32 } (i32)*">
  %21 = llvm.call %20(%arg0) : (!llvm.i32) -> !llvm<"{ i32, double, i32 }">


// Terminator operations and their successors.
//
// CHECK: llvm.br ^[[BB1:.*]]
  llvm.br ^bb1

// CHECK: ^[[BB1]]
^bb1:
// CHECK: llvm.cond_br %7, ^[[BB2:.*]], ^[[BB1]]
  llvm.cond_br %7, ^bb2, ^bb1

// CHECK: ^[[BB2]]
^bb2:
// CHECK: %{{.*}} = llvm.mlir.undef : !llvm<"{ i32, double, i32 }">
// CHECK: %{{.*}} = llvm.mlir.constant(42 : i64) : !llvm.i47
  %22 = llvm.mlir.undef : !llvm<"{ i32, double, i32 }">
  %23 = llvm.mlir.constant(42) : !llvm.i47

// Misc operations.
// CHECK: %{{.*}} = llvm.select %{{.*}}, %{{.*}}, %{{.*}} : !llvm.i1, !llvm.i32
  %24 = llvm.select %7, %0, %1 : !llvm.i1, !llvm.i32

// Integer to pointer and pointer to integer conversions.
//
// CHECK: %[[PTR:.*]] = llvm.inttoptr %[[I32]] : !llvm.i32 to !llvm<"i32*">
// CHECK: %{{.*}} = llvm.ptrtoint %[[PTR]] : !llvm<"i32*"> to !llvm.i32
  %25 = llvm.inttoptr %arg0 : !llvm.i32 to !llvm<"i32*">
  %26 = llvm.ptrtoint %25 : !llvm<"i32*"> to !llvm.i32

// Extended and Quad floating point
//
// CHECK: %{{.*}} = llvm.fpext %[[FLOAT]] : !llvm.float to !llvm.x86_fp80
// CHECK: %{{.*}} = llvm.fpext %[[FLOAT]] : !llvm.float to !llvm.fp128
  %27 = llvm.fpext %arg1 : !llvm.float to !llvm.x86_fp80
  %28 = llvm.fpext %arg1 : !llvm.float to !llvm.fp128

// CHECK: %{{.*}} = llvm.fneg %[[FLOAT]] : !llvm.float
  %29 = llvm.fneg %arg1 : !llvm.float

// CHECK: "llvm.intr.sin"(%[[FLOAT]]) : (!llvm.float) -> !llvm.float
  %30 = "llvm.intr.sin"(%arg1) : (!llvm.float) -> !llvm.float

// CHECK: "llvm.intr.pow"(%[[FLOAT]], %[[FLOAT]]) : (!llvm.float, !llvm.float) -> !llvm.float
  %31 = "llvm.intr.pow"(%arg1, %arg1) : (!llvm.float, !llvm.float) -> !llvm.float

// CHECK: "llvm.intr.bitreverse"(%{{.*}}) : (!llvm.i32) -> !llvm.i32
  %32 = "llvm.intr.bitreverse"(%arg0) : (!llvm.i32) -> !llvm.i32

// CHECK: "llvm.intr.ctpop"(%{{.*}}) : (!llvm.i32) -> !llvm.i32
  %33 = "llvm.intr.ctpop"(%arg0) : (!llvm.i32) -> !llvm.i32

// CHECK: "llvm.intr.memcpy"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm<"i8*">, !llvm<"i8*">, !llvm.i32, !llvm.i1) -> ()
  "llvm.intr.memcpy"(%arg2, %arg3, %arg0, %arg4) : (!llvm<"i8*">, !llvm<"i8*">, !llvm.i32, !llvm.i1) -> ()

// CHECK: "llvm.intr.memcpy"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm<"i8*">, !llvm<"i8*">, !llvm.i32, !llvm.i1) -> ()
  "llvm.intr.memcpy"(%arg2, %arg3, %arg0, %arg4) : (!llvm<"i8*">, !llvm<"i8*">, !llvm.i32, !llvm.i1) -> ()

// CHECK: %[[SZ:.*]] = llvm.mlir.constant
  %sz = llvm.mlir.constant(10: i64) : !llvm.i64
// CHECK: "llvm.intr.memcpy.inline"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm<"i8*">, !llvm<"i8*">, !llvm.i64, !llvm.i1) -> ()
  "llvm.intr.memcpy.inline"(%arg2, %arg3, %sz, %arg4) : (!llvm<"i8*">, !llvm<"i8*">, !llvm.i64, !llvm.i1) -> ()

// CHECK:  llvm.return
  llvm.return
}

// An larger self-contained function.
// CHECK-LABEL: func @foo(%{{.*}}: !llvm.i32) -> !llvm<"{ i32, double, i32 }"> {
func @foo(%arg0: !llvm.i32) -> !llvm<"{ i32, double, i32 }"> {
// CHECK:  %[[V0:.*]] = llvm.mlir.constant(3 : i64) : !llvm.i32
// CHECK:  %[[V1:.*]] = llvm.mlir.constant(3 : i64) : !llvm.i32
// CHECK:  %[[V2:.*]] = llvm.mlir.constant(4.200000e+01 : f64) : !llvm.double
// CHECK:  %[[V3:.*]] = llvm.mlir.constant(4.200000e+01 : f64) : !llvm.double
// CHECK:  %[[V4:.*]] = llvm.add %[[V0]], %[[V1]] : !llvm.i32
// CHECK:  %[[V5:.*]] = llvm.mul %[[V4]], %[[V1]] : !llvm.i32
// CHECK:  %[[V6:.*]] = llvm.fadd %[[V2]], %[[V3]] : !llvm.double
// CHECK:  %[[V7:.*]] = llvm.fsub %[[V3]], %[[V6]] : !llvm.double
// CHECK:  %[[V8:.*]] = llvm.mlir.constant(1 : i64) : !llvm.i1
// CHECK:  llvm.cond_br %[[V8]], ^[[BB1:.*]](%[[V4]] : !llvm.i32), ^[[BB2:.*]](%[[V4]] : !llvm.i32)
  %0 = llvm.mlir.constant(3) : !llvm.i32
  %1 = llvm.mlir.constant(3) : !llvm.i32
  %2 = llvm.mlir.constant(4.200000e+01) : !llvm.double
  %3 = llvm.mlir.constant(4.200000e+01) : !llvm.double
  %4 = llvm.add %0, %1 : !llvm.i32
  %5 = llvm.mul %4, %1 : !llvm.i32
  %6 = llvm.fadd %2, %3 : !llvm.double
  %7 = llvm.fsub %3, %6 : !llvm.double
  %8 = llvm.mlir.constant(1) : !llvm.i1
  llvm.cond_br %8, ^bb1(%4 : !llvm.i32), ^bb2(%4 : !llvm.i32)

// CHECK:^[[BB1]](%[[V9:.*]]: !llvm.i32):
// CHECK:  %[[V10:.*]] = llvm.call @foo(%[[V9]]) : (!llvm.i32) -> !llvm<"{ i32, double, i32 }">
// CHECK:  %[[V11:.*]] = llvm.extractvalue %[[V10]][0] : !llvm<"{ i32, double, i32 }">
// CHECK:  %[[V12:.*]] = llvm.extractvalue %[[V10]][1] : !llvm<"{ i32, double, i32 }">
// CHECK:  %[[V13:.*]] = llvm.extractvalue %[[V10]][2] : !llvm<"{ i32, double, i32 }">
// CHECK:  %[[V14:.*]] = llvm.mlir.undef : !llvm<"{ i32, double, i32 }">
// CHECK:  %[[V15:.*]] = llvm.insertvalue %[[V5]], %[[V14]][0] : !llvm<"{ i32, double, i32 }">
// CHECK:  %[[V16:.*]] = llvm.insertvalue %[[V7]], %[[V15]][1] : !llvm<"{ i32, double, i32 }">
// CHECK:  %[[V17:.*]] = llvm.insertvalue %[[V11]], %[[V16]][2] : !llvm<"{ i32, double, i32 }">
// CHECK:  llvm.return %[[V17]] : !llvm<"{ i32, double, i32 }">
^bb1(%9: !llvm.i32):
  %10 = llvm.call @foo(%9) : (!llvm.i32) -> !llvm<"{ i32, double, i32 }">
  %11 = llvm.extractvalue %10[0] : !llvm<"{ i32, double, i32 }">
  %12 = llvm.extractvalue %10[1] : !llvm<"{ i32, double, i32 }">
  %13 = llvm.extractvalue %10[2] : !llvm<"{ i32, double, i32 }">
  %14 = llvm.mlir.undef : !llvm<"{ i32, double, i32 }">
  %15 = llvm.insertvalue %5, %14[0] : !llvm<"{ i32, double, i32 }">
  %16 = llvm.insertvalue %7, %15[1] : !llvm<"{ i32, double, i32 }">
  %17 = llvm.insertvalue %11, %16[2] : !llvm<"{ i32, double, i32 }">
  llvm.return %17 : !llvm<"{ i32, double, i32 }">

// CHECK:^[[BB2]](%[[V18:.*]]: !llvm.i32):
// CHECK:  %[[V19:.*]] = llvm.mlir.undef : !llvm<"{ i32, double, i32 }">
// CHECK:  %[[V20:.*]] = llvm.insertvalue %[[V18]], %[[V19]][0] : !llvm<"{ i32, double, i32 }">
// CHECK:  %[[V21:.*]] = llvm.insertvalue %[[V7]], %[[V20]][1] : !llvm<"{ i32, double, i32 }">
// CHECK:  %[[V22:.*]] = llvm.insertvalue %[[V5]], %[[V21]][2] : !llvm<"{ i32, double, i32 }">
// CHECK:  llvm.return %[[V22]] : !llvm<"{ i32, double, i32 }">
^bb2(%18: !llvm.i32):
  %19 = llvm.mlir.undef : !llvm<"{ i32, double, i32 }">
  %20 = llvm.insertvalue %18, %19[0] : !llvm<"{ i32, double, i32 }">
  %21 = llvm.insertvalue %7, %20[1] : !llvm<"{ i32, double, i32 }">
  %22 = llvm.insertvalue %5, %21[2] : !llvm<"{ i32, double, i32 }">
  llvm.return %22 : !llvm<"{ i32, double, i32 }">
}

// CHECK-LABEL: @casts
// CHECK-SAME: (%[[I32:.*]]: !llvm.i32, %[[I64:.*]]: !llvm.i64, %[[V4I32:.*]]: !llvm<"<4 x i32>">, %[[V4I64:.*]]: !llvm<"<4 x i64>">, %[[I32PTR:.*]]: !llvm<"i32*">)
func @casts(%arg0: !llvm.i32, %arg1: !llvm.i64, %arg2: !llvm<"<4 x i32>">,
            %arg3: !llvm<"<4 x i64>">, %arg4: !llvm<"i32*">) {
// CHECK:  = llvm.sext %[[I32]] : !llvm.i32 to !llvm.i56
  %0 = llvm.sext %arg0 : !llvm.i32 to !llvm.i56
// CHECK:  = llvm.zext %[[I32]] : !llvm.i32 to !llvm.i64
  %1 = llvm.zext %arg0 : !llvm.i32 to !llvm.i64
// CHECK:  = llvm.trunc %[[I64]] : !llvm.i64 to !llvm.i56
  %2 = llvm.trunc %arg1 : !llvm.i64 to !llvm.i56
// CHECK:  = llvm.sext %[[V4I32]] : !llvm<"<4 x i32>"> to !llvm<"<4 x i56>">
  %3 = llvm.sext %arg2 : !llvm<"<4 x i32>"> to !llvm<"<4 x i56>">
// CHECK:  = llvm.zext %[[V4I32]] : !llvm<"<4 x i32>"> to !llvm<"<4 x i64>">
  %4 = llvm.zext %arg2 : !llvm<"<4 x i32>"> to !llvm<"<4 x i64>">
// CHECK:  = llvm.trunc %[[V4I64]] : !llvm<"<4 x i64>"> to !llvm<"<4 x i56>">
  %5 = llvm.trunc %arg3 : !llvm<"<4 x i64>"> to !llvm<"<4 x i56>">
// CHECK:  = llvm.sitofp %[[I32]] : !llvm.i32 to !llvm.float
  %6 = llvm.sitofp %arg0 : !llvm.i32 to !llvm.float
// CHECK: %[[FLOAT:.*]] = llvm.uitofp %[[I32]] : !llvm.i32 to !llvm.float
  %7 = llvm.uitofp %arg0 : !llvm.i32 to !llvm.float
// CHECK:  = llvm.fptosi %[[FLOAT]] : !llvm.float to !llvm.i32
  %8 = llvm.fptosi %7 : !llvm.float to !llvm.i32
// CHECK:  = llvm.fptoui %[[FLOAT]] : !llvm.float to !llvm.i32
  %9 = llvm.fptoui %7 : !llvm.float to !llvm.i32
// CHECK:  = llvm.addrspacecast %[[I32PTR]] : !llvm<"i32*"> to !llvm<"i32 addrspace(2)*">
  %10 = llvm.addrspacecast %arg4 : !llvm<"i32*"> to !llvm<"i32 addrspace(2)*">
  llvm.return
}

// CHECK-LABEL: @vect
func @vect(%arg0: !llvm<"<4 x float>">, %arg1: !llvm.i32, %arg2: !llvm.float) {
// CHECK:  = llvm.extractelement {{.*}} : !llvm<"<4 x float>">
  %0 = llvm.extractelement %arg0[%arg1 : !llvm.i32] : !llvm<"<4 x float>">
// CHECK:  = llvm.insertelement {{.*}} : !llvm<"<4 x float>">
  %1 = llvm.insertelement %arg2, %arg0[%arg1 : !llvm.i32] : !llvm<"<4 x float>">
// CHECK:  = llvm.shufflevector {{.*}} [0 : i32, 0 : i32, 0 : i32, 0 : i32, 7 : i32] : !llvm<"<4 x float>">, !llvm<"<4 x float>">
  %2 = llvm.shufflevector %arg0, %arg0 [0 : i32, 0 : i32, 0 : i32, 0 : i32, 7 : i32] : !llvm<"<4 x float>">, !llvm<"<4 x float>">
// CHECK:  = llvm.mlir.constant(dense<1.000000e+00> : vector<4xf32>) : !llvm<"<4 x float>">
  %3 = llvm.mlir.constant(dense<1.0> : vector<4xf32>) : !llvm<"<4 x float>">
  return
}

// CHECK-LABEL: @alloca
func @alloca(%size : !llvm.i64) {
  // CHECK: llvm.alloca %{{.*}} x !llvm.i32 : (!llvm.i64) -> !llvm<"i32*">
  llvm.alloca %size x !llvm.i32 {alignment = 0} : (!llvm.i64) -> (!llvm<"i32*">)
  // CHECK: llvm.alloca %{{.*}} x !llvm.i32 {alignment = 8 : i64} : (!llvm.i64) -> !llvm<"i32*">
  llvm.alloca %size x !llvm.i32 {alignment = 8} : (!llvm.i64) -> (!llvm<"i32*">)
  llvm.return
}

// CHECK-LABEL: @null
func @null() {
  // CHECK: llvm.mlir.null : !llvm<"i8*">
  %0 = llvm.mlir.null : !llvm<"i8*">
  // CHECK: llvm.mlir.null : !llvm<"{ void (i32, void ()*)*, i64 }*">
  %1 = llvm.mlir.null : !llvm<"{void(i32, void()*)*, i64}*">
  llvm.return
}

// CHECK-LABEL: @atomicrmw
func @atomicrmw(%ptr : !llvm<"float*">, %val : !llvm.float) {
  // CHECK: llvm.atomicrmw fadd %{{.*}}, %{{.*}} unordered : !llvm.float
  %0 = llvm.atomicrmw fadd %ptr, %val unordered : !llvm.float
  llvm.return
}

// CHECK-LABEL: @cmpxchg
func @cmpxchg(%ptr : !llvm<"float*">, %cmp : !llvm.float, %new : !llvm.float) {
  // CHECK: llvm.cmpxchg %{{.*}}, %{{.*}}, %{{.*}} acq_rel monotonic : !llvm.float
  %0 = llvm.cmpxchg %ptr, %cmp, %new acq_rel monotonic : !llvm.float
  llvm.return
}

llvm.mlir.global external constant @_ZTIi() : !llvm<"i8*">
llvm.func @bar(!llvm<"i8*">, !llvm<"i8*">, !llvm<"i8*">)
llvm.func @__gxx_personality_v0(...) -> !llvm.i32

// CHECK-LABEL: @invokeLandingpad
llvm.func @invokeLandingpad() -> !llvm.i32 attributes { personality = @__gxx_personality_v0 } {
// CHECK: %[[a0:.*]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK: %{{.*}} = llvm.mlir.constant(3 : i32) : !llvm.i32
// CHECK: %[[a2:.*]] = llvm.mlir.constant("\01") : !llvm<"[1 x i8]">
// CHECK: %[[a3:.*]] = llvm.mlir.null : !llvm<"i8**">
// CHECK: %[[a4:.*]] = llvm.mlir.null : !llvm<"i8*">
// CHECK: %[[a5:.*]] = llvm.mlir.addressof @_ZTIi : !llvm<"i8**">
// CHECK: %[[a6:.*]] = llvm.bitcast %[[a5]] : !llvm<"i8**"> to !llvm<"i8*">
// CHECK: %[[a7:.*]] = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK: %[[a8:.*]] = llvm.alloca %[[a7]] x !llvm.i8 : (!llvm.i32) -> !llvm<"i8*">
// CHECK: %{{.*}} = llvm.invoke @foo(%[[a7]]) to ^[[BB2:.*]] unwind ^[[BB1:.*]] : (!llvm.i32) -> !llvm<"{ i32, double, i32 }">
  %0 = llvm.mlir.constant(0 : i32) : !llvm.i32
  %1 = llvm.mlir.constant(3 : i32) : !llvm.i32
  %2 = llvm.mlir.constant("\01") : !llvm<"[1 x i8]">
  %3 = llvm.mlir.null : !llvm<"i8**">
  %4 = llvm.mlir.null : !llvm<"i8*">
  %5 = llvm.mlir.addressof @_ZTIi : !llvm<"i8**">
  %6 = llvm.bitcast %5 : !llvm<"i8**"> to !llvm<"i8*">
  %7 = llvm.mlir.constant(1 : i32) : !llvm.i32
  %8 = llvm.alloca %7 x !llvm.i8 : (!llvm.i32) -> !llvm<"i8*">
  %9 = llvm.invoke @foo(%7) to ^bb2 unwind ^bb1 : (!llvm.i32) -> !llvm<"{ i32, double, i32 }">

// CHECK: ^[[BB1]]:
// CHECK:   %[[lp:.*]] = llvm.landingpad cleanup (catch %[[a3]] : !llvm<"i8**">) (catch %[[a6]] : !llvm<"i8*">) (filter %[[a2]] : !llvm<"[1 x i8]">) : !llvm<"{ i8*, i32 }">
// CHECK:   llvm.resume %[[lp]] : !llvm<"{ i8*, i32 }">
^bb1:
  %10 = llvm.landingpad cleanup (catch %3 : !llvm<"i8**">) (catch %6 : !llvm<"i8*">) (filter %2 : !llvm<"[1 x i8]">) : !llvm<"{ i8*, i32 }">
  llvm.resume %10 : !llvm<"{ i8*, i32 }">

// CHECK: ^[[BB2]]:
// CHECK:   llvm.return %[[a7]] : !llvm.i32
^bb2:
  llvm.return %7 : !llvm.i32

// CHECK: ^[[BB3:.*]]:
// CHECK:   llvm.invoke @bar(%[[a8]], %[[a6]], %[[a4]]) to ^[[BB2]] unwind ^[[BB1]] : (!llvm<"i8*">, !llvm<"i8*">, !llvm<"i8*">) -> ()
^bb3:
  llvm.invoke @bar(%8, %6, %4) to ^bb2 unwind ^bb1 : (!llvm<"i8*">, !llvm<"i8*">, !llvm<"i8*">) -> ()

// CHECK: ^[[BB4:.*]]:
// CHECK:   llvm.return %[[a0]] : !llvm.i32
^bb4:
  llvm.return %0 : !llvm.i32
}

// CHECK-LABEL: @useFreezeOp
func @useFreezeOp(%arg0: !llvm.i32) {
  // CHECK:  = llvm.freeze %[[ARG0:.*]] : !llvm.i32
  %0 = llvm.freeze %arg0 : !llvm.i32
  // CHECK: %[[x:.*]] = llvm.mlir.undef : !llvm.i8
  %1 = llvm.mlir.undef : !llvm.i8
  // CHECK:  = llvm.freeze %[[x]] : !llvm.i8
  %2 = llvm.freeze %1 : !llvm.i8
  return
}

// CHECK-LABEL: @useFenceInst
func @useFenceInst() {
  // CHECK:  syncscope("agent") seq_cst
  llvm.fence syncscope("agent") seq_cst
  // CHECK:  seq_cst
  llvm.fence syncscope("") seq_cst
  // CHECK:  release
  llvm.fence release
  return
}
