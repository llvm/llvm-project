// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK-LABEL: func @ops(%arg0: !llvm.i32, %arg1: !llvm.float)
func @ops(%arg0 : !llvm.i32, %arg1 : !llvm.float) {
// Integer arithmetic binary operations.
//
// CHECK-NEXT:  %0 = llvm.add %arg0, %arg0 : !llvm.i32
// CHECK-NEXT:  %1 = llvm.sub %arg0, %arg0 : !llvm.i32
// CHECK-NEXT:  %2 = llvm.mul %arg0, %arg0 : !llvm.i32
// CHECK-NEXT:  %3 = llvm.udiv %arg0, %arg0 : !llvm.i32
// CHECK-NEXT:  %4 = llvm.sdiv %arg0, %arg0 : !llvm.i32
// CHECK-NEXT:  %5 = llvm.urem %arg0, %arg0 : !llvm.i32
// CHECK-NEXT:  %6 = llvm.srem %arg0, %arg0 : !llvm.i32
// CHECK-NEXT:  %7 = llvm.icmp "ne" %arg0, %arg0 : !llvm.i32
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
// CHECK-NEXT:  %8 = llvm.fadd %arg1, %arg1 : !llvm.float
// CHECK-NEXT:  %9 = llvm.fsub %arg1, %arg1 : !llvm.float
// CHECK-NEXT:  %10 = llvm.fmul %arg1, %arg1 : !llvm.float
// CHECK-NEXT:  %11 = llvm.fdiv %arg1, %arg1 : !llvm.float
// CHECK-NEXT:  %12 = llvm.frem %arg1, %arg1 : !llvm.float
  %8 = llvm.fadd %arg1, %arg1 : !llvm.float
  %9 = llvm.fsub %arg1, %arg1 : !llvm.float
  %10 = llvm.fmul %arg1, %arg1 : !llvm.float
  %11 = llvm.fdiv %arg1, %arg1 : !llvm.float
  %12 = llvm.frem %arg1, %arg1 : !llvm.float

// Memory-related operations.
//
// CHECK-NEXT:  %13 = llvm.alloca %arg0 x !llvm.double : (!llvm.i32) -> !llvm<"double*">
// CHECK-NEXT:  %14 = llvm.getelementptr %13[%arg0, %arg0] : (!llvm<"double*">, !llvm.i32, !llvm.i32) -> !llvm<"double*">
// CHECK-NEXT:  %15 = llvm.load %14 : !llvm<"double*">
// CHECK-NEXT:  llvm.store %15, %13 : !llvm<"double*">
// CHECK-NEXT:  %16 = llvm.bitcast %13 : !llvm<"double*"> to !llvm<"i64*">
  %13 = llvm.alloca %arg0 x !llvm.double : (!llvm.i32) -> !llvm<"double*">
  %14 = llvm.getelementptr %13[%arg0, %arg0] : (!llvm<"double*">, !llvm.i32, !llvm.i32) -> !llvm<"double*">
  %15 = llvm.load %14 : !llvm<"double*">
  llvm.store %15, %13 : !llvm<"double*">
  %16 = llvm.bitcast %13 : !llvm<"double*"> to !llvm<"i64*">

// Function call-related operations.
//
// CHECK-NEXT:  %17 = llvm.call @foo(%arg0) : (!llvm.i32) -> !llvm<"{ i32, double, i32 }">
// CHECK-NEXT:  %18 = llvm.extractvalue %17[0] : !llvm<"{ i32, double, i32 }">
// CHECK-NEXT:  %19 = llvm.insertvalue %18, %17[2] : !llvm<"{ i32, double, i32 }">
// CHECK-NEXT:  %20 = llvm.mlir.constant(@foo) : !llvm<"{ i32, double, i32 } (i32)*">
// CHECK-NEXT:  %21 = llvm.call %20(%arg0) : (!llvm.i32) -> !llvm<"{ i32, double, i32 }">
  %17 = llvm.call @foo(%arg0) : (!llvm.i32) -> !llvm<"{ i32, double, i32 }">
  %18 = llvm.extractvalue %17[0] : !llvm<"{ i32, double, i32 }">
  %19 = llvm.insertvalue %18, %17[2] : !llvm<"{ i32, double, i32 }">
  %20 = llvm.mlir.constant(@foo) : !llvm<"{ i32, double, i32 } (i32)*">
  %21 = llvm.call %20(%arg0) : (!llvm.i32) -> !llvm<"{ i32, double, i32 }">


// Terminator operations and their successors.
//
// CHECK: llvm.br ^bb1
  llvm.br ^bb1

^bb1:
// CHECK: llvm.cond_br %7, ^bb2, ^bb1
  llvm.cond_br %7, ^bb2, ^bb1

^bb2:
// CHECK:       %22 = llvm.mlir.undef : !llvm<"{ i32, double, i32 }">
// CHECK-NEXT:  %23 = llvm.mlir.constant(42 : i64) : !llvm.i47
  %22 = llvm.mlir.undef : !llvm<"{ i32, double, i32 }">
  %23 = llvm.mlir.constant(42) : !llvm.i47

// Misc operations.
// CHECK:       %24 = llvm.select %7, %0, %1 : !llvm.i1, !llvm.i32
  %24 = llvm.select %7, %0, %1 : !llvm.i1, !llvm.i32

// Integer to pointer and pointer to integer conversions.
//
// CHECK:       %25 = llvm.inttoptr %arg0 : !llvm.i32 to !llvm<"i32*">
// CHECK-NEXT:  %26 = llvm.ptrtoint %25 : !llvm<"i32*"> to !llvm.i32
  %25 = llvm.inttoptr %arg0 : !llvm.i32 to !llvm<"i32*">
  %26 = llvm.ptrtoint %25 : !llvm<"i32*"> to !llvm.i32

// Extended and Quad floating point
//
// CHECK:       %27 = llvm.fpext %arg1 : !llvm.float to !llvm.x86_fp80
// CHECK-NEXT:  %28 = llvm.fpext %arg1 : !llvm.float to !llvm.fp128
  %27 = llvm.fpext %arg1 : !llvm.float to !llvm.x86_fp80
  %28 = llvm.fpext %arg1 : !llvm.float to !llvm.fp128

// CHECK:  %29 = llvm.fneg %arg1 : !llvm.float
  %29 = llvm.fneg %arg1 : !llvm.float

// CHECK: "llvm.intr.sin"(%arg1) : (!llvm.float) -> !llvm.float
  %30 = "llvm.intr.sin"(%arg1) : (!llvm.float) -> !llvm.float

// CHECK: "llvm.intr.pow"(%arg1, %arg1) : (!llvm.float, !llvm.float) -> !llvm.float
  %31 = "llvm.intr.pow"(%arg1, %arg1) : (!llvm.float, !llvm.float) -> !llvm.float

// CHECK: "llvm.intr.bitreverse"(%{{.*}}) : (!llvm.i32) -> !llvm.i32
  %32 = "llvm.intr.bitreverse"(%arg0) : (!llvm.i32) -> !llvm.i32

// CHECK: "llvm.intr.ctpop"(%{{.*}}) : (!llvm.i32) -> !llvm.i32
  %33 = "llvm.intr.ctpop"(%arg0) : (!llvm.i32) -> !llvm.i32

// CHECK:  llvm.return
  llvm.return
}

// An larger self-contained function.
// CHECK-LABEL:func @foo(%arg0: !llvm.i32) -> !llvm<"{ i32, double, i32 }"> {
func @foo(%arg0: !llvm.i32) -> !llvm<"{ i32, double, i32 }"> {
// CHECK-NEXT:  %0 = llvm.mlir.constant(3 : i64) : !llvm.i32
// CHECK-NEXT:  %1 = llvm.mlir.constant(3 : i64) : !llvm.i32
// CHECK-NEXT:  %2 = llvm.mlir.constant(4.200000e+01 : f64) : !llvm.double
// CHECK-NEXT:  %3 = llvm.mlir.constant(4.200000e+01 : f64) : !llvm.double
// CHECK-NEXT:  %4 = llvm.add %0, %1 : !llvm.i32
// CHECK-NEXT:  %5 = llvm.mul %4, %1 : !llvm.i32
// CHECK-NEXT:  %6 = llvm.fadd %2, %3 : !llvm.double
// CHECK-NEXT:  %7 = llvm.fsub %3, %6 : !llvm.double
// CHECK-NEXT:  %8 = llvm.mlir.constant(1 : i64) : !llvm.i1
// CHECK-NEXT:  llvm.cond_br %8, ^bb1(%4 : !llvm.i32), ^bb2(%4 : !llvm.i32)
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

// CHECK-NEXT:^bb1(%9: !llvm.i32):
// CHECK-NEXT:  %10 = llvm.call @foo(%9) : (!llvm.i32) -> !llvm<"{ i32, double, i32 }">
// CHECK-NEXT:  %11 = llvm.extractvalue %10[0] : !llvm<"{ i32, double, i32 }">
// CHECK-NEXT:  %12 = llvm.extractvalue %10[1] : !llvm<"{ i32, double, i32 }">
// CHECK-NEXT:  %13 = llvm.extractvalue %10[2] : !llvm<"{ i32, double, i32 }">
// CHECK-NEXT:  %14 = llvm.mlir.undef : !llvm<"{ i32, double, i32 }">
// CHECK-NEXT:  %15 = llvm.insertvalue %5, %14[0] : !llvm<"{ i32, double, i32 }">
// CHECK-NEXT:  %16 = llvm.insertvalue %7, %15[1] : !llvm<"{ i32, double, i32 }">
// CHECK-NEXT:  %17 = llvm.insertvalue %11, %16[2] : !llvm<"{ i32, double, i32 }">
// CHECK-NEXT:  llvm.return %17 : !llvm<"{ i32, double, i32 }">
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

// CHECK-NEXT:^bb2(%18: !llvm.i32):	// pred: ^bb0
// CHECK-NEXT:  %19 = llvm.mlir.undef : !llvm<"{ i32, double, i32 }">
// CHECK-NEXT:  %20 = llvm.insertvalue %18, %19[0] : !llvm<"{ i32, double, i32 }">
// CHECK-NEXT:  %21 = llvm.insertvalue %7, %20[1] : !llvm<"{ i32, double, i32 }">
// CHECK-NEXT:  %22 = llvm.insertvalue %5, %21[2] : !llvm<"{ i32, double, i32 }">
// CHECK-NEXT:  llvm.return %22 : !llvm<"{ i32, double, i32 }">
^bb2(%18: !llvm.i32):	// pred: ^bb0
  %19 = llvm.mlir.undef : !llvm<"{ i32, double, i32 }">
  %20 = llvm.insertvalue %18, %19[0] : !llvm<"{ i32, double, i32 }">
  %21 = llvm.insertvalue %7, %20[1] : !llvm<"{ i32, double, i32 }">
  %22 = llvm.insertvalue %5, %21[2] : !llvm<"{ i32, double, i32 }">
  llvm.return %22 : !llvm<"{ i32, double, i32 }">
}

// CHECK-LABEL: @casts
func @casts(%arg0: !llvm.i32, %arg1: !llvm.i64, %arg2: !llvm<"<4 x i32>">,
            %arg3: !llvm<"<4 x i64>">, %arg4: !llvm<"i32*">) {
// CHECK-NEXT:  = llvm.sext %arg0 : !llvm.i32 to !llvm.i56
  %0 = llvm.sext %arg0 : !llvm.i32 to !llvm.i56
// CHECK-NEXT:  = llvm.zext %arg0 : !llvm.i32 to !llvm.i64
  %1 = llvm.zext %arg0 : !llvm.i32 to !llvm.i64
// CHECK-NEXT:  = llvm.trunc %arg1 : !llvm.i64 to !llvm.i56
  %2 = llvm.trunc %arg1 : !llvm.i64 to !llvm.i56
// CHECK-NEXT:  = llvm.sext %arg2 : !llvm<"<4 x i32>"> to !llvm<"<4 x i56>">
  %3 = llvm.sext %arg2 : !llvm<"<4 x i32>"> to !llvm<"<4 x i56>">
// CHECK-NEXT:  = llvm.zext %arg2 : !llvm<"<4 x i32>"> to !llvm<"<4 x i64>">
  %4 = llvm.zext %arg2 : !llvm<"<4 x i32>"> to !llvm<"<4 x i64>">
// CHECK-NEXT:  = llvm.trunc %arg3 : !llvm<"<4 x i64>"> to !llvm<"<4 x i56>">
  %5 = llvm.trunc %arg3 : !llvm<"<4 x i64>"> to !llvm<"<4 x i56>">
// CHECK-NEXT:  = llvm.sitofp %arg0 : !llvm.i32 to !llvm.float
  %6 = llvm.sitofp %arg0 : !llvm.i32 to !llvm.float
// CHECK-NEXT:  = llvm.uitofp %arg0 : !llvm.i32 to !llvm.float
  %7 = llvm.uitofp %arg0 : !llvm.i32 to !llvm.float
// CHECK-NEXT:  = llvm.fptosi %7 : !llvm.float to !llvm.i32
  %8 = llvm.fptosi %7 : !llvm.float to !llvm.i32
// CHECK-NEXT:  = llvm.fptoui %7 : !llvm.float to !llvm.i32
  %9 = llvm.fptoui %7 : !llvm.float to !llvm.i32
// CHECK-NEXT:  = llvm.addrspacecast %arg4 : !llvm<"i32*"> to !llvm<"i32 addrspace(2)*">
  %10 = llvm.addrspacecast %arg4 : !llvm<"i32*"> to !llvm<"i32 addrspace(2)*">
  llvm.return
}

// CHECK-LABEL: @vect
func @vect(%arg0: !llvm<"<4 x float>">, %arg1: !llvm.i32, %arg2: !llvm.float) {
// CHECK-NEXT:  = llvm.extractelement {{.*}} : !llvm<"<4 x float>">
  %0 = llvm.extractelement %arg0[%arg1 : !llvm.i32] : !llvm<"<4 x float>">
// CHECK-NEXT:  = llvm.insertelement {{.*}} : !llvm<"<4 x float>">
  %1 = llvm.insertelement %arg2, %arg0[%arg1 : !llvm.i32] : !llvm<"<4 x float>">
// CHECK-NEXT:  = llvm.shufflevector {{.*}} [0 : i32, 0 : i32, 0 : i32, 0 : i32, 7 : i32] : !llvm<"<4 x float>">, !llvm<"<4 x float>">
  %2 = llvm.shufflevector %arg0, %arg0 [0 : i32, 0 : i32, 0 : i32, 0 : i32, 7 : i32] : !llvm<"<4 x float>">, !llvm<"<4 x float>">
// CHECK-NEXT:  = llvm.mlir.constant(dense<1.000000e+00> : vector<4xf32>) : !llvm<"<4 x float>">
  %3 = llvm.mlir.constant(dense<1.0> : vector<4xf32>) : !llvm<"<4 x float>">
  return
}

// CHECK-LABEL: @alloca
func @alloca(%size : !llvm.i64) {
  //      CHECK: llvm.alloca %{{.*}} x !llvm.i32 : (!llvm.i64) -> !llvm<"i32*">
  llvm.alloca %size x !llvm.i32 {alignment = 0} : (!llvm.i64) -> (!llvm<"i32*">)
  // CHECK-NEXT: llvm.alloca %{{.*}} x !llvm.i32 {alignment = 8 : i64} : (!llvm.i64) -> !llvm<"i32*">
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
// CHECK-NEXT: %[[a0:[0-9]+]] = llvm.mlir.constant(0 : i32) : !llvm.i32
// CHECK-NEXT: %{{[0-9]+}} = llvm.mlir.constant(3 : i32) : !llvm.i32
// CHECK-NEXT: %[[a2:[0-9]+]] = llvm.mlir.constant("\01") : !llvm<"[1 x i8]">
// CHECK-NEXT: %[[a3:[0-9]+]] = llvm.mlir.null : !llvm<"i8**">
// CHECK-NEXT: %[[a4:[0-9]+]] = llvm.mlir.null : !llvm<"i8*">
// CHECK-NEXT: %[[a5:[0-9]+]] = llvm.mlir.addressof @_ZTIi : !llvm<"i8**">
// CHECK-NEXT: %[[a6:[0-9]+]] = llvm.bitcast %[[a5]] : !llvm<"i8**"> to !llvm<"i8*">
// CHECK-NEXT: %[[a7:[0-9]+]] = llvm.mlir.constant(1 : i32) : !llvm.i32
// CHECK-NEXT: %[[a8:[0-9]+]] = llvm.alloca %[[a7]] x !llvm.i8 : (!llvm.i32) -> !llvm<"i8*">
// CHECK-NEXT: %{{[0-9]+}} = llvm.invoke @foo(%[[a7]]) to ^bb2 unwind ^bb1 : (!llvm.i32) -> !llvm<"{ i32, double, i32 }">
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

// CHECK-NEXT: ^bb1:
// CHECK-NEXT:   %[[lp:[0-9]+]] = llvm.landingpad cleanup (catch %[[a3]] : !llvm<"i8**">) (catch %[[a6]] : !llvm<"i8*">) (filter %[[a2]] : !llvm<"[1 x i8]">) : !llvm<"{ i8*, i32 }">
// CHECK-NEXT:   llvm.resume %[[lp]] : !llvm<"{ i8*, i32 }">
^bb1:
  %10 = llvm.landingpad cleanup (catch %3 : !llvm<"i8**">) (catch %6 : !llvm<"i8*">) (filter %2 : !llvm<"[1 x i8]">) : !llvm<"{ i8*, i32 }">
  llvm.resume %10 : !llvm<"{ i8*, i32 }">

// CHECK-NEXT: ^bb2:
// CHECK-NEXT:   llvm.return %[[a7]] : !llvm.i32
^bb2:
  llvm.return %7 : !llvm.i32

// CHECK-NEXT: ^bb3:
// CHECK-NEXT:   llvm.invoke @bar(%[[a8]], %[[a6]], %[[a4]]) to ^bb2 unwind ^bb1 : (!llvm<"i8*">, !llvm<"i8*">, !llvm<"i8*">) -> ()
^bb3:
  llvm.invoke @bar(%8, %6, %4) to ^bb2 unwind ^bb1 : (!llvm<"i8*">, !llvm<"i8*">, !llvm<"i8*">) -> ()

// CHECK-NEXT: ^bb4:
// CHECK-NEXT:   llvm.return %[[a0]] : !llvm.i32
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