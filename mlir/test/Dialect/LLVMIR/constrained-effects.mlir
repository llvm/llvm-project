// RUN: mlir-opt %s -canonicalize -cse -loop-invariant-code-motion | FileCheck %s
// RUN: mlir-opt %s -cse | FileCheck %s

llvm.func @change_environment()

// CHECK-LABEL: llvm.func @unused_strict
// CHECK: llvm.intr.experimental.constrained.sqrt {{.*}} tonearest strict
// CHECK: llvm.return
llvm.func @unused_strict(%input: f32) {
  %unused = llvm.intr.experimental.constrained.sqrt %input tonearest strict : f32
  llvm.return
}

// CHECK-LABEL: llvm.func @unused_maytrap
// CHECK: llvm.intr.experimental.constrained.sqrt {{.*}} tonearest maytrap
// CHECK: llvm.return
llvm.func @unused_maytrap(%input: f32) {
  %unused = llvm.intr.experimental.constrained.sqrt %input tonearest maytrap : f32
  llvm.return
}

// CHECK-LABEL: llvm.func @unused_ignore
// CHECK-NOT: llvm.intr.experimental.constrained
// CHECK: llvm.return
llvm.func @unused_ignore(%input: f32) {
  %unused = llvm.intr.experimental.constrained.sqrt %input tonearest ignore : f32
  llvm.return
}

// CHECK-LABEL: llvm.func @unused_binary_strict
// CHECK: llvm.intr.experimental.constrained.fadd {{.*}} dynamic strict
// CHECK: llvm.return
llvm.func @unused_binary_strict(%input: f32, %increment: f32) {
  %unused = llvm.intr.experimental.constrained.fadd %input, %increment dynamic strict : f32
  llvm.return
}

// CHECK-LABEL: llvm.func @unused_no_rounding_strict
// CHECK: llvm.intr.experimental.constrained.ceil {{.*}} strict
// CHECK: llvm.return
llvm.func @unused_no_rounding_strict(%input: f32) {
  %unused = llvm.intr.experimental.constrained.ceil %input strict : f32
  llvm.return
}

// CHECK-LABEL: llvm.func @strict_not_cse
// CHECK: llvm.intr.experimental.constrained.sqrt
// CHECK: llvm.intr.experimental.constrained.sqrt
// CHECK: llvm.return
llvm.func @strict_not_cse(%input: f32) -> f32 {
  %first = llvm.intr.experimental.constrained.sqrt %input tonearest strict : f32
  %second = llvm.intr.experimental.constrained.sqrt %input tonearest strict : f32
  %sum = llvm.fadd %first, %second : f32
  llvm.return %sum : f32
}

// CHECK-LABEL: llvm.func @ignore_can_cse
// CHECK: %[[VALUE:.*]] = llvm.intr.experimental.constrained.sqrt
// CHECK-NOT: llvm.intr.experimental.constrained.sqrt
// CHECK: llvm.fadd %[[VALUE]], %[[VALUE]]
// CHECK: llvm.return
llvm.func @ignore_can_cse(%input: f32) -> f32 {
  %first = llvm.intr.experimental.constrained.sqrt %input tonearest ignore : f32
  %second = llvm.intr.experimental.constrained.sqrt %input tonearest ignore : f32
  %sum = llvm.fadd %first, %second : f32
  llvm.return %sum : f32
}

// CHECK-LABEL: llvm.func @dynamic_rounding_not_cse
// CHECK: llvm.intr.experimental.constrained.fadd {{.*}} dynamic ignore
// CHECK: llvm.call @change_environment
// CHECK: llvm.intr.experimental.constrained.fadd {{.*}} dynamic ignore
// CHECK: llvm.return
llvm.func @dynamic_rounding_not_cse(%input: f32, %increment: f32) -> f32 {
  %first = llvm.intr.experimental.constrained.fadd %input, %increment dynamic ignore : f32
  llvm.call @change_environment() : () -> ()
  %second = llvm.intr.experimental.constrained.fadd %input, %increment dynamic ignore : f32
  %difference = llvm.fsub %second, %first : f32
  llvm.return %difference : f32
}

// CHECK-LABEL: func.func @strict_stays_in_loop
// CHECK-NOT: llvm.intr.experimental.constrained.sqrt
// CHECK: scf.for
// CHECK: llvm.intr.experimental.constrained.sqrt {{.*}} tonearest strict
// CHECK: return
func.func @strict_stays_in_loop(%lower: index, %upper: index, %step: index, %input: f32) {
  scf.for %index = %lower to %upper step %step {
    %unused = llvm.intr.experimental.constrained.sqrt %input tonearest strict : f32
  }
  return
}

// CHECK-LABEL: llvm.func @unused_quiet_comparison_strict
// CHECK: llvm.intr.experimental.constrained.fcmp oeq {{.*}} strict
// CHECK: llvm.return
llvm.func @unused_quiet_comparison_strict(%left: f32, %right: f32) {
  %unused = llvm.intr.experimental.constrained.fcmp "oeq" %left, %right strict : f32
  llvm.return
}

// CHECK-LABEL: llvm.func @unused_signaling_comparison_strict
// CHECK: llvm.intr.experimental.constrained.fcmps oeq {{.*}} strict
// CHECK: llvm.return
llvm.func @unused_signaling_comparison_strict(%left: f32, %right: f32) {
  %unused = llvm.intr.experimental.constrained.fcmps "oeq" %left, %right strict : f32
  llvm.return
}

// CHECK-LABEL: llvm.func @unused_signaling_comparison_maytrap
// CHECK: llvm.intr.experimental.constrained.fcmps oeq {{.*}} maytrap
// CHECK: llvm.return
llvm.func @unused_signaling_comparison_maytrap(%left: f32, %right: f32) {
  %unused = llvm.intr.experimental.constrained.fcmps "oeq" %left, %right maytrap : f32
  llvm.return
}

// CHECK-LABEL: llvm.func @unused_signaling_comparison_ignore
// CHECK-NOT: llvm.intr.experimental.constrained
// CHECK: llvm.return
llvm.func @unused_signaling_comparison_ignore(%left: f32, %right: f32) {
  %unused = llvm.intr.experimental.constrained.fcmps "oeq" %left, %right ignore : f32
  llvm.return
}

// CHECK-LABEL: llvm.func @unused_conversion_strict
// CHECK: llvm.intr.experimental.constrained.fptosi {{.*}} strict
// CHECK: llvm.return
llvm.func @unused_conversion_strict(%input: f64) {
  %unused = llvm.intr.experimental.constrained.fptosi %input strict : f64 to i32
  llvm.return
}

// CHECK-LABEL: llvm.func @unused_truncation_strict
// CHECK: llvm.intr.experimental.constrained.fptrunc {{.*}} dynamic strict
// CHECK: llvm.return
llvm.func @unused_truncation_strict(%input: f64) {
  %unused = llvm.intr.experimental.constrained.fptrunc %input dynamic strict : f64 to f32
  llvm.return
}

// CHECK-LABEL: llvm.func @unused_vector_maytrap
// CHECK: llvm.intr.experimental.constrained.sqrt {{.*}} upward maytrap
// CHECK: llvm.return
llvm.func @unused_vector_maytrap(%input: vector<4xf32>) {
  %unused = llvm.intr.experimental.constrained.sqrt %input upward maytrap : vector<4xf32>
  llvm.return
}

// CHECK-LABEL: llvm.func @unused_dynamic_ignore
// CHECK-NOT: llvm.intr.experimental.constrained
// CHECK: llvm.return
llvm.func @unused_dynamic_ignore(%input: f64) {
  %unused = llvm.intr.experimental.constrained.sqrt %input dynamic ignore : f64
  llvm.return
}

// CHECK-LABEL: llvm.func @maytrap_not_cse
// CHECK: llvm.intr.experimental.constrained.sqrt
// CHECK: llvm.intr.experimental.constrained.sqrt
// CHECK: llvm.return
llvm.func @maytrap_not_cse(%input: f64) -> f64 {
  %first = llvm.intr.experimental.constrained.sqrt %input downward maytrap : f64
  %second = llvm.intr.experimental.constrained.sqrt %input downward maytrap : f64
  %sum = llvm.fadd %first, %second : f64
  llvm.return %sum : f64
}

// CHECK-LABEL: llvm.func @comparison_ignore_can_cse
// CHECK: %[[COMPARE:.*]] = llvm.intr.experimental.constrained.fcmps
// CHECK-NOT: llvm.intr.experimental.constrained.fcmps
// CHECK: llvm.xor %[[COMPARE]], %[[COMPARE]]
// CHECK: llvm.return
llvm.func @comparison_ignore_can_cse(%left: vector<4xf32>, %right: vector<4xf32>) -> vector<4xi1> {
  %first = llvm.intr.experimental.constrained.fcmps "olt" %left, %right ignore : vector<4xf32>
  %second = llvm.intr.experimental.constrained.fcmps "olt" %left, %right ignore : vector<4xf32>
  %difference = llvm.xor %first, %second : vector<4xi1>
  llvm.return %difference : vector<4xi1>
}
