// RUN: mlir-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: @nvvm_special_regs
func.func @nvvm_special_regs() -> i32 {
  // CHECK: nvvm.read.ptx.sreg.tid.x : i32
  %0 = nvvm.read.ptx.sreg.tid.x : i32
  // CHECK: nvvm.read.ptx.sreg.tid.y : i32
  %1 = nvvm.read.ptx.sreg.tid.y : i32
  // CHECK: nvvm.read.ptx.sreg.tid.z : i32
  %2 = nvvm.read.ptx.sreg.tid.z : i32
  // CHECK: nvvm.read.ptx.sreg.ntid.x : i32
  %3 = nvvm.read.ptx.sreg.ntid.x : i32
  // CHECK: nvvm.read.ptx.sreg.ntid.y : i32
  %4 = nvvm.read.ptx.sreg.ntid.y : i32
  // CHECK: nvvm.read.ptx.sreg.ntid.z : i32
  %5 = nvvm.read.ptx.sreg.ntid.z : i32
  // CHECK: nvvm.read.ptx.sreg.ctaid.x : i32
  %6 = nvvm.read.ptx.sreg.ctaid.x : i32
  // CHECK: nvvm.read.ptx.sreg.ctaid.y : i32
  %7 = nvvm.read.ptx.sreg.ctaid.y : i32
  // CHECK: nvvm.read.ptx.sreg.ctaid.z : i32
  %8 = nvvm.read.ptx.sreg.ctaid.z : i32
  // CHECK: nvvm.read.ptx.sreg.nctaid.x : i32
  %9 = nvvm.read.ptx.sreg.nctaid.x : i32
  // CHECK: nvvm.read.ptx.sreg.nctaid.y : i32
  %10 = nvvm.read.ptx.sreg.nctaid.y : i32
  // CHECK: nvvm.read.ptx.sreg.nctaid.z : i32
  %11 = nvvm.read.ptx.sreg.nctaid.z : i32
  llvm.return %0 : i32
}

// CHECK-LABEL: @llvm_nvvm_barrier0
func.func @llvm_nvvm_barrier0() {
  // CHECK: nvvm.barrier0
  nvvm.barrier0
  llvm.return
}

// CHECK-LABEL: @nvvm_shfl
func.func @nvvm_shfl(
    %arg0 : i32, %arg1 : i32, %arg2 : i32,
    %arg3 : i32, %arg4 : f32) -> i32 {
  // CHECK: nvvm.shfl.sync bfly %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : i32 -> i32
  %0 = nvvm.shfl.sync bfly %arg0, %arg3, %arg1, %arg2 : i32 -> i32
  // CHECK: nvvm.shfl.sync bfly %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f32 -> f32
  %1 = nvvm.shfl.sync bfly %arg0, %arg4, %arg1, %arg2 : f32 -> f32
  // CHECK: nvvm.shfl.sync up %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f32 -> f32
  %2 = nvvm.shfl.sync up %arg0, %arg4, %arg1, %arg2 : f32 -> f32
  // CHECK: nvvm.shfl.sync down %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f32 -> f32
  %3 = nvvm.shfl.sync down %arg0, %arg4, %arg1, %arg2 : f32 -> f32
  // CHECK: nvvm.shfl.sync idx %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f32 -> f32
  %4 = nvvm.shfl.sync idx %arg0, %arg4, %arg1, %arg2 : f32 -> f32
  llvm.return %0 : i32
}

// CHECK-LABEL: @nvvm_shfl_pred
func.func @nvvm_shfl_pred(
    %arg0 : i32, %arg1 : i32, %arg2 : i32,
    %arg3 : i32, %arg4 : f32) -> !llvm.struct<(i32, i1)> {
  // CHECK: nvvm.shfl.sync bfly %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {return_value_and_is_valid} : i32 -> !llvm.struct<(i32, i1)>
  %0 = nvvm.shfl.sync bfly %arg0, %arg3, %arg1, %arg2 {return_value_and_is_valid} : i32 -> !llvm.struct<(i32, i1)>
  // CHECK: nvvm.shfl.sync bfly %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {return_value_and_is_valid} : f32 -> !llvm.struct<(f32, i1)>
  %1 = nvvm.shfl.sync bfly %arg0, %arg4, %arg1, %arg2 {return_value_and_is_valid} : f32 -> !llvm.struct<(f32, i1)>
  llvm.return %0 : !llvm.struct<(i32, i1)>
}

// CHECK-LABEL: @nvvm_vote(
func.func @nvvm_vote(%arg0 : i32, %arg1 : i1) -> i32 {
  // CHECK: nvvm.vote.ballot.sync %{{.*}}, %{{.*}} : i32
  %0 = nvvm.vote.ballot.sync %arg0, %arg1 : i32
  llvm.return %0 : i32
}

// CHECK-LABEL: @nvvm_mma_m8n8k4_row_col_f32_f32
func.func @nvvm_mma_m8n8k4_row_col_f32_f32(%a0 : vector<2xf16>, %a1 : vector<2xf16>,
               %b0 : vector<2xf16>, %b1 : vector<2xf16>,
               %c0 : f32, %c1 : f32, %c2 : f32, %c3 : f32, %c4 : f32, %c5 : f32, %c6 : f32, %c7 : f32) {
  // CHECK: nvvm.mma.sync
  %0 = nvvm.mma.sync A[%a0, %a1] B[%b0, %b1] C[%c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>,
     shape = {k = 4 : i32, m = 8 : i32, n = 8 : i32}} : (vector<2xf16>, vector<2xf16>, f32) -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
  llvm.return %0 : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mma_m8n8k4_f16_f16
func.func @nvvm_mma_m8n8k4_f16_f16(%a0 : vector<2xf16>, %a1 : vector<2xf16>,
                              %b0 : vector<2xf16>, %b1 : vector<2xf16>,
                              %c0 : vector<2xf16>, %c1 : vector<2xf16>, %c2 : vector<2xf16>, %c3 : vector<2xf16>) {
  // CHECK: nvvm.mma.sync A[{{.*}}] B[{{.*}}] C[{{.*}}]
  %0 = nvvm.mma.sync A[%a0, %a1] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>,
     shape = {k = 4 : i32, m = 8 : i32, n = 8 : i32}} : (vector<2xf16>,vector<2xf16>,vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
  llvm.return %0 : !llvm.struct<(vector<2xf16>, vector<2xf16>, vector<2xf16>, vector<2xf16>)>
}

// CHECK-LABEL: @nvvm_mma_m8n8k16_s8_s8
func.func @nvvm_mma_m8n8k16_s8_s8(%a0 : i32, %b0 : i32,
                             %c0 : i32, %c1 : i32) {
  // CHECK: nvvm.mma.sync A[{{.*}}] B[{{.*}}] C[{{.*}}, {{.*}}] {intOverflowBehavior = #nvvm.mma_int_overflow<wrapped>, layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>, multiplicandAPtxType = #nvvm.mma_type<s8>, multiplicandBPtxType = #nvvm.mma_type<s8>, shape = {k = 16 : i32, m = 8 : i32, n = 8 : i32}} : (i32, i32, i32) -> !llvm.struct<(i32, i32)>
  %0 = nvvm.mma.sync A[%a0] B[%b0] C[%c0, %c1]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>,
     multiplicandAPtxType = #nvvm.mma_type<s8>, multiplicandBPtxType = #nvvm.mma_type<s8>,
     intOverflowBehavior=#nvvm.mma_int_overflow<wrapped>,
     shape = {k = 16 : i32, m = 8 : i32, n = 8 : i32}} : (i32, i32, i32) -> !llvm.struct<(i32, i32)>
  llvm.return %0 : !llvm.struct<(i32, i32)>
}

// CHECK-LABEL: @nvvm_mma_m16n8k8_f16_f16
func.func @nvvm_mma_m16n8k8_f16_f16(%a0 : vector<2xf16>, %a1 : vector<2xf16>,
                               %b0 : vector<2xf16>,
                               %c0 : vector<2xf16>, %c1 : vector<2xf16>) {
  // CHECK: nvvm.mma.sync A[%{{.*}}, %{{.*}}] B[%{{.*}}] C[%{{.*}}, %{{.*}}] {{{.*}}} : (vector<2xf16>, vector<2xf16>, vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  %0 = nvvm.mma.sync A[%a0, %a1] B[%b0] C[%c0, %c1]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>,
     shape = {k = 8 : i32, m = 16 : i32, n = 8 : i32}} : (vector<2xf16>, vector<2xf16>, vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  llvm.return %0 : !llvm.struct<(vector<2xf16>, vector<2xf16>)>
}

// CHECK-LABEL: @nvvm_mma_m16n8k16_f16_f16
func.func @nvvm_mma_m16n8k16_f16_f16(%a0 : vector<2xf16>, %a1 : vector<2xf16>,
                                %a2 : vector<2xf16>, %a3 : vector<2xf16>,
                                %b0 : vector<2xf16>, %b1 : vector<2xf16>,
                                %c0 : vector<2xf16>, %c1 : vector<2xf16>) {
  // CHECK: nvvm.mma.sync A[{{.*}}, {{.*}}, {{.*}}, {{.*}}] B[{{.*}}, {{.*}}] C[{{.*}}, {{.*}}] {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>, shape = {k = 16 : i32, m = 16 : i32, n = 8 : i32}} : (vector<2xf16>, vector<2xf16>, vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  %0 = nvvm.mma.sync A[%a0, %a1, %a2, %a3] B[%b0, %b1] C[%c0, %c1]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>,
     shape = {k = 16 : i32, m = 16 : i32, n = 8 : i32}} : (vector<2xf16>, vector<2xf16>, vector<2xf16>) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  llvm.return %0 : !llvm.struct<(vector<2xf16>, vector<2xf16>)>
}

// CHECK-LABEL: @nvvm_mma_m16n8k16_f32_f16
func.func @nvvm_mma_m16n8k16_f32_f16(%a0 : vector<2xf16>, %a1 : vector<2xf16>,
                                %a2 : vector<2xf16>, %a3 : vector<2xf16>,
                                %b0 : vector<2xf16>, %b1 : vector<2xf16>,
                                %c0 : vector<2xf16>, %c1 : vector<2xf16>) {
  // CHECK: nvvm.mma.sync A[{{.*}}, {{.*}}, {{.*}}, {{.*}}] B[{{.*}}, {{.*}}] C[{{.*}}, {{.*}}] {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>, shape = {k = 16 : i32, m = 16 : i32, n = 8 : i32}} : (vector<2xf16>, vector<2xf16>, vector<2xf16>) -> !llvm.struct<(f32, f32, f32, f32)>
  %0 = nvvm.mma.sync A[%a0, %a1, %a2, %a3] B[%b0, %b1] C[%c0, %c1]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>,
     shape = {k = 16 : i32, m = 16 : i32, n = 8 : i32}} : (vector<2xf16>,vector<2xf16>,vector<2xf16>) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mma_m16n8k16_f16_f32
func.func @nvvm_mma_m16n8k16_f16_f32(%a0 : vector<2xf16>, %a1 : vector<2xf16>,
                                %a2 : vector<2xf16>, %a3 : vector<2xf16>,
                                %b0 : vector<2xf16>, %b1 : vector<2xf16>,
                                %c0 : f32, %c1 : f32, %c2 : f32, %c3 : f32) {
  // CHECK: nvvm.mma.sync A[{{.*}}, {{.*}}, {{.*}}, {{.*}}] B[{{.*}}, {{.*}}] C[{{.*}}, {{.*}}, {{.*}}, {{.*}}] {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>, shape = {k = 16 : i32, m = 16 : i32, n = 8 : i32}} : (vector<2xf16>, vector<2xf16>, f32) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  %0 = nvvm.mma.sync A[%a0, %a1, %a2, %a3] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>,
     shape = {k = 16 : i32, m = 16 : i32, n = 8 : i32}} : (vector<2xf16>, vector<2xf16>, f32) -> !llvm.struct<(vector<2xf16>, vector<2xf16>)>
  llvm.return %0 : !llvm.struct<(vector<2xf16>, vector<2xf16>)>
}

// CHECK-LABEL: @nvvm_mma_m16n8k16_f32_f32
func.func @nvvm_mma_m16n8k16_f32_f32(%a0 : vector<2xf16>, %a1 : vector<2xf16>,
                                %a2 : vector<2xf16>, %a3 : vector<2xf16>,
                                %b0 : vector<2xf16>, %b1 : vector<2xf16>,
                                %c0 : f32, %c1 : f32, %c2 : f32, %c3 : f32) {
  // CHECK: nvvm.mma.sync A[{{.*}}, {{.*}}, {{.*}}, {{.*}}] B[{{.*}}, {{.*}}] C[{{.*}}, {{.*}}, {{.*}}, {{.*}}] {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>, shape = {k = 16 : i32, m = 16 : i32, n = 8 : i32}} : (vector<2xf16>, vector<2xf16>, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  %0 = nvvm.mma.sync A[%a0, %a1, %a2, %a3] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>,
     shape = {k = 16 : i32, m = 16 : i32, n = 8 : i32}} : (vector<2xf16>, vector<2xf16>, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mma_m16n8k4_tf32_f32
func.func @nvvm_mma_m16n8k4_tf32_f32(%a0 : i32, %a1 : i32,
                                     %b0 : i32,
                                     %c0 : f32, %c1 : f32, %c2 : f32, %c3 : f32) {
  // CHECK: nvvm.mma.sync A[{{.*}}, {{.*}}] B[{{.*}}] C[{{.*}}, {{.*}}, {{.*}}, {{.*}}] {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>, multiplicandAPtxType = #nvvm.mma_type<tf32>, multiplicandBPtxType = #nvvm.mma_type<tf32>, shape = {k = 4 : i32, m = 16 : i32, n = 8 : i32}} : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  %0 = nvvm.mma.sync A[%a0, %a1] B[%b0] C[%c0, %c1, %c2, %c3]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>,
     multiplicandAPtxType = #nvvm.mma_type<tf32>, multiplicandBPtxType = #nvvm.mma_type<tf32>,
     shape = {k = 4 : i32, m = 16 : i32, n = 8 : i32}} : (i32, i32, f32) -> !llvm.struct<(f32, f32, f32, f32)>
  llvm.return %0 : !llvm.struct<(f32, f32, f32, f32)>
}

// CHECK-LABEL: @nvvm_mma_m16n8k16_s8_s8
func.func @nvvm_mma_m16n8k16_s8_s8(%a0 : i32, %a1 : i32, %b0 : i32,
                              %c0 : i32, %c1 : i32, %c2 : i32, %c3 : i32) {
  // CHECK: nvvm.mma.sync A[{{.*}}, {{.*}}] B[{{.*}}] C[{{.*}}, {{.*}}, {{.*}}, {{.*}}] {intOverflowBehavior = #nvvm.mma_int_overflow<wrapped>, layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>, multiplicandAPtxType = #nvvm.mma_type<s8>, multiplicandBPtxType = #nvvm.mma_type<s8>, shape = {k = 16 : i32, m = 16 : i32, n = 8 : i32}} : (i32, i32, i32) -> !llvm.struct<(i32, i32, i32, i32)>
  %0 = nvvm.mma.sync A[%a0, %a1] B[%b0] C[%c0, %c1, %c2, %c3]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>,
     multiplicandAPtxType = #nvvm.mma_type<s8>, multiplicandBPtxType = #nvvm.mma_type<s8>,
     intOverflowBehavior=#nvvm.mma_int_overflow<wrapped>,
     shape = {k = 16 : i32, m = 16 : i32, n = 8 : i32}} : (i32, i32, i32) -> !llvm.struct<(i32,i32,i32,i32)>
  llvm.return %0 : !llvm.struct<(i32,i32,i32,i32)>
}

// CHECK-LABEL: @nvvm_mma_m16n8k16_s8_u8
func.func @nvvm_mma_m16n8k16_s8_u8(%a0 : i32, %a1 : i32,
                                %b0 : i32,
                                %c0 : i32, %c1 : i32, %c2 : i32, %c3 : i32) {
  // CHECK: nvvm.mma.sync A[{{.*}}, {{.*}}] B[{{.*}}] C[{{.*}}, {{.*}}, {{.*}}, {{.*}}] {intOverflowBehavior = #nvvm.mma_int_overflow<satfinite>, layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>, multiplicandAPtxType = #nvvm.mma_type<s8>, multiplicandBPtxType = #nvvm.mma_type<u8>, shape = {k = 16 : i32, m = 16 : i32, n = 8 : i32}} : (i32, i32, i32) -> !llvm.struct<(i32, i32, i32, i32)>
  %0 = nvvm.mma.sync A[%a0, %a1] B[%b0] C[%c0, %c1, %c2, %c3]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>,
     multiplicandAPtxType = #nvvm.mma_type<s8>, multiplicandBPtxType = #nvvm.mma_type<u8>,
     intOverflowBehavior=#nvvm.mma_int_overflow<satfinite>,
     shape = {k = 16 : i32, m = 16 : i32, n = 8 : i32}} : (i32, i32, i32) -> !llvm.struct<(i32,i32,i32,i32)>
  llvm.return %0 : !llvm.struct<(i32,i32,i32,i32)>
}

// CHECK-LABEL: @nvvm_mma_m16n8k256_b1_b1
func.func @nvvm_mma_m16n8k256_b1_b1(%a0 : i32, %a1 : i32, %a2 : i32, %a3 : i32,
                               %b0 : i32, %b1 : i32,
                               %c0 : i32, %c1 : i32, %c2 : i32, %c3 : i32) {
  // CHECK: nvvm.mma.sync A[{{.*}}, {{.*}}, {{.*}}, {{.*}}] B[{{.*}}] C[{{.*}}, {{.*}}, {{.*}}, {{.*}}] {b1Op = #nvvm.mma_b1op<xor_popc>, layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>, multiplicandAPtxType = #nvvm.mma_type<b1>, multiplicandBPtxType = #nvvm.mma_type<b1>, shape = {k = 256 : i32, m = 16 : i32, n = 8 : i32}} : (i32, i32, i32) -> !llvm.struct<(i32, i32, i32, i32)>
  %0 = nvvm.mma.sync A[%a0, %a1, %a2, %a3] B[%b0, %b1] C[%c0, %c1, %c2, %c3]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>,
     multiplicandAPtxType = #nvvm.mma_type<b1>, multiplicandBPtxType = #nvvm.mma_type<b1>,
     b1Op = #nvvm.mma_b1op<xor_popc>, shape = {k = 256 : i32, m = 16 : i32, n = 8 : i32}} : (i32, i32, i32) -> !llvm.struct<(i32,i32,i32,i32)>
  llvm.return %0 : !llvm.struct<(i32,i32,i32,i32)>
}

// CHECK-LABEL: @nvvm_mma_m16n8k128_b1_b1
func.func @nvvm_mma_m16n8k128_b1_b1(%a0 : i32, %a1 : i32,
                               %b0 : i32,
                               %c0 : i32, %c1 : i32, %c2 : i32, %c3 : i32) {
  // CHECK: nvvm.mma.sync A[{{.*}}, {{.*}}] B[{{.*}}] C[{{.*}}, {{.*}}, {{.*}}, {{.*}}] {b1Op = #nvvm.mma_b1op<xor_popc>, layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>, multiplicandAPtxType = #nvvm.mma_type<b1>, multiplicandBPtxType = #nvvm.mma_type<b1>, shape = {k = 128 : i32, m = 16 : i32, n = 8 : i32}} : (i32, i32, i32) -> !llvm.struct<(i32, i32, i32, i32)>
  %0 = nvvm.mma.sync A[%a0, %a1] B[%b0] C[%c0, %c1, %c2, %c3]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>,
     multiplicandAPtxType = #nvvm.mma_type<b1>, multiplicandBPtxType = #nvvm.mma_type<b1>,
     b1Op = #nvvm.mma_b1op<xor_popc>,
     shape = {k = 128 : i32, m = 16 : i32, n = 8 : i32}} : (i32, i32, i32) -> !llvm.struct<(i32,i32,i32,i32)>
  llvm.return %0 : !llvm.struct<(i32,i32,i32,i32)>
}

// CHECK-LABEL: @nvvm_mma_m8n8k128_b1_b1
func.func @nvvm_mma_m8n8k128_b1_b1(%a0 : i32,
                              %b0 : i32,
                              %c0 : i32, %c1 : i32) {
  // CHECK: nvvm.mma.sync A[{{.*}}] B[{{.*}}] C[{{.*}}, {{.*}}] {b1Op = #nvvm.mma_b1op<xor_popc>, layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>, multiplicandAPtxType = #nvvm.mma_type<b1>, multiplicandBPtxType = #nvvm.mma_type<b1>, shape = {k = 128 : i32, m = 8 : i32, n = 8 : i32}} : (i32, i32, i32) -> !llvm.struct<(i32, i32)>
  %0 = nvvm.mma.sync A[%a0] B[%b0] C[%c0, %c1]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>,
     multiplicandAPtxType = #nvvm.mma_type<b1>, multiplicandBPtxType = #nvvm.mma_type<b1>,
     b1Op = #nvvm.mma_b1op<xor_popc>, shape = {k = 128 : i32, m = 8 : i32, n = 8 : i32}} : (i32, i32, i32) -> !llvm.struct<(i32,i32)>
  llvm.return %0 : !llvm.struct<(i32,i32)>
}

// CHECK-LABEL: @nvvm_mma_m16n8k32_s4_s4
func.func @nvvm_mma_m16n8k32_s4_s4(%a0 : i32, %a1 : i32,
                               %b0 : i32,
                               %c0 : i32, %c1 : i32, %c2 : i32, %c3 : i32) {
  // CHECK: nvvm.mma.sync A[{{.*}}, {{.*}}] B[{{.*}}] C[{{.*}}, {{.*}}, {{.*}}, {{.*}}] {intOverflowBehavior = #nvvm.mma_int_overflow<wrapped>, layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>, multiplicandAPtxType = #nvvm.mma_type<s4>, multiplicandBPtxType = #nvvm.mma_type<s4>, shape = {k = 32 : i32, m = 16 : i32, n = 8 : i32}} : (i32, i32, i32) -> !llvm.struct<(i32, i32, i32, i32)>
  %0 = nvvm.mma.sync A[%a0, %a1] B[%b0] C[%c0, %c1, %c2, %c3]
    {layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>,
     multiplicandAPtxType = #nvvm.mma_type<s4>, multiplicandBPtxType = #nvvm.mma_type<s4>,
     intOverflowBehavior=#nvvm.mma_int_overflow<wrapped>,
     shape = {k = 32 : i32, m = 16 : i32, n = 8 : i32}} : (i32, i32, i32) -> !llvm.struct<(i32,i32,i32,i32)>
  llvm.return %0 : !llvm.struct<(i32,i32,i32,i32)>
}

// CHECK-LABEL: @nvvm_wmma_load_tf32
func.func @nvvm_wmma_load_tf32(%arg0: !llvm.ptr<i32>, %arg1 : i32) -> !llvm.struct<(i32, i32, i32, i32)> {
  // CHECK: nvvm.wmma.load {{.*}} {eltype = #nvvm.mma_type<tf32>, frag = #nvvm.mma_frag<a>, k = 8 : i32, layout = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32}
  %0 = nvvm.wmma.load %arg0, %arg1
    {eltype = #nvvm.mma_type<tf32>, frag = #nvvm.mma_frag<a>, k = 8 : i32, layout = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32}
    : (!llvm.ptr<i32>) -> !llvm.struct<(i32, i32, i32, i32)>
  llvm.return %0 : !llvm.struct<(i32, i32, i32, i32)>
}

// CHECK-LABEL: @nvvm_wmma_mma
func.func @nvvm_wmma_mma(%0 : i32, %1 : i32, %2 : i32, %3 : i32, %4 : i32, %5 : i32,
                    %6 : i32, %7 : i32, %8 : f32, %9 : f32, %10 : f32,
                    %11 : f32, %12 : f32, %13 : f32, %14 : f32, %15 : f32)
                   -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)> {
  // CHECK: nvvm.wmma.mma {{.*}} {eltypeA = #nvvm.mma_type<tf32>, eltypeB = #nvvm.mma_type<f32>, k = 8 : i32, layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32}
  %r = nvvm.wmma.mma %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15
    {eltypeA = #nvvm.mma_type<tf32>, eltypeB = #nvvm.mma_type<f32>, k = 8 : i32, layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<row>, m = 16 : i32, n = 16 : i32}
    : (i32, i32, i32, i32, i32, i32, i32, i32, f32, f32, f32, f32, f32, f32, f32, f32)
    -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
  llvm.return %r : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
}

// CHECK-LABEL: @cp_async
llvm.func @cp_async(%arg0: !llvm.ptr<i8, 3>, %arg1: !llvm.ptr<i8, 1>) {
// CHECK:  nvvm.cp.async.shared.global %{{.*}}, %{{.*}}, 16
  nvvm.cp.async.shared.global %arg0, %arg1, 16
// CHECK:  nvvm.cp.async.shared.global %{{.*}}, %{{.*}}, 16 {bypass_l1}
  nvvm.cp.async.shared.global %arg0, %arg1, 16 {bypass_l1}
// CHECK: nvvm.cp.async.commit.group
  nvvm.cp.async.commit.group
// CHECK: nvvm.cp.async.wait.group 0
  nvvm.cp.async.wait.group 0
  llvm.return
}

// CHECK-LABEL: llvm.func @ld_matrix
llvm.func @ld_matrix(%arg0: !llvm.ptr<i32, 3>) {
  // CHECK: nvvm.ldmatrix %{{.*}} {layout = #nvvm.mma_layout<row>, num = 1 : i32} : (!llvm.ptr<i32, 3>) -> i32
  %l1 = nvvm.ldmatrix %arg0 {num = 1 : i32, layout = #nvvm.mma_layout<row>} : (!llvm.ptr<i32, 3>) -> i32
  // CHECK: nvvm.ldmatrix %{{.*}} {layout = #nvvm.mma_layout<row>, num = 2 : i32} : (!llvm.ptr<i32, 3>) -> !llvm.struct<(i32, i32)>
  %l2 = nvvm.ldmatrix %arg0 {num = 2 : i32, layout = #nvvm.mma_layout<row>} : (!llvm.ptr<i32, 3>) -> !llvm.struct<(i32, i32)>
  // CHECK: nvvm.ldmatrix %{{.*}} {layout = #nvvm.mma_layout<row>, num = 4 : i32} : (!llvm.ptr<i32, 3>) -> !llvm.struct<(i32, i32, i32, i32)>
  %l4 = nvvm.ldmatrix %arg0 {num = 4 : i32, layout = #nvvm.mma_layout<row>} : (!llvm.ptr<i32, 3>) -> !llvm.struct<(i32, i32, i32, i32)>
  llvm.return
}
// -----

// expected-error@below {{attribute attached to unexpected op}}
func.func private @expected_llvm_func() attributes { nvvm.kernel }
