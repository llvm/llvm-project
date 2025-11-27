// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: @test_prmt_default
llvm.func @test_prmt_default(%lo: i32, %sel: i32, %hi: i32) -> i32 {
  // CHECK: call i32 @llvm.nvvm.prmt(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %result = nvvm.prmt #nvvm.permute_mode<default> %lo, %sel, %hi : i32
  llvm.return %result : i32
}

// CHECK-LABEL: @test_prmt_f4e
llvm.func @test_prmt_f4e(%lo: i32, %pos: i32, %hi: i32) -> i32 {
  // CHECK: call i32 @llvm.nvvm.prmt.f4e(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %result = nvvm.prmt #nvvm.permute_mode<f4e> %lo, %pos, %hi : i32
  llvm.return %result : i32
}

// CHECK-LABEL: @test_prmt_b4e
llvm.func @test_prmt_b4e(%lo: i32, %pos: i32, %hi: i32) -> i32 {
  // CHECK: call i32 @llvm.nvvm.prmt.b4e(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %result = nvvm.prmt #nvvm.permute_mode<b4e> %lo, %pos, %hi : i32
  llvm.return %result : i32
}

// CHECK-LABEL: @test_prmt_rc8
llvm.func @test_prmt_rc8(%val: i32, %sel: i32) -> i32 {
  // CHECK: call i32 @llvm.nvvm.prmt.rc8(i32 %{{.*}}, i32 %{{.*}})
  %result = nvvm.prmt #nvvm.permute_mode<rc8> %val, %sel : i32
  llvm.return %result : i32
}

// CHECK-LABEL: @test_prmt_ecl
llvm.func @test_prmt_ecl(%val: i32, %sel: i32) -> i32 {
  // CHECK: call i32 @llvm.nvvm.prmt.ecl(i32 %{{.*}}, i32 %{{.*}})
  %result = nvvm.prmt #nvvm.permute_mode<ecl> %val, %sel : i32
  llvm.return %result : i32
}

// CHECK-LABEL: @test_prmt_ecr
llvm.func @test_prmt_ecr(%val: i32, %sel: i32) -> i32 {
  // CHECK: call i32 @llvm.nvvm.prmt.ecr(i32 %{{.*}}, i32 %{{.*}})
  %result = nvvm.prmt #nvvm.permute_mode<ecr> %val, %sel : i32
  llvm.return %result : i32
}

// CHECK-LABEL: @test_prmt_rc16
llvm.func @test_prmt_rc16(%val: i32, %sel: i32) -> i32 {
  // CHECK: call i32 @llvm.nvvm.prmt.rc16(i32 %{{.*}}, i32 %{{.*}})
  %result = nvvm.prmt #nvvm.permute_mode<rc16> %val, %sel : i32
  llvm.return %result : i32
}

// CHECK-LABEL: @test_prmt_mixed
llvm.func @test_prmt_mixed(%lo: i32, %sel: i32, %hi: i32) -> i32 {
  // CHECK: call i32 @llvm.nvvm.prmt(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %r1 = nvvm.prmt #nvvm.permute_mode<default> %lo, %sel, %hi : i32

  // CHECK: call i32 @llvm.nvvm.prmt.rc8(i32 %{{.*}}, i32 %{{.*}})
  %r2 = nvvm.prmt #nvvm.permute_mode<rc8> %r1, %sel : i32

  // CHECK: call i32 @llvm.nvvm.prmt.f4e(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %r3 = nvvm.prmt #nvvm.permute_mode<f4e> %r2, %lo, %sel : i32

  llvm.return %r3 : i32
}
