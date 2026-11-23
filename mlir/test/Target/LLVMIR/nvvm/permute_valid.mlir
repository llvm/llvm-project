// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: @test_prmt_default
llvm.func @test_prmt_default(%sel: i32, %lo: i32, %hi: i32) -> i32 {
  // CHECK: call i32 @llvm.nvvm.prmt(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %result = nvvm.prmt #nvvm.permute_mode<default> %sel, %lo, %hi : i32
  llvm.return %result : i32
}

// CHECK-LABEL: @test_prmt_f4e
llvm.func @test_prmt_f4e(%pos: i32, %lo: i32, %hi: i32) -> i32 {
  // CHECK: call i32 @llvm.nvvm.prmt.f4e(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %result = nvvm.prmt #nvvm.permute_mode<f4e> %pos, %lo, %hi : i32
  llvm.return %result : i32
}

// CHECK-LABEL: @test_prmt_b4e
llvm.func @test_prmt_b4e(%pos: i32, %lo: i32, %hi: i32) -> i32 {
  // CHECK: call i32 @llvm.nvvm.prmt.b4e(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %result = nvvm.prmt #nvvm.permute_mode<b4e> %pos, %lo, %hi : i32
  llvm.return %result : i32
}

// CHECK-LABEL: @test_prmt_rc8
llvm.func @test_prmt_rc8(%sel: i32, %val: i32) -> i32 {
  // CHECK: call i32 @llvm.nvvm.prmt.rc8(i32 %{{.*}}, i32 %{{.*}})
  %result = nvvm.prmt #nvvm.permute_mode<rc8> %sel, %val : i32
  llvm.return %result : i32
}

// CHECK-LABEL: @test_prmt_ecl
llvm.func @test_prmt_ecl(%sel: i32, %val: i32) -> i32 {
  // CHECK: call i32 @llvm.nvvm.prmt.ecl(i32 %{{.*}}, i32 %{{.*}})
  %result = nvvm.prmt #nvvm.permute_mode<ecl> %sel, %val : i32
  llvm.return %result : i32
}

// CHECK-LABEL: @test_prmt_ecr
llvm.func @test_prmt_ecr(%sel: i32, %val: i32) -> i32 {
  // CHECK: call i32 @llvm.nvvm.prmt.ecr(i32 %{{.*}}, i32 %{{.*}})
  %result = nvvm.prmt #nvvm.permute_mode<ecr> %sel, %val : i32
  llvm.return %result : i32
}

// CHECK-LABEL: @test_prmt_rc16
llvm.func @test_prmt_rc16(%sel: i32, %val: i32) -> i32 {
  // CHECK: call i32 @llvm.nvvm.prmt.rc16(i32 %{{.*}}, i32 %{{.*}})
  %result = nvvm.prmt #nvvm.permute_mode<rc16> %sel, %val : i32
  llvm.return %result : i32
}

// CHECK-LABEL: @test_prmt_mixed
llvm.func @test_prmt_mixed(%sel: i32, %lo: i32, %hi: i32) -> i32 {
  // CHECK: call i32 @llvm.nvvm.prmt(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %r1 = nvvm.prmt #nvvm.permute_mode<default> %sel, %lo, %hi : i32

  // CHECK: call i32 @llvm.nvvm.prmt.rc8(i32 %{{.*}}, i32 %{{.*}})
  %r2 = nvvm.prmt #nvvm.permute_mode<rc8> %sel, %r1 : i32

  // CHECK: call i32 @llvm.nvvm.prmt.f4e(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  %r3 = nvvm.prmt #nvvm.permute_mode<f4e> %lo, %r2, %sel : i32

  llvm.return %r3 : i32
}
