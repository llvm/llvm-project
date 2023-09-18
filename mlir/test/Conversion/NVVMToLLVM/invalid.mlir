// RUN: mlir-opt --convert-nvvm-to-llvm --split-input-file -verify-diagnostics %s

!mat64f32 = !llvm.struct<(f32, f32, f32, f32, f32, f32, f32)>
func.func @wgmma_f32_f16_f16(%descA : i64, %descB : i64) -> !mat64f32{  
  %result = llvm.mlir.undef : !mat64f32
  // expected-error @+1 {{'nvvm.wgmma.mma_async' op results 64, however output struct has 7 elements}}
  %res = nvvm.wgmma.mma_async %descA, %descB, 
      #nvvm.shape<m = 64, n = 128, k = 16>, 
      D [%result, <zero>],
      A [<f16>, #nvvm.wgmma_scale_in<neg>, <col>], 
      B [<f16>, #nvvm.wgmma_scale_in<neg>, <col>]
      : !mat64f32 -> !mat64f32
  return %res : !mat64f32
}

// -----

func.func @wgmma_f32_satfinite(%descA : i64, %descB : i64) {  
  %result = llvm.mlir.undef : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
  // expected-error @+1 {{`satfinite` can be only used with s32 accumulator, however the current accumulator is 'f32'}}
  %res = nvvm.wgmma.mma_async %descA, %descB, 
      #nvvm.shape<m = 64, n = 16, k = 16>, 
      D [%result, <zero>, <satfinite>], 
      A [<f16>, #nvvm.wgmma_scale_in<neg>, <col>], 
      B [<f16>, #nvvm.wgmma_scale_in<neg>, <col>]
      : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
      -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
  return 
}

// -----

func.func @wgmma_f32_m32(%descA : i64, %descB : i64) {  
  %result = llvm.mlir.undef : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
  // expected-error @+1 {{shape 'm' must be 64}}
  %res = nvvm.wgmma.mma_async %descA, %descB, 
      #nvvm.shape<m = 32, n = 16, k = 16>, 
      D [%result, <zero>], 
      A [<f16>, #nvvm.wgmma_scale_in<neg>, <col>], 
      B [<f16>, #nvvm.wgmma_scale_in<neg>, <col>]
      : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
      -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
  return 
}

// -----

func.func @wgmma_f32_m32(%descA : i64, %descB : i64) { 
  %result = llvm.mlir.undef : !llvm.struct<(f32, f32, i32, f32, f32, f32, f32, f32)> 
  // expected-error @+1 {{op all elements in struct must be same type but there is 'i32'}}
  %res = nvvm.wgmma.mma_async %descA, %descB, 
      #nvvm.shape<m = 64, n = 16, k = 16>, 
      D [%result, <zero>], 
      A [<f16>, #nvvm.wgmma_scale_in<neg>, <col>], 
      B [<f16>, #nvvm.wgmma_scale_in<neg>, <col>]
      : !llvm.struct<(f32, f32, i32, f32, f32, f32, f32, f32)> 
      -> !llvm.struct<(f32, f32, i32, f32, f32, f32, f32, f32)> 
  return 
}

// -----

func.func @wgmma_f32_m32(%descA : i64, %descB : i64) {  
  %result = llvm.mlir.undef : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)> 
  // expected-error @+1 {{op shape 'k' must be 16 for input type f16}}
  %res = nvvm.wgmma.mma_async %descA, %descB, 
      #nvvm.shape<m = 64, n = 16, k = 3>, 
      D [%result, <zero>], 
      A [<f16>, #nvvm.wgmma_scale_in<neg>, <col>], 
      B [<f16>, #nvvm.wgmma_scale_in<neg>, <col>]
      : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)> 
      -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)> 
  return 
}

// -----

func.func @wgmma_transpose(%descA : i64, %descB : i64) {
  %result = llvm.mlir.undef : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)> 
  // expected-error @+1 {{op given layouts layout_a = col and layout_b = col for input types tf32 and tf32 requires transpose. However, this is only supported for: f16 and bf16}}
  %res = nvvm.wgmma.mma_async %descA, %descB, 
      #nvvm.shape<m = 64, n = 16, k = 8>, 
      D [%result, <zero>], 
      A [<tf32>, #nvvm.wgmma_scale_in<neg>, <col>], 
      B [<tf32>, #nvvm.wgmma_scale_in<neg>, <col>]
      : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)> 
      -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)> 
  return 
}

// -----

func.func @wgmma_transpose(%descA : i64, %descB : i64) {  
  %result = llvm.mlir.undef : !llvm.struct<(f16, f16, f16, f16)>
  // expected-error @+1 {{'nvvm.wgmma.mma_async' op 'f16' += tf32 * tf32, it is not supported.}}
  %res = nvvm.wgmma.mma_async %descA, %descB, 
      #nvvm.shape<m = 64, n = 16, k = 8>, 
      D [%result, <zero>], 
      A [<tf32>, #nvvm.wgmma_scale_in<neg>, <col>], 
      B [<tf32>, #nvvm.wgmma_scale_in<neg>, <col>]
      :!llvm.struct<(f16, f16, f16, f16)>
      -> !llvm.struct<(f16, f16, f16, f16)>
  return 
}

// -----

func.func @wgmma_f32_m32(%descA : i64, %descB : i64) {  
  %result = llvm.mlir.undef : !llvm.struct<(i32, i32, i32, i32)>
  // expected-error @+1 {{input struct and result struct must be the same type}}
  %res = nvvm.wgmma.mma_async %descA, %descB, 
      #nvvm.shape<m = 64, n = 8, k = 16>, 
      D [%result, <zero>], 
      A [<f16>, #nvvm.wgmma_scale_in<neg>, <col>], 
      B [<f16>, #nvvm.wgmma_scale_in<neg>, <col>]
      : !llvm.struct<(i32, i32, i32, i32)> 
      -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)> 
  return 
}

// -----

func.func @wgmma_f32_m32(%descA : i64, %descB : i64) {  
  %result = llvm.mlir.undef : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)> 
  // expected-error @+1 {{op 'f32' += bf16 * f16, it is not supported}}
  %res = nvvm.wgmma.mma_async %descA, %descB, 
      #nvvm.shape<m = 64, n = 8, k = 16>, 
      D [%result, <zero>], 
      A [<bf16>, #nvvm.wgmma_scale_in<neg>, <col>], 
      B [<f16>, #nvvm.wgmma_scale_in<neg>, <col>]
      : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)> 
      -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)> 
  return 
}
