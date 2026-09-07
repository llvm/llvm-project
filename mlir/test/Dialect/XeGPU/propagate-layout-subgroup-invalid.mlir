// RUN: mlir-opt -xevm-attach-target='chip=cri' -test-xegpu-propagate-layouts="layout-kind=subgroup" -split-input-file -verify-diagnostics %s

// Ops whose required layout cannot be determined are a hard failure: no valid
// subgroup layout exists for the shape, so propagation stops with an error
// instead of silently leaving the op unlabeled.

gpu.module @test {
  gpu.func @store_fails(%arg0: memref<2048x8192xf16>) kernel attributes {known_block_size = array<i32: 8, 1, 16>} {
    %cst = arith.constant dense<0.000000e+00> : vector<8x16xf16>
    %c0 = arith.constant 0 : index
    %tdesc = xegpu.create_nd_tdesc %arg0 : memref<2048x8192xf16> -> !xegpu.tensor_desc<8x16xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
    %loaded = xegpu.load_nd %tdesc[%c0, %c0]  : !xegpu.tensor_desc<8x16xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<8x16xf16>
    %loaded_add = arith.addf %loaded, %cst : vector<8x16xf16>
    // 8 subgroups could load 1x16 each, but the current infra only considers the largest inst size (larger than 1x16) -> fail propagation.
    // expected-error@+1 {{Failed to determine required layout for store_nd.}}
    xegpu.store_nd %loaded_add, %tdesc[%c0, %c0]  : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
    gpu.return
  }
}

// -----
gpu.module @test {
  // No valid sg layout: 32 subgroups over 256 elements forces sg_data = 8 <
  // 16-lane tile, so every candidate is rejected.
  gpu.func @store_scatter_no_valid_sg_layout(%dest: memref<256xf16>) kernel attributes {known_block_size = array<i32: 512, 1, 1>} {
    %val = arith.constant dense<25.5> : vector<256xf16>
    %offset = arith.constant dense<0> : vector<256xindex>
    %mask = arith.constant dense<1> : vector<256xi1>
    // expected-error@+1 {{Failed to determine required layout for store scatter.}}
    xegpu.store %val, %dest[%offset], %mask <{chunk_size = 1, l1_hint = #xegpu.cache_hint<cached>}>
      : vector<256xf16>, memref<256xf16>, vector<256xindex>, vector<256xi1>
    gpu.return
  }
}

// -----
// Without @known_block_size the subgroup count is unknown, which is fatal in
// subgroup mode.
gpu.module @test {
  gpu.func @store_no_known_block_size(%arg0: memref<256x128xf32>) kernel {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0.000000e+00> : vector<256x128xf32>
    %tdesc = xegpu.create_nd_tdesc %arg0 : memref<256x128xf32> -> !xegpu.tensor_desc<256x128xf32>
    // expected-error@+1 {{Unable to determine the number of subgroups for the operation. Please check @known_block_size is properly attached as kernel attributes}}
    xegpu.store_nd %cst, %tdesc[%c0, %c0] : vector<256x128xf32>, !xegpu.tensor_desc<256x128xf32>
    gpu.return
  }
}

// -----
// A @known_block_size covering less than one subgroup leaves no subgroup to
// distribute over.
gpu.module @test {
  gpu.func @store_block_smaller_than_subgroup(%arg0: memref<256x128xf32>) kernel attributes {known_block_size = array<i32: 8, 1, 1>} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0.000000e+00> : vector<256x128xf32>
    %tdesc = xegpu.create_nd_tdesc %arg0 : memref<256x128xf32> -> !xegpu.tensor_desc<256x128xf32>
    // expected-error@+1 {{Unable to determine the number of subgroups for the operation.}}
    xegpu.store_nd %cst, %tdesc[%c0, %c0] : vector<256x128xf32>, !xegpu.tensor_desc<256x128xf32>
    gpu.return
  }
}

// -----
// A non-power-of-two @known_block_size dimension yields a subgroup count that
// cannot be used for distribution.
gpu.module @test {
  gpu.func @store_non_power_of_two_block(%arg0: memref<256x128xf32>) kernel attributes {known_block_size = array<i32: 16, 3, 1>} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0.000000e+00> : vector<256x128xf32>
    %tdesc = xegpu.create_nd_tdesc %arg0 : memref<256x128xf32> -> !xegpu.tensor_desc<256x128xf32>
    // expected-error@+1 {{Unable to determine the number of subgroups for the operation.}}
    xegpu.store_nd %cst, %tdesc[%c0, %c0] : vector<256x128xf32>, !xegpu.tensor_desc<256x128xf32>
    gpu.return
  }
}

// -----
// A 4D 1x1x8x1xf16 block load/store whose innermost dim (1) is below the
// minimum hardware block width used to crash in VectorType::get via a -1
// inst_data entry; it must now fail cleanly at the anchor op.
gpu.module @entry_kernel {
  gpu.func @entry_kernel(%arg0: memref<1x24x1024x1xf16>, %arg1: memref<1x24x1024x1xf16>) kernel attributes {intel_reqd_sub_group_size = 16 : i32, known_block_size = array<i32: 16, 1, 1>} {
    %cst = arith.constant dense<0.000000e+00> : vector<1x1x8x1xf16>
    %block_id_x = gpu.block_id x
    %block_id_y = gpu.block_id y
    %0 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%block_id_y]
    %1 = xegpu.create_nd_tdesc %arg0 : memref<1x24x1024x1xf16> -> !xegpu.tensor_desc<1x1x8x1xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
    %2 = xegpu.load_nd %1[0, %block_id_x, %0, 0]  : !xegpu.tensor_desc<1x1x8x1xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<1x1x8x1xf16>
    %3 = arith.maximumf %2, %cst : vector<1x1x8x1xf16>
    %4 = xegpu.create_nd_tdesc %arg1 : memref<1x24x1024x1xf16> -> !xegpu.tensor_desc<1x1x8x1xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
    // expected-error@+1 {{Failed to determine required layout for store_nd.}}
    xegpu.store_nd %3, %4[0, %block_id_x, %0, 0]  : vector<1x1x8x1xf16>, !xegpu.tensor_desc<1x1x8x1xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
    gpu.return
  }
}

// -----
// scf.while's "after" region argument is tied to no init operand, so its layout
// is only recoverable when the before region forwards a region argument
// unchanged. Here scf.condition forwards a freshly loaded value instead, so
// there is nowhere to attribute the layout the after argument requires.
gpu.module @test {
  func.func @while_after_arg_not_pass_through(%src: memref<256x128xf32>, %cond: i1) {
    %cst = arith.constant dense<0.000000e+00> : vector<256x128xf32>
    %tdesc = xegpu.create_nd_tdesc %src : memref<256x128xf32> -> !xegpu.tensor_desc<256x128xf32>
    %0 = scf.while (%before = %cst) : (vector<256x128xf32>) -> vector<256x128xf32> {
      %loaded = xegpu.load_nd %tdesc[0, 0] : !xegpu.tensor_desc<256x128xf32> -> vector<256x128xf32>
      // expected-error@+2 {{unsupported region structure: the successor argument it feeds is not tied to an init operand, so its value must be passed through from predecessor region argument.}}
      // expected-error@+1 {{Failed to update operation with the layout.}}
      scf.condition(%cond) %loaded : vector<256x128xf32>
    } do {
    ^bb0(%after: vector<256x128xf32>):
      xegpu.store_nd %after, %tdesc[0, 0] <{layout = #xegpu.layout<sg_layout = [8, 4], sg_data = [32, 32]>}>
        : vector<256x128xf32>, !xegpu.tensor_desc<256x128xf32>
      scf.yield %after : vector<256x128xf32>
    }
    return
  }
}
