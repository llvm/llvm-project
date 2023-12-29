// RUN: mlir-opt --split-input-file --transform-interpreter %s | FileCheck %s

// CHECK-LABEL: func @matmul_divisible
//       CHECK:   scf.forall
//   CHECK-NOT:     memref.copy
//       CHECK:     linalg.fill
//       CHECK:     scf.for
//       CHECK:       memref.alloc() : memref<128x16xf32, 3>
//       CHECK:       scf.forall
//       CHECK:         vector.create_mask
//       CHECK:         vector.transfer_read
//       CHECK:         vector.transfer_write
//       CHECK:       memref.alloc() : memref<16x128xf32, 3>
//       CHECK:       scf.forall
//       CHECK:         vector.create_mask
//       CHECK:         vector.transfer_read
//       CHECK:         vector.transfer_write
//       CHECK:       memref.alloc() : memref<128x128xf32, 3>
//       CHECK:       scf.forall
//       CHECK:         vector.create_mask
//       CHECK:         vector.transfer_read
//       CHECK:         vector.transfer_write
//       CHECK:       linalg.matmul
//       CHECK:       scf.forall
//       CHECK:         vector.transfer_read
//       CHECK:         vector.transfer_write
func.func @matmul_divisible(%A: tensor<1024x1024xf32>,
                            %B: tensor<1024x1024xf32>,
                            %C: tensor<1024x1024xf32>)
    -> tensor<1024x1024xf32>
{
  %cst = arith.constant 0.000000e+00 : f32
  %0 = linalg.fill ins(%cst : f32)
                   outs(%C : tensor<1024x1024xf32>)
      -> tensor<1024x1024xf32>
  %1 = linalg.matmul ins(%A, %B : tensor<1024x1024xf32>, tensor<1024x1024xf32>)
                     outs(%0 : tensor<1024x1024xf32>)
      -> tensor<1024x1024xf32>
  return %1 : tensor<1024x1024xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.consumed}) {
    // Fuse linalg.fill into linalg.matmul and tile.
    %matmul_op = transform.structured.match ops{["linalg.matmul"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %fill_op = transform.structured.match ops{["linalg.fill"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %tiled_matmul_op, %forall_op = transform.structured.tile_using_forall %matmul_op num_threads [] tile_sizes [128, 128](mapping = [#gpu.block<y>, #gpu.block<x>])
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_op, %new_containing_op = transform.structured.fuse_into_containing_op %fill_op into %forall_op
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Tile linalg.matmul a second time.
    %tiled_linalg_op, %loops = transform.structured.tile_using_for %tiled_matmul_op[0, 0, 16] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Pad linalg.matmul.
    %padded, %pad, %copy_back = transform.structured.pad %tiled_linalg_op
        {padding_values=[0.0 : f32, 0.0 : f32, 0.0 : f32],
         padding_dimensions=[0, 1, 2], pack_paddings=[1, 1, 1],
         copy_back_op = "linalg.copy"}
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Map and tile tensor.pad.
    %pad_forall_op, %tiled_pad_op = transform.structured.gpu.map_copy_to_threads
        %pad total_num_threads = 32 desired_bit_alignment = 128
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.foreach %pad_forall_op : !transform.any_op {
    ^bb2(%arg2 : !transform.any_op):
      %if_op = transform.structured.match ops{["scf.if"]} in %arg2
          : (!transform.any_op) -> !transform.any_op
      // TODO: The scf.if can be avoided with 0x... tensors.
      transform.scf.take_assumed_branch %if_op take_else_branch
          : (!transform.any_op) -> ()
    }

    // Map and tile copy back.
    %copy_forall_op, %tiled_copy_op = transform.structured.gpu.map_copy_to_threads
        %copy_back total_num_threads = 32 desired_bit_alignment = 128
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Apply masked vectorization to padding ops.
    transform.structured.vectorize %tiled_pad_op vector_sizes [128, 4]
        : !transform.any_op

    // Assign shared memory buffer to padding.
    %buffer, %new_ops = transform.structured.bufferize_to_allocation
        %pad_forall_op {memory_space = 3, bufferize_destination_only, emit_dealloc}
        : !transform.any_op

    // Bufferize.
    %func_op_1 = transform.structured.match ops{["func.func"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    transform.bufferization.eliminate_empty_tensors %func_op_1 : !transform.any_op
    transform.apply_dce to %func_op_1 : !transform.any_op
    transform.apply_cse to %func_op_1 : !transform.any_op
    %bufferized = transform.bufferization.one_shot_bufferize
        layout{IdentityLayoutMap} %arg1 {bufferize_function_boundaries=true}
        : (!transform.any_op) -> !transform.any_op

    // Apply vectorization to copy back from shared memory.
    // TODO: Find a way to retain the handle to linalg.copy throughout
    // bufferization.
    %func_op_2 = transform.structured.match ops{["func.func"]} in %bufferized
        : (!transform.any_op) -> !transform.any_op
    %bufferized_copy_back = transform.structured.match ops{["linalg.copy"]} in %func_op_2
        : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize
        %bufferized_copy_back vector_sizes [128, 4] : !transform.any_op

    // Canonicalize, cleanup and vector lowering. This step also removes buffer
    // self-copies.
    transform.apply_patterns to %func_op_2 {
      transform.apply_patterns.canonicalization
      transform.apply_patterns.vector.lower_masked_transfers
    } {apply_cse} : !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @matmul_not_divisible
//       CHECK:   scf.forall
//   CHECK-NOT:     memref.copy
//       CHECK:     linalg.fill
//       CHECK:     scf.for
//       CHECK:       memref.alloc() : memref<128x16xf32, 3>
//       CHECK:       scf.forall
//       CHECK:         vector.create_mask
//       CHECK:         vector.transfer_read
//       CHECK:         vector.transfer_write
//       CHECK:       memref.alloc() : memref<16x128xf32, 3>
//       CHECK:       scf.forall
//       CHECK:         vector.create_mask
//       CHECK:         vector.transfer_read
//       CHECK:         vector.transfer_write
//       CHECK:       memref.alloc() : memref<128x128xf32, 3>
//       CHECK:       scf.forall
//       CHECK:         vector.create_mask
//       CHECK:         vector.transfer_read
//       CHECK:         vector.transfer_write
//       CHECK:       linalg.matmul
//       CHECK:       vector.transfer_read
//       CHECK:       vector.transfer_write
func.func @matmul_not_divisible(%A: tensor<1023x1023xf32>,
                                %B: tensor<1023x1023xf32>,
                                %C: tensor<1023x1023xf32>)
    -> tensor<1023x1023xf32>
{
  %cst = arith.constant 0.000000e+00 : f32
  %0 = linalg.fill ins(%cst : f32)
                   outs(%C : tensor<1023x1023xf32>)
      -> tensor<1023x1023xf32>
  %1 = linalg.matmul ins(%A, %B : tensor<1023x1023xf32>, tensor<1023x1023xf32>)
                     outs(%0 : tensor<1023x1023xf32>)
      -> tensor<1023x1023xf32>
  return %1 : tensor<1023x1023xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.consumed}) {
    // Fuse linalg.fill into linalg.matmul and tile.
    %matmul_op = transform.structured.match ops{["linalg.matmul"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %fill_op = transform.structured.match ops{["linalg.fill"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    %tiled_matmul_op, %forall_op = transform.structured.tile_using_forall %matmul_op num_threads [] tile_sizes [128, 128](mapping = [#gpu.block<y>, #gpu.block<x>])
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %fused_op, %new_containing_op = transform.structured.fuse_into_containing_op %fill_op into %forall_op
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Tile linalg.matmul a second time.
    %tiled_linalg_op, %loops = transform.structured.tile_using_for %tiled_matmul_op[0, 0, 16] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Pad linalg.matmul.
    %padded, %pad, %copy_back = transform.structured.pad %tiled_linalg_op
        {padding_values=[0.0 : f32, 0.0 : f32, 0.0 : f32],
         padding_dimensions=[0, 1, 2], pack_paddings=[1, 1, 1],
         copy_back_op = "linalg.copy"}
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Map and tile tensor.pad.
    %pad_forall_op, %tiled_pad_op = transform.structured.gpu.map_copy_to_threads
        %pad total_num_threads = 32 desired_bit_alignment = 128
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.foreach %pad_forall_op : !transform.any_op {
    ^bb2(%arg2 : !transform.any_op):
      %if_op = transform.structured.match ops{["scf.if"]} in %arg2
          : (!transform.any_op) -> !transform.any_op
      // TODO: The scf.if can be avoided with 0x... tensors.
      transform.scf.take_assumed_branch %if_op take_else_branch
          : (!transform.any_op) -> ()
    }

    // Apply masked vectorization to padding ops.
    transform.structured.vectorize %tiled_pad_op vector_sizes [128, 4]
        : !transform.any_op

    // Assign shared memory buffer to padding.
    %buffer, %new_ops = transform.structured.bufferize_to_allocation
        %pad_forall_op {memory_space = 3, bufferize_destination_only, emit_dealloc}
        : !transform.any_op

    // Bufferize.
    %func_op_1 = transform.structured.match ops{["func.func"]} in %arg1
        : (!transform.any_op) -> !transform.any_op
    transform.bufferization.eliminate_empty_tensors %func_op_1 : !transform.any_op
    transform.apply_dce to %func_op_1 : !transform.any_op
    transform.apply_cse to %func_op_1 : !transform.any_op
    %bufferized = transform.bufferization.one_shot_bufferize
        layout{IdentityLayoutMap} %arg1 {bufferize_function_boundaries=true}
        : (!transform.any_op) -> !transform.any_op

    // Apply vectorization to copy back from shared memory.
    // TODO: Find a way to retain the handle to linalg.copy throughout
    // bufferization.
    %func_op_2 = transform.structured.match ops{["func.func"]} in %bufferized
        : (!transform.any_op) -> !transform.any_op
    %bufferized_copy_back = transform.structured.match ops{["linalg.copy"]} in %func_op_2
        : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize
        %bufferized_copy_back vector_sizes [128, 4] : !transform.any_op

    // Canonicalize, cleanup and vector lowering. This step also removes buffer
    // self-copies.
    transform.apply_patterns to %func_op_2 {
      transform.apply_patterns.canonicalization
      transform.apply_patterns.vector.lower_masked_transfers
    } {apply_cse} : !transform.any_op
    transform.yield
  }
}
