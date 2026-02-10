//===-- AMDGPUTypes.td - AMDGPU dialect types *- tablegen -*---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AMDGPU_IR_AMDGPUTYPES_TD
#define MLIR_DIALECT_AMDGPU_IR_AMDGPUTYPES_TD

include "mlir/Dialect/AMDGPU/IR/AMDGPUBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

//===----------------------------------------------------------------------===//
// AMDGPU Type definitions
//===----------------------------------------------------------------------===//

class AMDGPU_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<AMDGPU_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def AMDGPU_TDMBaseType : AMDGPU_Type<"TDMBase", "tdm_base"> {
  let summary = "Pair of base addresses that move data between LDS and global storage.";
  let description = [{
    This type is opaque and it is used to represent a struct of two addresses.
    One address is in LDS while the other is in global memory.

    The value defined by this operation is only intended to be used by
    amdgpu.tdm_make_descriptor.
  }];
  let parameters = (ins "Type":$elementType);
  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType), [{
      return $_get(elementType.getContext(), elementType);
    }]>
  ];
  let assemblyFormat = "`<` $elementType `>`";
}

def AMDGPU_TDMGatherBaseType : AMDGPU_Type<"TDMGatherBase", "tdm_gather_base"> {
  let summary = "Pair of base addresses that move data between LDS and global storage.";
  let description = [{
    This type is opaque and it is used to represent a struct of two addresses.
    One address is in LDS while the other is in global memory.

    This operation is similar to amdgpu.tdm_make_base but intended to be
    used in gather mode.

    The value defined by this operation is only intended to be used by
    amdgpu.tdm_make_gather_descriptor.
  }];
  let parameters = (ins "Type":$elementType, "Type":$indexType);
  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType, "Type": $indexType), [{
      return $_get(elementType.getContext(), elementType, indexType);
    }]>
  ];
  let assemblyFormat = "`<` $elementType `,` $indexType`>`";
  let genVerifyDecl = 1;
}

def AMDGPU_TDMDescriptorType : AMDGPU_Type<"TDMDescriptor", "tdm_descriptor"> {
  let summary = "Descriptors used in tensor store/load operations.";
  let description = [{
    This type is opaque and corresponds to the two or four descriptor groups
    used in tensor_load_to_lds or tensor_store_from_lds.
  }];
}

def AMDGPU_DsBarrierStateType : AMDGPU_Type<"DsBarrierState", "ds_barrier_state",
    [MemRefElementTypeInterface]> {
  let summary = "State of an in-LDS barrier.";
  let description = [{
    Type that encodes the state of an in-LDS barrier as used by the atomic barrier
    instructions introduced on gfx1250.

    It consists of a 29-bit count of the number of pending arrivals at the barrier (the
    *pending count*) in bits [28:0], a 3-bit *phase* in bits [31:29], and the 32-bit count
    to re-initialize the pending count to on phase change (the *init count*) in bits [63:32].

    When an instruction (either one of the explicit arrival primitives or tensor data
    movement) *arrives* at such a barrier, the pending count is decremented. If this
    decrement would cause the pending count to underflow, the count is instead reset
    to the init count and the phase is decremented (wrapping back to 0). When the
    phase is decremented, sleeping waves are woken up so they can check the barrier.

    The barrier state resides in LDS, but an old barrier state can be returned from atomic
    arrival instructions or though atomic loads.

    This feature is not available prior to gfx1250.
  }];
}

#endif // MLIR_DIALECT_AMDGPU_IR_AMDGPUTYPES_TD
