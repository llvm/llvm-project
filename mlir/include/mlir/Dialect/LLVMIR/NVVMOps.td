//===-- NVVMOps.td - NVVM IR dialect op definition file ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the NVVM IR operation definition file.
//
//===----------------------------------------------------------------------===//

#ifndef NVVMIR_OPS
#define NVVMIR_OPS

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// NVVM dialect definitions
//===----------------------------------------------------------------------===//

def NVVM_Dialect : Dialect {
  let name = "nvvm";
  let cppNamespace = "::mlir::NVVM";
  let dependentDialects = ["LLVM::LLVMDialect"];
}

//===----------------------------------------------------------------------===//
// NVVM op definitions
//===----------------------------------------------------------------------===//

class NVVM_Op<string mnemonic, list<OpTrait> traits = []> :
  LLVM_OpBase<NVVM_Dialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// NVVM intrinsic operations
//===----------------------------------------------------------------------===//

class NVVM_IntrOp<string mnem, list<int> overloadedResults,
                  list<int> overloadedOperands, list<OpTrait> traits,
                  int numResults>
  : LLVM_IntrOpBase<NVVM_Dialect, mnem, "nvvm_" # !subst(".", "_", mnem),
                    overloadedResults, overloadedOperands, traits, numResults>;


//===----------------------------------------------------------------------===//
// NVVM special register op definitions
//===----------------------------------------------------------------------===//

class NVVM_SpecialRegisterOp<string mnemonic,
    list<OpTrait> traits = []> :
  NVVM_IntrOp<mnemonic, [], [], !listconcat(traits, [NoSideEffect]), 1>,
  Arguments<(ins)> {
  let assemblyFormat = "attr-dict `:` type($res)";
}

//===----------------------------------------------------------------------===//
// Lane index and range
def NVVM_LaneIdOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.laneid">;
def NVVM_WarpSizeOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.warpsize">;

//===----------------------------------------------------------------------===//
// Thread index and range
def NVVM_ThreadIdXOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.tid.x">;
def NVVM_ThreadIdYOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.tid.y">;
def NVVM_ThreadIdZOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.tid.z">;
def NVVM_BlockDimXOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.ntid.x">;
def NVVM_BlockDimYOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.ntid.y">;
def NVVM_BlockDimZOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.ntid.z">;

//===----------------------------------------------------------------------===//
// Block index and range
def NVVM_BlockIdXOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.ctaid.x">;
def NVVM_BlockIdYOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.ctaid.y">;
def NVVM_BlockIdZOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.ctaid.z">;
def NVVM_GridDimXOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.nctaid.x">;
def NVVM_GridDimYOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.nctaid.y">;
def NVVM_GridDimZOp : NVVM_SpecialRegisterOp<"read.ptx.sreg.nctaid.z">;

//===----------------------------------------------------------------------===//
// NVVM synchronization op definitions
//===----------------------------------------------------------------------===//

def NVVM_Barrier0Op : NVVM_Op<"barrier0"> {
  string llvmBuilder = [{
      createIntrinsicCall(builder, llvm::Intrinsic::nvvm_barrier0);
  }];
  let assemblyFormat = "attr-dict";
}

def NVVM_ShflBflyOp :
  NVVM_Op<"shfl.sync.bfly">,
  Results<(outs LLVM_Type:$res)>,
  Arguments<(ins LLVM_Type:$dst,
                 LLVM_Type:$val,
                 LLVM_Type:$offset,
                 LLVM_Type:$mask_and_clamp,
                 OptionalAttr<UnitAttr>:$return_value_and_is_valid)> {
  string llvmBuilder = [{
      auto intId = getShflBflyIntrinsicId(
          $_resultType, static_cast<bool>($return_value_and_is_valid));
      $res = createIntrinsicCall(builder,
          intId, {$dst, $val, $offset, $mask_and_clamp});
  }];
  let parser = [{ return parseNVVMShflSyncBflyOp(parser, result); }];
  let printer = [{ printNVVMIntrinsicOp(p, this->getOperation()); }];
  let verifier = [{
    if (!(*this)->getAttrOfType<UnitAttr>("return_value_and_is_valid"))
      return success();
    auto type = getType().dyn_cast<LLVM::LLVMStructType>();
    auto elementType = (type && type.getBody().size() == 2)
                     ? type.getBody()[1].dyn_cast<IntegerType>()
                     : nullptr;
    if (!elementType || elementType.getWidth() != 1)
      return emitError("expected return type to be a two-element struct with "
                       "i1 as the second element");
    return success();
  }];
}

def NVVM_VoteBallotOp :
  NVVM_Op<"vote.ballot.sync">,
  Results<(outs LLVM_Type:$res)>,
  Arguments<(ins LLVM_Type:$mask, LLVM_Type:$pred)> {
  string llvmBuilder = [{
      $res = createIntrinsicCall(builder,
            llvm::Intrinsic::nvvm_vote_ballot_sync, {$mask, $pred});
  }];
  let parser = [{ return parseNVVMVoteBallotOp(parser, result); }];
  let printer = [{ printNVVMIntrinsicOp(p, this->getOperation()); }];
}

def NVVM_MmaOp :
  NVVM_Op<"mma.sync">,
  Results<(outs LLVM_Type:$res)>,
  Arguments<(ins Variadic<LLVM_Type>:$args)> {
  string llvmBuilder = [{
    $res = createIntrinsicCall(
        builder, llvm::Intrinsic::nvvm_mma_m8n8k4_row_col_f32_f32, $args);
  }];
  let assemblyFormat = "$args attr-dict `:` functional-type($args, $res)";
  let verifier = [{ return ::verify(*this); }];
}

#endif // NVVMIR_OPS
