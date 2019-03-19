//===- OCLUtil.cpp - OCL Utilities ----------------------------------------===//
//
//                     The LLVM/SPIRV Translator
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// Copyright (c) 2014 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimers in the documentation
// and/or other materials provided with the distribution.
// Neither the names of Advanced Micro Devices, Inc., nor the names of its
// contributors may be used to endorse or promote products derived from this
// Software without specific prior written permission.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.
//
//===----------------------------------------------------------------------===//
//
// This file implements OCL utility functions.
//
//===----------------------------------------------------------------------===//
#define DEBUG_TYPE "oclutil"

#include "OCLUtil.h"
#include "SPIRVEntry.h"
#include "SPIRVFunction.h"
#include "SPIRVInstruction.h"
#include "SPIRVInternal.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/PassSupport.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace llvm;
using namespace SPIRV;

namespace OCLUtil {

#ifndef SPIRV_OCL_SPECIAL_TYPES_DEFAULT_ADDR_SPACE
#define SPIRV_OCL_SPECIAL_TYPES_DEFAULT_ADDR_SPACE SPIRAS_Private
#endif

#ifndef SPIRV_QUEUE_T_ADDR_SPACE
#define SPIRV_QUEUE_T_ADDR_SPACE SPIRV_OCL_SPECIAL_TYPES_DEFAULT_ADDR_SPACE
#endif

#ifndef SPIRV_EVENT_T_ADDR_SPACE
#define SPIRV_EVENT_T_ADDR_SPACE SPIRV_OCL_SPECIAL_TYPES_DEFAULT_ADDR_SPACE
#endif

#ifndef SPIRV_CLK_EVENT_T_ADDR_SPACE
#define SPIRV_CLK_EVENT_T_ADDR_SPACE SPIRV_OCL_SPECIAL_TYPES_DEFAULT_ADDR_SPACE
#endif

#ifndef SPIRV_SAMPLER_T_ADDR_SPACE
#define SPIRV_SAMPLER_T_ADDR_SPACE SPIRV_OCL_SPECIAL_TYPES_DEFAULT_ADDR_SPACE
#endif

#ifndef SPIRV_RESERVE_ID_T_ADDR_SPACE
#define SPIRV_RESERVE_ID_T_ADDR_SPACE SPIRV_OCL_SPECIAL_TYPES_DEFAULT_ADDR_SPACE
#endif
// Excerpt from SPIR 2.0 spec.:
//   Pipe objects are represented using pointers to the opaque %opencl.pipe LLVM
//   structure type which reside in the global address space.
#ifndef SPIRV_PIPE_ADDR_SPACE
#define SPIRV_PIPE_ADDR_SPACE SPIRAS_Global
#endif
// Excerpt from SPIR 2.0 spec.:
//   Note: Images data types reside in global memory and hence should be marked
//   as such in the "kernel arg addr space" metadata.
#ifndef SPIRV_IMAGE_ADDR_SPACE
#define SPIRV_IMAGE_ADDR_SPACE SPIRAS_Global
#endif

///////////////////////////////////////////////////////////////////////////////
//
// Functions for getting builtin call info
//
///////////////////////////////////////////////////////////////////////////////
AtomicWorkItemFenceLiterals getAtomicWorkItemFenceLiterals(CallInst *CI) {
  return std::make_tuple(getArgAsInt(CI, 0),
                         static_cast<OCLMemOrderKind>(getArgAsInt(CI, 1)),
                         static_cast<OCLScopeKind>(getArgAsInt(CI, 2)));
}

size_t getAtomicBuiltinNumMemoryOrderArgs(StringRef Name) {
  if (Name.startswith("atomic_compare_exchange"))
    return 2;
  return 1;
}

bool isComputeAtomicOCLBuiltin(StringRef DemangledName) {
  if (!DemangledName.startswith(kOCLBuiltinName::AtomicPrefix) &&
      !DemangledName.startswith(kOCLBuiltinName::AtomPrefix))
    return false;

  return llvm::StringSwitch<bool>(DemangledName)
      .EndsWith("add", true)
      .EndsWith("sub", true)
      .EndsWith("inc", true)
      .EndsWith("dec", true)
      .EndsWith("cmpxchg", true)
      .EndsWith("min", true)
      .EndsWith("max", true)
      .EndsWith("and", true)
      .EndsWith("or", true)
      .EndsWith("xor", true)
      .EndsWith("add_explicit", true)
      .EndsWith("sub_explicit", true)
      .EndsWith("or_explicit", true)
      .EndsWith("xor_explicit", true)
      .EndsWith("and_explicit", true)
      .EndsWith("min_explicit", true)
      .EndsWith("max_explicit", true)
      .Default(false);
}

BarrierLiterals getBarrierLiterals(CallInst *CI) {
  auto N = CI->getNumArgOperands();
  assert(N == 1 || N == 2);

  std::string DemangledName;
  assert(CI->getCalledFunction() && "Unexpected indirect call");
  if (!oclIsBuiltin(CI->getCalledFunction()->getName(), &DemangledName)) {
    assert(0 &&
           "call must a builtin (work_group_barrier or sub_group_barrier)");
  }

  OCLScopeKind Scope = OCLMS_work_group;
  if (DemangledName == kOCLBuiltinName::SubGroupBarrier) {
    Scope = OCLMS_sub_group;
  }

  return std::make_tuple(getArgAsInt(CI, 0),
                         N == 1 ? OCLMS_work_group
                                : static_cast<OCLScopeKind>(getArgAsInt(CI, 1)),
                         Scope);
}

unsigned getExtOp(StringRef OrigName, const std::string &GivenDemangledName) {
  std::string DemangledName = GivenDemangledName;
  if (!oclIsBuiltin(OrigName, DemangledName.empty() ? &DemangledName : nullptr))
    return ~0U;
  LLVM_DEBUG(dbgs() << "getExtOp: demangled name: " << DemangledName << '\n');
  OCLExtOpKind EOC;
  bool Found = OCLExtOpMap::rfind(DemangledName, &EOC);
  if (!Found) {
    std::string Prefix;
    switch (lastFuncParamType(OrigName)) {
    case ParamType::UNSIGNED:
      Prefix = "u_";
      break;
    case ParamType::SIGNED:
      Prefix = "s_";
      break;
    case ParamType::FLOAT:
      Prefix = "f";
      break;
    default:
      llvm_unreachable("unknown mangling!");
    }
    Found = OCLExtOpMap::rfind(Prefix + DemangledName, &EOC);
  }
  if (Found)
    return EOC;
  else
    return ~0U;
}

std::unique_ptr<SPIRVEntry> getSPIRVInst(const OCLBuiltinTransInfo &Info) {
  Op OC = OpNop;
  unsigned ExtOp = ~0U;
  SPIRVEntry *Entry = nullptr;
  if (OCLSPIRVBuiltinMap::find(Info.UniqName, &OC))
    Entry = SPIRVEntry::create(OC);
  else if ((ExtOp = getExtOp(Info.MangledName, Info.UniqName)) != ~0U)
    Entry = static_cast<SPIRVEntry *>(
        SPIRVEntry::createUnique(SPIRVEIS_OpenCL, ExtOp).get());
  return std::unique_ptr<SPIRVEntry>(Entry);
}

///////////////////////////////////////////////////////////////////////////////
//
// Functions for getting module info
//
///////////////////////////////////////////////////////////////////////////////

unsigned encodeOCLVer(unsigned short Major, unsigned char Minor,
                      unsigned char Rev) {
  return (Major * 100 + Minor) * 1000 + Rev;
}

std::tuple<unsigned short, unsigned char, unsigned char>
decodeOCLVer(unsigned Ver) {
  unsigned short Major = Ver / 100000;
  unsigned char Minor = (Ver % 100000) / 1000;
  unsigned char Rev = Ver % 1000;
  return std::make_tuple(Major, Minor, Rev);
}

unsigned getOCLVersion(Module *M, bool AllowMulti) {
  NamedMDNode *NamedMD = M->getNamedMetadata(kSPIR2MD::OCLVer);
  if (!NamedMD)
    return 0;
  assert(NamedMD->getNumOperands() > 0 && "Invalid SPIR");
  if (!AllowMulti && NamedMD->getNumOperands() != 1)
    report_fatal_error("Multiple OCL version metadata not allowed");

  // If the module was linked with another module, there may be multiple
  // operands.
  auto GetVer = [=](unsigned I) {
    auto MD = NamedMD->getOperand(I);
    return std::make_pair(getMDOperandAsInt(MD, 0), getMDOperandAsInt(MD, 1));
  };
  auto Ver = GetVer(0);
  for (unsigned I = 1, E = NamedMD->getNumOperands(); I != E; ++I)
    if (Ver != GetVer(I))
      report_fatal_error("OCL version mismatch");

  return encodeOCLVer(Ver.first, Ver.second, 0);
}

void decodeMDNode(MDNode *N, unsigned &X, unsigned &Y, unsigned &Z) {
  if (N == NULL)
    return;
  X = getMDOperandAsInt(N, 0);
  Y = getMDOperandAsInt(N, 1);
  Z = getMDOperandAsInt(N, 2);
}

/// Encode LLVM type by SPIR-V execution mode VecTypeHint
unsigned encodeVecTypeHint(Type *Ty) {
  if (Ty->isHalfTy())
    return 4;
  if (Ty->isFloatTy())
    return 5;
  if (Ty->isDoubleTy())
    return 6;
  if (IntegerType *IntTy = dyn_cast<IntegerType>(Ty)) {
    switch (IntTy->getIntegerBitWidth()) {
    case 8:
      return 0;
    case 16:
      return 1;
    case 32:
      return 2;
    case 64:
      return 3;
    default:
      llvm_unreachable("invalid integer type");
    }
  }
  if (VectorType *VecTy = dyn_cast<VectorType>(Ty)) {
    Type *EleTy = VecTy->getElementType();
    unsigned Size = VecTy->getVectorNumElements();
    return Size << 16 | encodeVecTypeHint(EleTy);
  }
  llvm_unreachable("invalid type");
  return ~0U;
}

Type *decodeVecTypeHint(LLVMContext &C, unsigned Code) {
  unsigned VecWidth = Code >> 16;
  unsigned Scalar = Code & 0xFFFF;
  Type *ST = nullptr;
  switch (Scalar) {
  case 0:
  case 1:
  case 2:
  case 3:
    ST = IntegerType::get(C, 1 << (3 + Scalar));
    break;
  case 4:
    ST = Type::getHalfTy(C);
    break;
  case 5:
    ST = Type::getFloatTy(C);
    break;
  case 6:
    ST = Type::getDoubleTy(C);
    break;
  default:
    llvm_unreachable("Invalid vec type hint");
    return nullptr;
  }
  if (VecWidth < 1)
    return ST;
  return VectorType::get(ST, VecWidth);
}

unsigned transVecTypeHint(MDNode *Node) {
  return encodeVecTypeHint(getMDOperandAsType(Node, 0));
}

SPIRAddressSpace getOCLOpaqueTypeAddrSpace(Op OpCode) {
  switch (OpCode) {
  case OpTypeQueue:
    return SPIRV_QUEUE_T_ADDR_SPACE;
  case OpTypeEvent:
    return SPIRV_EVENT_T_ADDR_SPACE;
  case OpTypeDeviceEvent:
    return SPIRV_CLK_EVENT_T_ADDR_SPACE;
  case OpTypeReserveId:
    return SPIRV_RESERVE_ID_T_ADDR_SPACE;
  case OpTypePipe:
  case OpTypePipeStorage:
    return SPIRV_PIPE_ADDR_SPACE;
  case OpTypeImage:
  case OpTypeSampledImage:
    return SPIRV_IMAGE_ADDR_SPACE;
  case OpConstantSampler:
  case OpTypeSampler:
    return SPIRV_SAMPLER_T_ADDR_SPACE;
  default:
    assert(false && "No address space is determined for some OCL type");
    return SPIRV_OCL_SPECIAL_TYPES_DEFAULT_ADDR_SPACE;
  }
}

static SPIR::TypeAttributeEnum mapAddrSpaceEnums(SPIRAddressSpace Addrspace) {
  switch (Addrspace) {
  case SPIRAS_Private:
    return SPIR::ATTR_PRIVATE;
  case SPIRAS_Global:
    return SPIR::ATTR_GLOBAL;
  case SPIRAS_Constant:
    return SPIR::ATTR_CONSTANT;
  case SPIRAS_Local:
    return SPIR::ATTR_LOCAL;
  case SPIRAS_Generic:
    return SPIR::ATTR_GENERIC;
  default:
    llvm_unreachable("Invalid addrspace enum member");
  }
  return SPIR::ATTR_NONE;
}

SPIR::TypeAttributeEnum
getOCLOpaqueTypeAddrSpace(SPIR::TypePrimitiveEnum Prim) {
  switch (Prim) {
  case SPIR::PRIMITIVE_QUEUE_T:
    return mapAddrSpaceEnums(SPIRV_QUEUE_T_ADDR_SPACE);
  case SPIR::PRIMITIVE_EVENT_T:
    return mapAddrSpaceEnums(SPIRV_EVENT_T_ADDR_SPACE);
  case SPIR::PRIMITIVE_CLK_EVENT_T:
    return mapAddrSpaceEnums(SPIRV_CLK_EVENT_T_ADDR_SPACE);
  case SPIR::PRIMITIVE_RESERVE_ID_T:
    return mapAddrSpaceEnums(SPIRV_RESERVE_ID_T_ADDR_SPACE);
  case SPIR::PRIMITIVE_PIPE_RO_T:
  case SPIR::PRIMITIVE_PIPE_WO_T:
    return mapAddrSpaceEnums(SPIRV_PIPE_ADDR_SPACE);
  case SPIR::PRIMITIVE_IMAGE1D_RO_T:
  case SPIR::PRIMITIVE_IMAGE1D_ARRAY_RO_T:
  case SPIR::PRIMITIVE_IMAGE1D_BUFFER_RO_T:
  case SPIR::PRIMITIVE_IMAGE2D_RO_T:
  case SPIR::PRIMITIVE_IMAGE2D_ARRAY_RO_T:
  case SPIR::PRIMITIVE_IMAGE2D_DEPTH_RO_T:
  case SPIR::PRIMITIVE_IMAGE2D_ARRAY_DEPTH_RO_T:
  case SPIR::PRIMITIVE_IMAGE2D_MSAA_RO_T:
  case SPIR::PRIMITIVE_IMAGE2D_ARRAY_MSAA_RO_T:
  case SPIR::PRIMITIVE_IMAGE2D_MSAA_DEPTH_RO_T:
  case SPIR::PRIMITIVE_IMAGE2D_ARRAY_MSAA_DEPTH_RO_T:
  case SPIR::PRIMITIVE_IMAGE3D_RO_T:
  case SPIR::PRIMITIVE_IMAGE1D_WO_T:
  case SPIR::PRIMITIVE_IMAGE1D_ARRAY_WO_T:
  case SPIR::PRIMITIVE_IMAGE1D_BUFFER_WO_T:
  case SPIR::PRIMITIVE_IMAGE2D_WO_T:
  case SPIR::PRIMITIVE_IMAGE2D_ARRAY_WO_T:
  case SPIR::PRIMITIVE_IMAGE2D_DEPTH_WO_T:
  case SPIR::PRIMITIVE_IMAGE2D_ARRAY_DEPTH_WO_T:
  case SPIR::PRIMITIVE_IMAGE2D_MSAA_WO_T:
  case SPIR::PRIMITIVE_IMAGE2D_ARRAY_MSAA_WO_T:
  case SPIR::PRIMITIVE_IMAGE2D_MSAA_DEPTH_WO_T:
  case SPIR::PRIMITIVE_IMAGE2D_ARRAY_MSAA_DEPTH_WO_T:
  case SPIR::PRIMITIVE_IMAGE3D_WO_T:
  case SPIR::PRIMITIVE_IMAGE1D_RW_T:
  case SPIR::PRIMITIVE_IMAGE1D_ARRAY_RW_T:
  case SPIR::PRIMITIVE_IMAGE1D_BUFFER_RW_T:
  case SPIR::PRIMITIVE_IMAGE2D_RW_T:
  case SPIR::PRIMITIVE_IMAGE2D_ARRAY_RW_T:
  case SPIR::PRIMITIVE_IMAGE2D_DEPTH_RW_T:
  case SPIR::PRIMITIVE_IMAGE2D_ARRAY_DEPTH_RW_T:
  case SPIR::PRIMITIVE_IMAGE2D_MSAA_RW_T:
  case SPIR::PRIMITIVE_IMAGE2D_ARRAY_MSAA_RW_T:
  case SPIR::PRIMITIVE_IMAGE2D_MSAA_DEPTH_RW_T:
  case SPIR::PRIMITIVE_IMAGE2D_ARRAY_MSAA_DEPTH_RW_T:
  case SPIR::PRIMITIVE_IMAGE3D_RW_T:
    return mapAddrSpaceEnums(SPIRV_IMAGE_ADDR_SPACE);
  default:
    llvm_unreachable("No address space is determined for a SPIR primitive");
  }
  return SPIR::ATTR_NONE;
}

// Fetch type of invoke function passed to device execution built-ins
static FunctionType *getBlockInvokeTy(Function *F, unsigned BlockIdx) {
  auto Params = F->getFunctionType()->params();
  PointerType *FuncPtr = cast<PointerType>(Params[BlockIdx]);
  return cast<FunctionType>(FuncPtr->getElementType());
}

class OCLBuiltinFuncMangleInfo : public SPIRV::BuiltinFuncMangleInfo {
public:
  OCLBuiltinFuncMangleInfo(Function *F) : F(F) {}
  void init(const std::string &UniqName) override {
    UnmangledName = UniqName;
    size_t Pos = std::string::npos;

    if (UnmangledName.find("async_work_group") == 0) {
      addUnsignedArg(-1);
      setArgAttr(1, SPIR::ATTR_CONST);
    } else if (UnmangledName.find("write_imageui") == 0)
      addUnsignedArg(2);
    else if (UnmangledName == "prefetch") {
      addUnsignedArg(1);
      setArgAttr(0, SPIR::ATTR_CONST);
    } else if (UnmangledName == "get_kernel_work_group_size" ||
               UnmangledName ==
                   "get_kernel_preferred_work_group_size_multiple") {
      assert(F && "lack of necessary information");
      const size_t BlockArgIdx = 0;
      FunctionType *InvokeTy = getBlockInvokeTy(F, BlockArgIdx);
      if (InvokeTy->getNumParams() > 1)
        setLocalArgBlock(BlockArgIdx);
    } else if (UnmangledName == "enqueue_kernel") {
      assert(F && "lack of necessary information");
      setEnumArg(1, SPIR::PRIMITIVE_KERNEL_ENQUEUE_FLAGS_T);
      addUnsignedArg(3);
      setArgAttr(4, SPIR::ATTR_CONST);
      // If there are arguments other then block context then these are pointers
      // to local memory so this built-in must be mangled accordingly.
      const size_t BlockArgIdx = 6;
      FunctionType *InvokeTy = getBlockInvokeTy(F, BlockArgIdx);
      if (InvokeTy->getNumParams() > 1) {
        setLocalArgBlock(BlockArgIdx);
        addUnsignedArg(BlockArgIdx + 1);
        setVarArg(BlockArgIdx + 2);
      }
    } else if (UnmangledName.find("get_") == 0 || UnmangledName == "nan" ||
               UnmangledName == "mem_fence" ||
               UnmangledName.find("shuffle") == 0) {
      addUnsignedArg(-1);
      if (UnmangledName.find(kOCLBuiltinName::GetFence) == 0) {
        setArgAttr(0, SPIR::ATTR_CONST);
        addVoidPtrArg(0);
      }
    } else if (UnmangledName.find("barrier") == 0 ||
               UnmangledName.find("work_group_barrier") == 0 ||
               UnmangledName.find("sub_group_barrier") == 0) {
      addUnsignedArg(0);
    } else if (UnmangledName.find("atomic_work_item_fence") == 0) {
      addUnsignedArg(0);
      setEnumArg(1, SPIR::PRIMITIVE_MEMORY_ORDER);
      setEnumArg(2, SPIR::PRIMITIVE_MEMORY_SCOPE);
    } else if (UnmangledName.find("atomic") == 0) {
      setArgAttr(0, SPIR::ATTR_VOLATILE);
      if (UnmangledName.find("atomic_umax") == 0 ||
          UnmangledName.find("atomic_umin") == 0) {
        addUnsignedArg(0);
        addUnsignedArg(1);
        UnmangledName.erase(7, 1);
      } else if (UnmangledName.find("atomic_fetch_umin") == 0 ||
                 UnmangledName.find("atomic_fetch_umax") == 0) {
        addUnsignedArg(0);
        addUnsignedArg(1);
        UnmangledName.erase(13, 1);
      }
      if (UnmangledName.find("store_explicit") != std::string::npos ||
          UnmangledName.find("exchange_explicit") != std::string::npos ||
          (UnmangledName.find("atomic_fetch") == 0 &&
           UnmangledName.find("explicit") != std::string::npos)) {
        setEnumArg(2, SPIR::PRIMITIVE_MEMORY_ORDER);
        setEnumArg(3, SPIR::PRIMITIVE_MEMORY_SCOPE);
      } else if (UnmangledName.find("load_explicit") != std::string::npos ||
                 (UnmangledName.find("atomic_flag") == 0 &&
                  UnmangledName.find("explicit") != std::string::npos)) {
        setEnumArg(1, SPIR::PRIMITIVE_MEMORY_ORDER);
        setEnumArg(2, SPIR::PRIMITIVE_MEMORY_SCOPE);
      } else if (UnmangledName.find("compare_exchange_strong_explicit") !=
                     std::string::npos ||
                 UnmangledName.find("compare_exchange_weak_explicit") !=
                     std::string::npos) {
        setEnumArg(3, SPIR::PRIMITIVE_MEMORY_ORDER);
        setEnumArg(4, SPIR::PRIMITIVE_MEMORY_ORDER);
        setEnumArg(5, SPIR::PRIMITIVE_MEMORY_SCOPE);
      }
      // Don't set atomic property to the first argument of 1.2 atomic
      // built-ins.
      if (UnmangledName.find("atomic_add") != 0 &&
          UnmangledName.find("atomic_sub") != 0 &&
          UnmangledName.find("atomic_xchg") != 0 &&
          UnmangledName.find("atomic_inc") != 0 &&
          UnmangledName.find("atomic_dec") != 0 &&
          UnmangledName.find("atomic_cmpxchg") != 0 &&
          UnmangledName.find("atomic_min") != 0 &&
          UnmangledName.find("atomic_max") != 0 &&
          UnmangledName.find("atomic_and") != 0 &&
          UnmangledName.find("atomic_or") != 0 &&
          UnmangledName.find("atomic_xor") != 0 &&
          UnmangledName.find("atom_") != 0) {
        addAtomicArg(0);
      }

    } else if (UnmangledName.find("uconvert_") == 0) {
      addUnsignedArg(0);
      UnmangledName.erase(0, 1);
    } else if (UnmangledName.find("s_") == 0) {
      UnmangledName.erase(0, 2);
    } else if (UnmangledName.find("u_") == 0) {
      addUnsignedArg(-1);
      UnmangledName.erase(0, 2);
    } else if (UnmangledName == "fclamp") {
      UnmangledName.erase(0, 1);
    }
    // handle [read|write]pipe builtins (plus two i32 literal args
    // required by SPIR 2.0 provisional specification):
    else if (UnmangledName == "read_pipe_2" ||
             UnmangledName == "write_pipe_2") {
      // with 2 arguments (plus two i32 literals):
      // int read_pipe (read_only pipe gentype p, gentype *ptr)
      // int write_pipe (write_only pipe gentype p, const gentype *ptr)
      addVoidPtrArg(1);
      addUnsignedArg(2);
      addUnsignedArg(3);
    } else if (UnmangledName == "read_pipe_4" ||
               UnmangledName == "write_pipe_4") {
      // with 4 arguments (plus two i32 literals):
      // int read_pipe (read_only pipe gentype p, reserve_id_t reserve_id, uint
      // index, gentype *ptr) int write_pipe (write_only pipe gentype p,
      // reserve_id_t reserve_id, uint index, const gentype *ptr)
      addUnsignedArg(2);
      addVoidPtrArg(3);
      addUnsignedArg(4);
      addUnsignedArg(5);
    } else if (UnmangledName.find("reserve_read_pipe") != std::string::npos ||
               UnmangledName.find("reserve_write_pipe") != std::string::npos) {
      // process [|work_group|sub_group]reserve[read|write]pipe builtins
      addUnsignedArg(1);
      addUnsignedArg(2);
      addUnsignedArg(3);
    } else if (UnmangledName.find("commit_read_pipe") != std::string::npos ||
               UnmangledName.find("commit_write_pipe") != std::string::npos) {
      // process [|work_group|sub_group]commit[read|write]pipe builtins
      addUnsignedArg(2);
      addUnsignedArg(3);
    } else if (UnmangledName == "capture_event_profiling_info") {
      addVoidPtrArg(2);
      setEnumArg(1, SPIR::PRIMITIVE_CLK_PROFILING_INFO);
    } else if (UnmangledName == "enqueue_marker") {
      setArgAttr(2, SPIR::ATTR_CONST);
      addUnsignedArg(1);
    } else if (UnmangledName.find("vload") == 0) {
      addUnsignedArg(0);
      setArgAttr(1, SPIR::ATTR_CONST);
    } else if (UnmangledName.find("vstore") == 0) {
      addUnsignedArg(1);
    } else if (UnmangledName.find("ndrange_") == 0) {
      addUnsignedArg(-1);
      if (UnmangledName[8] == '2' || UnmangledName[8] == '3') {
        setArgAttr(-1, SPIR::ATTR_CONST);
      }
    } else if ((Pos = UnmangledName.find("umax")) != std::string::npos ||
               (Pos = UnmangledName.find("umin")) != std::string::npos) {
      addUnsignedArg(-1);
      UnmangledName.erase(Pos, 1);
    } else if (UnmangledName.find("broadcast") != std::string::npos)
      addUnsignedArg(-1);
    else if (UnmangledName.find(kOCLBuiltinName::SampledReadImage) == 0) {
      UnmangledName.erase(0, strlen(kOCLBuiltinName::Sampled));
      addSamplerArg(1);
    }
  }
  // Auxiliarry information, it is expected what it is relevant at the moment
  // the init method is called.
  Function *F; // SPIRV decorated function
};

CallInst *mutateCallInstOCL(
    Module *M, CallInst *CI,
    std::function<std::string(CallInst *, std::vector<Value *> &)> ArgMutate,
    AttributeList *Attrs) {
  OCLBuiltinFuncMangleInfo BtnInfo(CI->getCalledFunction());
  return mutateCallInst(M, CI, ArgMutate, &BtnInfo, Attrs);
}

Instruction *mutateCallInstOCL(
    Module *M, CallInst *CI,
    std::function<std::string(CallInst *, std::vector<Value *> &, Type *&RetTy)>
        ArgMutate,
    std::function<Instruction *(CallInst *)> RetMutate, AttributeList *Attrs) {
  OCLBuiltinFuncMangleInfo BtnInfo(CI->getCalledFunction());
  return mutateCallInst(M, CI, ArgMutate, RetMutate, &BtnInfo, Attrs);
}

void mutateFunctionOCL(
    Function *F,
    std::function<std::string(CallInst *, std::vector<Value *> &)> ArgMutate,
    AttributeList *Attrs) {
  OCLBuiltinFuncMangleInfo BtnInfo(F);
  return mutateFunction(F, ArgMutate, &BtnInfo, Attrs, false);
}

static std::pair<StringRef, StringRef>
getSrcAndDstElememntTypeName(BitCastInst *BIC) {
  if (!BIC)
    return std::pair<StringRef, StringRef>("", "");

  Type *SrcTy = BIC->getSrcTy();
  Type *DstTy = BIC->getDestTy();
  if (SrcTy->isPointerTy())
    SrcTy = SrcTy->getPointerElementType();
  if (DstTy->isPointerTy())
    DstTy = DstTy->getPointerElementType();
  auto SrcST = dyn_cast<StructType>(SrcTy);
  auto DstST = dyn_cast<StructType>(DstTy);
  if (!DstST || !DstST->hasName() || !SrcST || !SrcST->hasName())
    return std::pair<StringRef, StringRef>("", "");

  return std::make_pair(SrcST->getName(), DstST->getName());
}

bool isSamplerInitializer(Instruction *Inst) {
  BitCastInst *BIC = dyn_cast<BitCastInst>(Inst);
  auto Names = getSrcAndDstElememntTypeName(BIC);
  if (Names.second == getSPIRVTypeName(kSPIRVTypeName::Sampler) &&
      Names.first == getSPIRVTypeName(kSPIRVTypeName::ConstantSampler))
    return true;

  return false;
}

bool isPipeStorageInitializer(Instruction *Inst) {
  BitCastInst *BIC = dyn_cast<BitCastInst>(Inst);
  auto Names = getSrcAndDstElememntTypeName(BIC);
  if (Names.second == getSPIRVTypeName(kSPIRVTypeName::PipeStorage) &&
      Names.first == getSPIRVTypeName(kSPIRVTypeName::ConstantPipeStorage))
    return true;

  return false;
}

bool isSpecialTypeInitializer(Instruction *Inst) {
  return isSamplerInitializer(Inst) || isPipeStorageInitializer(Inst);
}

bool isSamplerTy(Type *Ty) {
  auto PTy = dyn_cast<PointerType>(Ty);
  if (!PTy)
    return false;

  auto STy = dyn_cast<StructType>(PTy->getElementType());
  return STy && STy->hasName() && STy->getName() == kSPR2TypeName::Sampler;
}

bool isPipeBI(const StringRef MangledName) {
  return MangledName == "write_pipe_2" || MangledName == "read_pipe_2" ||
         MangledName == "write_pipe_4" || MangledName == "read_pipe_4" ||
         MangledName == "reserve_write_pipe" ||
         MangledName == "reserve_read_pipe" ||
         MangledName == "commit_write_pipe" ||
         MangledName == "commit_read_pipe" ||
         MangledName == "work_group_reserve_write_pipe" ||
         MangledName == "work_group_reserve_read_pipe" ||
         MangledName == "work_group_commit_write_pipe" ||
         MangledName == "work_group_commit_read_pipe" ||
         MangledName == "get_pipe_num_packets_ro" ||
         MangledName == "get_pipe_max_packets_ro" ||
         MangledName == "get_pipe_num_packets_wo" ||
         MangledName == "get_pipe_max_packets_wo" ||
         MangledName == "sub_group_reserve_write_pipe" ||
         MangledName == "sub_group_reserve_read_pipe" ||
         MangledName == "sub_group_commit_write_pipe" ||
         MangledName == "sub_group_commit_read_pipe";
}

bool isEnqueueKernelBI(const StringRef MangledName) {
  return MangledName == "__enqueue_kernel_basic" ||
         MangledName == "__enqueue_kernel_basic_events" ||
         MangledName == "__enqueue_kernel_varargs" ||
         MangledName == "__enqueue_kernel_events_varargs";
}

bool isKernelQueryBI(const StringRef MangledName) {
  return MangledName == "__get_kernel_work_group_size_impl" ||
         MangledName == "__get_kernel_sub_group_count_for_ndrange_impl" ||
         MangledName == "__get_kernel_max_sub_group_size_for_ndrange_impl" ||
         MangledName == "__get_kernel_preferred_work_group_size_multiple_impl";
}

// Checks if we have the following (most common for fp contranction) pattern
// in LLVM IR:
// %mul = fmul float %a, %b
// %add = fadd float %mul, %c
// This pattern indicates that fp contraction could have been disabled by
// // #pragma OPENCL FP_CONTRACT OFF. Otherwise the current version of clang
// would generate:
// %0 = call float @llvm.fmuladd.f32(float %a, float %b, float %c)
// TODO We need a more reliable mechanism to expres the FP_CONTRACT pragma
// in LLVM IR. Fox example adding the 'contract' attribute to fp operations
// by default (according the OpenCL spec fp contraction is enabled by default).
void checkFpContract(BinaryOperator *B, SPIRVBasicBlock *BB) {
  if (B->getOpcode() != Instruction::FAdd &&
      B->getOpcode() != Instruction::FSub)
    return;
  // Ok, this is fadd or fsub. Now check its operands.
  for (auto *Op : B->operand_values()) {
    if (auto *I = dyn_cast<Instruction>(Op)) {
      if (I->getOpcode() == Instruction::FMul) {
        SPIRVFunction *BF = BB->getParent();
        BF->setUncontractedFMulAddFound();
        break;
      }
    }
  }
}
} // namespace OCLUtil

void llvm::mangleOpenClBuiltin(const std::string &UniqName,
                               ArrayRef<Type *> ArgTypes,
                               std::string &MangledName) {
  OCLUtil::OCLBuiltinFuncMangleInfo BtnInfo(nullptr);
  MangledName = SPIRV::mangleBuiltin(UniqName, ArgTypes, &BtnInfo);
}
