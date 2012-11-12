//===-- NVPTXUtilities - Utilities -----------------------------*- C++ -*-====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the NVVM specific utility functions.
//
//===----------------------------------------------------------------------===//

#ifndef NVPTXUTILITIES_H
#define NVPTXUTILITIES_H

#include "llvm/Value.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Function.h"
#include "llvm/IntrinsicInst.h"
#include <cstdarg>
#include <set>
#include <string>
#include <vector>

namespace llvm
{

#define NVCL_IMAGE2D_READONLY_FUNCNAME "__is_image2D_readonly"
#define NVCL_IMAGE3D_READONLY_FUNCNAME "__is_image3D_readonly"

bool findOneNVVMAnnotation(const llvm::GlobalValue *, std::string, unsigned &);
bool findAllNVVMAnnotation(const llvm::GlobalValue *, std::string,
                           std::vector<unsigned> &);

bool isTexture(const llvm::Value &);
bool isSurface(const llvm::Value &);
bool isSampler(const llvm::Value &);
bool isImage(const llvm::Value &);
bool isImageReadOnly(const llvm::Value &);
bool isImageWriteOnly(const llvm::Value &);

std::string getTextureName(const llvm::Value &);
std::string getSurfaceName(const llvm::Value &);
std::string getSamplerName(const llvm::Value &);

bool getMaxNTIDx(const llvm::Function &, unsigned &);
bool getMaxNTIDy(const llvm::Function &, unsigned &);
bool getMaxNTIDz(const llvm::Function &, unsigned &);

bool getReqNTIDx(const llvm::Function &, unsigned &);
bool getReqNTIDy(const llvm::Function &, unsigned &);
bool getReqNTIDz(const llvm::Function &, unsigned &);

bool getMinCTASm(const llvm::Function &, unsigned &);
bool isKernelFunction(const llvm::Function &);

bool getAlign(const llvm::Function &, unsigned index, unsigned &);
bool getAlign(const llvm::CallInst &, unsigned index, unsigned &);

bool isBarrierIntrinsic(llvm::Intrinsic::ID);

/// make_vector - Helper function which is useful for building temporary vectors
/// to pass into type construction of CallInst ctors.  This turns a null
/// terminated list of pointers (or other value types) into a real live vector.
///
template<typename T>
inline std::vector<T> make_vector(T A, ...) {
  va_list Args;
  va_start(Args, A);
  std::vector<T> Result;
  Result.push_back(A);
  while (T Val = va_arg(Args, T))
    Result.push_back(Val);
  va_end(Args);
  return Result;
}

bool isMemorySpaceTransferIntrinsic(Intrinsic::ID id);
const Value *skipPointerTransfer(const Value *V, bool ignore_GEP_indices);
const Value *skipPointerTransfer(const Value *V,
                                 std::set<const Value *> &processed);
BasicBlock *getParentBlock(Value *v);
Function *getParentFunction(Value *v);
void dumpBlock(Value *v, char *blockName);
Instruction *getInst(Value *base, char *instName);
void dumpInst(Value *base, char *instName);
void dumpInstRec(Value *v, std::set<Instruction *> *visited);
void dumpInstRec(Value *v);
void dumpParent(Value *v);

}

#endif
