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

#ifndef LLVM_LIB_TARGET_NVPTX_NVPTXUTILITIES_H
#define LLVM_LIB_TARGET_NVPTX_NVPTXUTILITIES_H

#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Value.h"
#include <cstdarg>
#include <set>
#include <string>
#include <vector>

namespace llvm {

#define NVCL_IMAGE2D_READONLY_FUNCNAME "__is_image2D_readonly"
#define NVCL_IMAGE3D_READONLY_FUNCNAME "__is_image3D_readonly"

void clearAnnotationCache(const llvm::Module *);

bool findOneNVVMAnnotation(const llvm::GlobalValue *, const std::string &,
                           unsigned &);
bool findAllNVVMAnnotation(const llvm::GlobalValue *, const std::string &,
                           std::vector<unsigned> &);

bool isTexture(const llvm::Value &);
bool isSurface(const llvm::Value &);
bool isSampler(const llvm::Value &);
bool isImage(const llvm::Value &);
bool isImageReadOnly(const llvm::Value &);
bool isImageWriteOnly(const llvm::Value &);
bool isImageReadWrite(const llvm::Value &);
bool isManaged(const llvm::Value &);

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
