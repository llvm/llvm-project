//===- NVPTXUtilities.cpp - Utility Functions -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains miscellaneous utility functions
//===----------------------------------------------------------------------===//

#include "NVPTXUtilities.h"
#include "NVPTX.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MutexGuard.h"
#include <algorithm>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using namespace llvm;

typedef std::map<std::string, std::vector<unsigned> > key_val_pair_t;
typedef std::map<const GlobalValue *, key_val_pair_t> global_val_annot_t;
typedef std::map<const Module *, global_val_annot_t> per_module_annot_t;

ManagedStatic<per_module_annot_t> annotationCache;
static sys::Mutex Lock;

void llvm::clearAnnotationCache(const llvm::Module *Mod) {
  MutexGuard Guard(Lock);
  annotationCache->erase(Mod);
}

static void cacheAnnotationFromMD(const MDNode *md, key_val_pair_t &retval) {
  MutexGuard Guard(Lock);
  assert(md && "Invalid mdnode for annotation");
  assert((md->getNumOperands() % 2) == 1 && "Invalid number of operands");
  // start index = 1, to skip the global variable key
  // increment = 2, to skip the value for each property-value pairs
  for (unsigned i = 1, e = md->getNumOperands(); i != e; i += 2) {
    // property
    const MDString *prop = dyn_cast<MDString>(md->getOperand(i));
    assert(prop && "Annotation property not a string");

    // value
    ConstantInt *Val = mdconst::dyn_extract<ConstantInt>(md->getOperand(i + 1));
    assert(Val && "Value operand not a constant int");

    std::string keyname = prop->getString().str();
    if (retval.find(keyname) != retval.end())
      retval[keyname].push_back(Val->getZExtValue());
    else {
      std::vector<unsigned> tmp;
      tmp.push_back(Val->getZExtValue());
      retval[keyname] = tmp;
    }
  }
}

static void cacheAnnotationFromMD(const Module *m, const GlobalValue *gv) {
  MutexGuard Guard(Lock);
  NamedMDNode *NMD = m->getNamedMetadata(llvm::NamedMDForAnnotations);
  if (!NMD)
    return;
  key_val_pair_t tmp;
  for (unsigned i = 0, e = NMD->getNumOperands(); i != e; ++i) {
    const MDNode *elem = NMD->getOperand(i);

    GlobalValue *entity =
        mdconst::dyn_extract_or_null<GlobalValue>(elem->getOperand(0));
    // entity may be null due to DCE
    if (!entity)
      continue;
    if (entity != gv)
      continue;

    // accumulate annotations for entity in tmp
    cacheAnnotationFromMD(elem, tmp);
  }

  if (tmp.empty()) // no annotations for this gv
    return;

  if ((*annotationCache).find(m) != (*annotationCache).end())
    (*annotationCache)[m][gv] = std::move(tmp);
  else {
    global_val_annot_t tmp1;
    tmp1[gv] = std::move(tmp);
    (*annotationCache)[m] = std::move(tmp1);
  }
}

bool llvm::findOneNVVMAnnotation(const GlobalValue *gv, const std::string &prop,
                                 unsigned &retval) {
  MutexGuard Guard(Lock);
  const Module *m = gv->getParent();
  if ((*annotationCache).find(m) == (*annotationCache).end())
    cacheAnnotationFromMD(m, gv);
  else if ((*annotationCache)[m].find(gv) == (*annotationCache)[m].end())
    cacheAnnotationFromMD(m, gv);
  if ((*annotationCache)[m][gv].find(prop) == (*annotationCache)[m][gv].end())
    return false;
  retval = (*annotationCache)[m][gv][prop][0];
  return true;
}

bool llvm::findAllNVVMAnnotation(const GlobalValue *gv, const std::string &prop,
                                 std::vector<unsigned> &retval) {
  MutexGuard Guard(Lock);
  const Module *m = gv->getParent();
  if ((*annotationCache).find(m) == (*annotationCache).end())
    cacheAnnotationFromMD(m, gv);
  else if ((*annotationCache)[m].find(gv) == (*annotationCache)[m].end())
    cacheAnnotationFromMD(m, gv);
  if ((*annotationCache)[m][gv].find(prop) == (*annotationCache)[m][gv].end())
    return false;
  retval = (*annotationCache)[m][gv][prop];
  return true;
}

bool llvm::isTexture(const llvm::Value &val) {
  if (const GlobalValue *gv = dyn_cast<GlobalValue>(&val)) {
    unsigned annot;
    if (llvm::findOneNVVMAnnotation(
            gv, llvm::PropertyAnnotationNames[llvm::PROPERTY_ISTEXTURE],
            annot)) {
      assert((annot == 1) && "Unexpected annotation on a texture symbol");
      return true;
    }
  }
  return false;
}

bool llvm::isSurface(const llvm::Value &val) {
  if (const GlobalValue *gv = dyn_cast<GlobalValue>(&val)) {
    unsigned annot;
    if (llvm::findOneNVVMAnnotation(
            gv, llvm::PropertyAnnotationNames[llvm::PROPERTY_ISSURFACE],
            annot)) {
      assert((annot == 1) && "Unexpected annotation on a surface symbol");
      return true;
    }
  }
  return false;
}

bool llvm::isSampler(const llvm::Value &val) {
  if (const GlobalValue *gv = dyn_cast<GlobalValue>(&val)) {
    unsigned annot;
    if (llvm::findOneNVVMAnnotation(
            gv, llvm::PropertyAnnotationNames[llvm::PROPERTY_ISSAMPLER],
            annot)) {
      assert((annot == 1) && "Unexpected annotation on a sampler symbol");
      return true;
    }
  }
  if (const Argument *arg = dyn_cast<Argument>(&val)) {
    const Function *func = arg->getParent();
    std::vector<unsigned> annot;
    if (llvm::findAllNVVMAnnotation(
            func, llvm::PropertyAnnotationNames[llvm::PROPERTY_ISSAMPLER],
            annot)) {
      if (std::find(annot.begin(), annot.end(), arg->getArgNo()) != annot.end())
        return true;
    }
  }
  return false;
}

bool llvm::isImageReadOnly(const llvm::Value &val) {
  if (const Argument *arg = dyn_cast<Argument>(&val)) {
    const Function *func = arg->getParent();
    std::vector<unsigned> annot;
    if (llvm::findAllNVVMAnnotation(func,
                                    llvm::PropertyAnnotationNames[
                                        llvm::PROPERTY_ISREADONLY_IMAGE_PARAM],
                                    annot)) {
      if (std::find(annot.begin(), annot.end(), arg->getArgNo()) != annot.end())
        return true;
    }
  }
  return false;
}

bool llvm::isImageWriteOnly(const llvm::Value &val) {
  if (const Argument *arg = dyn_cast<Argument>(&val)) {
    const Function *func = arg->getParent();
    std::vector<unsigned> annot;
    if (llvm::findAllNVVMAnnotation(func,
                                    llvm::PropertyAnnotationNames[
                                        llvm::PROPERTY_ISWRITEONLY_IMAGE_PARAM],
                                    annot)) {
      if (std::find(annot.begin(), annot.end(), arg->getArgNo()) != annot.end())
        return true;
    }
  }
  return false;
}

bool llvm::isImageReadWrite(const llvm::Value &val) {
  if (const Argument *arg = dyn_cast<Argument>(&val)) {
    const Function *func = arg->getParent();
    std::vector<unsigned> annot;
    if (llvm::findAllNVVMAnnotation(func,
                                    llvm::PropertyAnnotationNames[
                                        llvm::PROPERTY_ISREADWRITE_IMAGE_PARAM],
                                    annot)) {
      if (std::find(annot.begin(), annot.end(), arg->getArgNo()) != annot.end())
        return true;
    }
  }
  return false;
}

bool llvm::isImage(const llvm::Value &val) {
  return llvm::isImageReadOnly(val) || llvm::isImageWriteOnly(val) ||
         llvm::isImageReadWrite(val);
}

bool llvm::isManaged(const llvm::Value &val) {
  if(const GlobalValue *gv = dyn_cast<GlobalValue>(&val)) {
    unsigned annot;
    if(llvm::findOneNVVMAnnotation(gv,
                          llvm::PropertyAnnotationNames[llvm::PROPERTY_MANAGED],
                                   annot)) {
      assert((annot == 1) && "Unexpected annotation on a managed symbol");
      return true;
    }
  }
  return false;
}

std::string llvm::getTextureName(const llvm::Value &val) {
  assert(val.hasName() && "Found texture variable with no name");
  return val.getName();
}

std::string llvm::getSurfaceName(const llvm::Value &val) {
  assert(val.hasName() && "Found surface variable with no name");
  return val.getName();
}

std::string llvm::getSamplerName(const llvm::Value &val) {
  assert(val.hasName() && "Found sampler variable with no name");
  return val.getName();
}

bool llvm::getMaxNTIDx(const Function &F, unsigned &x) {
  return (llvm::findOneNVVMAnnotation(
      &F, llvm::PropertyAnnotationNames[llvm::PROPERTY_MAXNTID_X], x));
}

bool llvm::getMaxNTIDy(const Function &F, unsigned &y) {
  return (llvm::findOneNVVMAnnotation(
      &F, llvm::PropertyAnnotationNames[llvm::PROPERTY_MAXNTID_Y], y));
}

bool llvm::getMaxNTIDz(const Function &F, unsigned &z) {
  return (llvm::findOneNVVMAnnotation(
      &F, llvm::PropertyAnnotationNames[llvm::PROPERTY_MAXNTID_Z], z));
}

bool llvm::getReqNTIDx(const Function &F, unsigned &x) {
  return (llvm::findOneNVVMAnnotation(
      &F, llvm::PropertyAnnotationNames[llvm::PROPERTY_REQNTID_X], x));
}

bool llvm::getReqNTIDy(const Function &F, unsigned &y) {
  return (llvm::findOneNVVMAnnotation(
      &F, llvm::PropertyAnnotationNames[llvm::PROPERTY_REQNTID_Y], y));
}

bool llvm::getReqNTIDz(const Function &F, unsigned &z) {
  return (llvm::findOneNVVMAnnotation(
      &F, llvm::PropertyAnnotationNames[llvm::PROPERTY_REQNTID_Z], z));
}

bool llvm::getMinCTASm(const Function &F, unsigned &x) {
  return (llvm::findOneNVVMAnnotation(
      &F, llvm::PropertyAnnotationNames[llvm::PROPERTY_MINNCTAPERSM], x));
}

bool llvm::isKernelFunction(const Function &F) {
  unsigned x = 0;
  bool retval = llvm::findOneNVVMAnnotation(
      &F, llvm::PropertyAnnotationNames[llvm::PROPERTY_ISKERNEL_FUNCTION], x);
  if (!retval) {
    // There is no NVVM metadata, check the calling convention
    return F.getCallingConv() == llvm::CallingConv::PTX_Kernel;
  }
  return (x == 1);
}

bool llvm::getAlign(const Function &F, unsigned index, unsigned &align) {
  std::vector<unsigned> Vs;
  bool retval = llvm::findAllNVVMAnnotation(
      &F, llvm::PropertyAnnotationNames[llvm::PROPERTY_ALIGN], Vs);
  if (!retval)
    return false;
  for (int i = 0, e = Vs.size(); i < e; i++) {
    unsigned v = Vs[i];
    if ((v >> 16) == index) {
      align = v & 0xFFFF;
      return true;
    }
  }
  return false;
}

bool llvm::getAlign(const CallInst &I, unsigned index, unsigned &align) {
  if (MDNode *alignNode = I.getMetadata("callalign")) {
    for (int i = 0, n = alignNode->getNumOperands(); i < n; i++) {
      if (const ConstantInt *CI =
              mdconst::dyn_extract<ConstantInt>(alignNode->getOperand(i))) {
        unsigned v = CI->getZExtValue();
        if ((v >> 16) == index) {
          align = v & 0xFFFF;
          return true;
        }
        if ((v >> 16) > index) {
          return false;
        }
      }
    }
  }
  return false;
}

// The following are some useful utilities for debugging

BasicBlock *llvm::getParentBlock(Value *v) {
  if (BasicBlock *B = dyn_cast<BasicBlock>(v))
    return B;

  if (Instruction *I = dyn_cast<Instruction>(v))
    return I->getParent();

  return nullptr;
}

Function *llvm::getParentFunction(Value *v) {
  if (Function *F = dyn_cast<Function>(v))
    return F;

  if (Instruction *I = dyn_cast<Instruction>(v))
    return I->getParent()->getParent();

  if (BasicBlock *B = dyn_cast<BasicBlock>(v))
    return B->getParent();

  return nullptr;
}

// Dump a block by name
void llvm::dumpBlock(Value *v, char *blockName) {
  Function *F = getParentFunction(v);
  if (!F)
    return;

  for (Function::iterator it = F->begin(), ie = F->end(); it != ie; ++it) {
    BasicBlock *B = &*it;
    if (strcmp(B->getName().data(), blockName) == 0) {
      B->dump();
      return;
    }
  }
}

// Find an instruction by name
Instruction *llvm::getInst(Value *base, char *instName) {
  Function *F = getParentFunction(base);
  if (!F)
    return nullptr;

  for (inst_iterator it = inst_begin(F), ie = inst_end(F); it != ie; ++it) {
    Instruction *I = &*it;
    if (strcmp(I->getName().data(), instName) == 0) {
      return I;
    }
  }

  return nullptr;
}

// Dump an instruction by name
void llvm::dumpInst(Value *base, char *instName) {
  Instruction *I = getInst(base, instName);
  if (I)
    I->dump();
}

// Dump an instruction and all dependent instructions
void llvm::dumpInstRec(Value *v, std::set<Instruction *> *visited) {
  if (Instruction *I = dyn_cast<Instruction>(v)) {

    if (visited->find(I) != visited->end())
      return;

    visited->insert(I);

    for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i)
      dumpInstRec(I->getOperand(i), visited);

    I->dump();
  }
}

// Dump an instruction and all dependent instructions
void llvm::dumpInstRec(Value *v) {
  std::set<Instruction *> visited;

  //BasicBlock *B = getParentBlock(v);

  dumpInstRec(v, &visited);
}

// Dump the parent for Instruction, block or function
void llvm::dumpParent(Value *v) {
  if (Instruction *I = dyn_cast<Instruction>(v)) {
    I->getParent()->dump();
    return;
  }

  if (BasicBlock *B = dyn_cast<BasicBlock>(v)) {
    B->getParent()->dump();
    return;
  }

  if (Function *F = dyn_cast<Function>(v)) {
    F->getParent()->dump();
    return;
  }
}
