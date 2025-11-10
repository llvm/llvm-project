#ifndef LLVM_TOOLS_LLVM_IR2VEC_EMBEDDINGCOMMON_H
#define LLVM_TOOLS_LLVM_IR2VEC_EMBEDDINGCOMMON_H

#include "llvm/Analysis/IR2Vec.h"
#include "llvm/IR/Function.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/CodeGen/CommandFlags.h"
#include <cxxabi.h>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

namespace llvm {

static inline std::string getDemagledName(const Function *F) {
  auto FunctionName = F->getName().str();
  std::size_t Sz = 17;
  int Status;
  char *const ReadableName =
      __cxxabiv1::__cxa_demangle(FunctionName.c_str(), 0, &Sz, &Status);
  auto DemangledName =
      Status == 0 ? std::string(ReadableName) : std::string(FunctionName);
  free(ReadableName);
  return DemangledName;
}

static inline std::string getActualName(const Function *F) {
  auto FunctionName = F->getName().str();
  auto DemangledName = getDemagledName(F);
  size_t Size = 1;
  char *Buf = static_cast<char *>(std::malloc(Size));
  const char *Mangled = FunctionName.c_str();
  char *BaseName;
  llvm::ItaniumPartialDemangler Mangler;
  if (Mangler.partialDemangle(Mangled)) {
    BaseName = &DemangledName[0];
  } else {
    BaseName = Mangler.getFunctionBaseName(Buf, &Size);
  }
  free(Buf);
  return BaseName ? std::string(BaseName) : std::string();
}
 
using Embedding = ir2vec::Embedding;

/// Embedding generation level
enum EmbeddingLevel {
  InstructionLevel, ///< Generate instruction-level embeddings
  BasicBlockLevel,  ///< Generate basic block-level embeddings
  FunctionLevel     ///< Generate function-level embeddings
};

/// Triplet for vocabulary training (IR2Vec/MIR2Vec)
struct Triplet {
  unsigned Head;
  unsigned Tail;
  unsigned Relation;
};

/// Result of triplet generation
struct TripletResult {
  unsigned MaxRelation;
  std::vector<Triplet> Triplets;
};

/// Entity mappings: entity_id -> entity_name
using EntityMap = std::vector<std::string>;

/// Basic block embeddings: bb_name -> Embedding
using BBVecList = std::vector<std::pair<std::string, Embedding>>;

/// Instruction embeddings: instruction_string -> Embedding
using InstVecList = std::vector<std::pair<std::string, Embedding>>;

/// Function embeddings: demangled_name -> (actual_name, Embedding)
using FuncVecMap = std::unordered_map<std::string, std::pair<std::string, Embedding>>;

} // namespace llvm

#endif