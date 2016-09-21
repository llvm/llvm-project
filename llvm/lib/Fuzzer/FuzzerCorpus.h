//===- FuzzerCorpus.h - Internal header for the Fuzzer ----------*- C++ -* ===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// fuzzer::InputCorpus
//===----------------------------------------------------------------------===//

#ifndef LLVM_FUZZER_CORPUS
#define LLVM_FUZZER_CORPUS

#include "FuzzerDefs.h"

namespace fuzzer {

struct InputInfo {
  Unit U;  // The actual input data.
};

class InputCorpus {
 public:
  InputCorpus() {
    Corpus.reserve(1 << 14);  // Avoid too many resizes.
  }
  size_t size() const { return Corpus.size(); }
  bool empty() const { return Corpus.empty(); }
  const Unit &operator[] (size_t Idx) const { return Corpus[Idx].U; }
  void Append(const std::vector<Unit> &V) {
    for (auto &U : V)
      push_back(U);
  }
  void push_back(const Unit &U) {
    auto H = Hash(U);
    if (!Hashes.insert(H).second) return;
    InputInfo II;
    II.U = U;
    Corpus.push_back(II);
  }

  typedef const std::vector<InputInfo>::const_iterator ConstIter;
  ConstIter begin() const { return Corpus.begin(); }
  ConstIter end() const { return Corpus.end(); }

  bool HasUnit(const Unit &U) { return Hashes.count(Hash(U)); }

 private:
  std::unordered_set<std::string> Hashes;
  std::vector<InputInfo> Corpus;
};

}  // namespace fuzzer

#endif  // LLVM_FUZZER_CORPUS
