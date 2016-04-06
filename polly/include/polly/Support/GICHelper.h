//===- Support/GICHelper.h -- Helper functions for ISL --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Helper functions for isl objects.
//
//===----------------------------------------------------------------------===//
//
#ifndef POLLY_SUPPORT_GIC_HELPER_H
#define POLLY_SUPPORT_GIC_HELPER_H

#include "llvm/ADT/APInt.h"
#include "llvm/Support/raw_ostream.h"
#include "isl/ctx.h"
#include <string>

struct isl_map;
struct isl_union_map;
struct isl_set;
struct isl_union_set;
struct isl_schedule;
struct isl_multi_aff;
struct isl_pw_multi_aff;
struct isl_union_pw_multi_aff;
struct isl_aff;
struct isl_pw_aff;
struct isl_val;
struct isl_space;

namespace llvm {
class Value;
}

namespace polly {
__isl_give isl_val *isl_valFromAPInt(isl_ctx *Ctx, const llvm::APInt Int,
                                     bool IsSigned);
llvm::APInt APIntFromVal(__isl_take isl_val *Val);

/// @brief Get c++ string from Isl objects.
//@{
std::string stringFromIslObj(__isl_keep isl_map *map);
std::string stringFromIslObj(__isl_keep isl_union_map *umap);
std::string stringFromIslObj(__isl_keep isl_set *set);
std::string stringFromIslObj(__isl_keep isl_union_set *uset);
std::string stringFromIslObj(__isl_keep isl_schedule *schedule);
std::string stringFromIslObj(__isl_keep isl_multi_aff *maff);
std::string stringFromIslObj(__isl_keep isl_pw_multi_aff *pma);
std::string stringFromIslObj(__isl_keep isl_union_pw_multi_aff *upma);
std::string stringFromIslObj(__isl_keep isl_aff *aff);
std::string stringFromIslObj(__isl_keep isl_pw_aff *pwaff);
std::string stringFromIslObj(__isl_keep isl_space *space);
//@}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     __isl_keep isl_union_map *Map) {
  OS << polly::stringFromIslObj(Map);
  return OS;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     __isl_keep isl_map *Map) {
  OS << polly::stringFromIslObj(Map);
  return OS;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     __isl_keep isl_set *Set) {
  OS << polly::stringFromIslObj(Set);
  return OS;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     __isl_keep isl_pw_aff *Map) {
  OS << polly::stringFromIslObj(Map);
  return OS;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     __isl_keep isl_pw_multi_aff *PMA) {
  OS << polly::stringFromIslObj(PMA);
  return OS;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     __isl_keep isl_multi_aff *MA) {
  OS << polly::stringFromIslObj(MA);
  return OS;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     __isl_keep isl_union_pw_multi_aff *UPMA) {
  OS << polly::stringFromIslObj(UPMA);
  return OS;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     __isl_keep isl_schedule *Schedule) {
  OS << polly::stringFromIslObj(Schedule);
  return OS;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     __isl_keep isl_space *Space) {
  OS << polly::stringFromIslObj(Space);
  return OS;
}

/// @brief Return @p Prefix + @p Val->getName() + @p Suffix but Isl compatible.
std::string getIslCompatibleName(const std::string &Prefix,
                                 const llvm::Value *Val,
                                 const std::string &Suffix);

std::string getIslCompatibleName(const std::string &Prefix,
                                 const std::string &Middle,
                                 const std::string &Suffix);

} // end namespace polly

#endif
