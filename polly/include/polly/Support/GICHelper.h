//===- Support/GICHelper.h -- Helper functions for GMP, ISL, and Cloog -----===/
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Helper functions for gmp, isl and Cloog objects.
//
//===----------------------------------------------------------------------===//
//
#ifndef POLLY_SUPPORT_GIC_HELPER_H
#define POLLY_SUPPORT_GIC_HELPER_H

#include "llvm/ADT/APInt.h"
#include <gmp.h>

struct isl_map;
struct isl_union_map;
struct isl_set;
struct isl_union_set;
struct isl_schedule;
struct isl_multi_aff;
struct isl_pw_multi_aff;
struct isl_aff;
struct isl_pw_aff;

namespace polly {

/// @brief Convert APInt to mpz.
///
/// @param v      The mpz_t object your want to hold the result.
/// @param apint  The APInt you want to convert.
void MPZ_from_APInt(mpz_t v, const llvm::APInt apint, bool is_signed = true);

/// @brief Convert mpz to APInt.
///
/// @param mpz    The mpz_t you want to convert.
llvm::APInt APInt_from_MPZ(const mpz_t mpz);

/// @brief Get c++ string from Isl objects.
//@{
std::string stringFromIslObj(/*__isl_keep*/ isl_map *map);
std::string stringFromIslObj(/*__isl_keep*/ isl_union_map *umap);
std::string stringFromIslObj(/*__isl_keep*/ isl_set *set);
std::string stringFromIslObj(/*__isl_keep*/ isl_union_set *uset);
std::string stringFromIslObj(/*__isl_keep*/ isl_schedule *schedule);
std::string stringFromIslObj(/*__isl_keep*/ isl_multi_aff *maff);
std::string stringFromIslObj(/*__isl_keep*/ isl_pw_multi_aff *pma);
std::string stringFromIslObj(/*__isl_keep*/ isl_aff *aff);
std::string stringFromIslObj(/*__isl_keep*/ isl_pw_aff *pwaff);

//@}
} //end namespace polly

#endif
