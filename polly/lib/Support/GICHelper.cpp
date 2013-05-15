//===- GmpConv.cpp - Recreate LLVM IR from the Scop.  ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Functions for converting between gmp objects and apint.
//
//===----------------------------------------------------------------------===//
#include "polly/Support/GICHelper.h"
#include "isl/aff.h"
#include "isl/map.h"
#include "isl/schedule.h"
#include "isl/set.h"
#include "isl/union_map.h"
#include "isl/union_set.h"

using namespace llvm;

void polly::MPZ_from_APInt(mpz_t v, const APInt apint, bool is_signed) {
  // There is no sign taken from the data, rop will simply be a positive
  // integer. An application can handle any sign itself, and apply it for
  // instance with mpz_neg.
  APInt abs;
  if (is_signed)
    abs = apint.abs();
  else
    abs = apint;

  const uint64_t *rawdata = abs.getRawData();
  unsigned numWords = abs.getNumWords();

  // TODO: Check if this is true for all platforms.
  mpz_import(v, numWords, 1, sizeof(uint64_t), 0, 0, rawdata);

  if (is_signed && apint.isNegative())
    mpz_neg(v, v);
}

APInt polly::APInt_from_MPZ(const mpz_t mpz) {
  uint64_t *p = NULL;
  size_t sz;

  p = (uint64_t *)mpz_export(p, &sz, 1, sizeof(uint64_t), 0, 0, mpz);

  if (p) {
    APInt A((unsigned) mpz_sizeinbase(mpz, 2), (unsigned) sz, p);
    A = A.zext(A.getBitWidth() + 1);
    free(p);

    if (mpz_sgn(mpz) == -1)
      return -A;
    else
      return A;
  } else {
    uint64_t val = 0;
    return APInt(1, 1, &val);
  }
}

template <typename ISLTy, typename ISL_CTX_GETTER, typename ISL_PRINTER>
static inline std::string stringFromIslObjInternal(__isl_keep ISLTy *isl_obj,
                                                   ISL_CTX_GETTER ctx_getter_fn,
                                                   ISL_PRINTER printer_fn) {
  isl_ctx *ctx = ctx_getter_fn(isl_obj);
  isl_printer *p = isl_printer_to_str(ctx);
  printer_fn(p, isl_obj);
  char *char_str = isl_printer_get_str(p);
  std::string string(char_str);
  free(char_str);
  isl_printer_free(p);
  return string;
}

static inline isl_ctx *schedule_get_ctx(__isl_keep isl_schedule *schedule) {
  return isl_union_map_get_ctx(isl_schedule_get_map(schedule));
}

std::string polly::stringFromIslObj(__isl_keep isl_map *map) {
  return stringFromIslObjInternal(map, isl_map_get_ctx, isl_printer_print_map);
}

std::string polly::stringFromIslObj(__isl_keep isl_set *set) {
  return stringFromIslObjInternal(set, isl_set_get_ctx, isl_printer_print_set);
}

std::string polly::stringFromIslObj(__isl_keep isl_union_map *umap) {
  return stringFromIslObjInternal(umap, isl_union_map_get_ctx,
                                  isl_printer_print_union_map);
}

std::string polly::stringFromIslObj(__isl_keep isl_union_set *uset) {
  return stringFromIslObjInternal(uset, isl_union_set_get_ctx,
                                  isl_printer_print_union_set);
}

std::string polly::stringFromIslObj(__isl_keep isl_schedule *schedule) {
  return stringFromIslObjInternal(schedule, schedule_get_ctx,
                                  isl_printer_print_schedule);
}

std::string polly::stringFromIslObj(__isl_keep isl_multi_aff *maff) {
  return stringFromIslObjInternal(maff, isl_multi_aff_get_ctx,
                                  isl_printer_print_multi_aff);
}

std::string polly::stringFromIslObj(__isl_keep isl_pw_multi_aff *pma) {
  return stringFromIslObjInternal(pma, isl_pw_multi_aff_get_ctx,
                                  isl_printer_print_pw_multi_aff);
}

std::string polly::stringFromIslObj(__isl_keep isl_aff *aff) {
  return stringFromIslObjInternal(aff, isl_aff_get_ctx, isl_printer_print_aff);
}

std::string polly::stringFromIslObj(__isl_keep isl_pw_aff *pwaff) {
  return stringFromIslObjInternal(pwaff, isl_pw_aff_get_ctx,
                                  isl_printer_print_pw_aff);
}
