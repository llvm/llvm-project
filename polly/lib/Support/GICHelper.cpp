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
#include "llvm/IR/Value.h"
#include "isl/aff.h"
#include "isl/map.h"
#include "isl/schedule.h"
#include "isl/set.h"
#include "isl/space.h"
#include "isl/union_map.h"
#include "isl/union_set.h"
#include "isl/val.h"

#include <climits>

using namespace llvm;

__isl_give isl_val *polly::isl_valFromAPInt(isl_ctx *Ctx, const APInt Int,
                                            bool IsSigned) {
  APInt Abs;
  isl_val *v;

  // As isl is interpreting the input always as unsigned value, we need some
  // additional pre and post processing to import signed values. The approach
  // we take is to first obtain the absolute value of Int and then negate the
  // value after it has been imported to isl.
  //
  // It should be noted that the smallest integer value represented in two's
  // complement with a certain amount of bits does not have a corresponding
  // positive representation in two's complement representation with the same
  // number of bits. E.g. 110 (-2) does not have a corresponding value for (2).
  // To ensure that there is always a corresponding value available we first
  // sign-extend the input by one bit and only then take the absolute value.
  if (IsSigned)
    Abs = Int.sext(Int.getBitWidth() + 1).abs();
  else
    Abs = Int;

  const uint64_t *Data = Abs.getRawData();
  unsigned Words = Abs.getNumWords();

  v = isl_val_int_from_chunks(Ctx, Words, sizeof(uint64_t), Data);

  if (IsSigned && Int.isNegative())
    v = isl_val_neg(v);

  return v;
}

APInt polly::APIntFromVal(__isl_take isl_val *Val) {
  uint64_t *Data;
  int NumChunks;
  const static int ChunkSize = sizeof(uint64_t);

  assert(isl_val_is_int(Val) && "Only integers can be converted to APInt");

  NumChunks = isl_val_n_abs_num_chunks(Val, ChunkSize);
  Data = (uint64_t *)malloc(NumChunks * ChunkSize);
  isl_val_get_abs_num_chunks(Val, ChunkSize, Data);
  int NumBits = CHAR_BIT * ChunkSize * NumChunks;
  APInt A(NumBits, NumChunks, Data);

  // As isl provides only an interface to obtain data that describes the
  // absolute value of an isl_val, A at this point always contains a positive
  // number. In case Val was originally negative, we expand the size of A by
  // one and negate the value (in two's complement representation). As a result,
  // the new value in A corresponds now with Val.
  if (isl_val_is_neg(Val)) {
    A = A.zext(A.getBitWidth() + 1);
    A = -A;
  }

  // isl may represent small numbers with more than the minimal number of bits.
  // We truncate the APInt to the minimal number of bits needed to represent the
  // signed value it contains, to ensure that the bitwidth is always minimal.
  if (A.getMinSignedBits() < A.getBitWidth())
    A = A.trunc(A.getMinSignedBits());

  free(Data);
  isl_val_free(Val);
  return A;
}

template <typename ISLTy, typename ISL_CTX_GETTER, typename ISL_PRINTER>
static inline std::string stringFromIslObjInternal(__isl_keep ISLTy *isl_obj,
                                                   ISL_CTX_GETTER ctx_getter_fn,
                                                   ISL_PRINTER printer_fn) {
  if (!isl_obj)
    return "null";
  isl_ctx *ctx = ctx_getter_fn(isl_obj);
  isl_printer *p = isl_printer_to_str(ctx);
  p = printer_fn(p, isl_obj);
  char *char_str = isl_printer_get_str(p);
  std::string string;
  if (char_str)
    string = char_str;
  else
    string = "null";
  free(char_str);
  isl_printer_free(p);
  return string;
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
  return stringFromIslObjInternal(schedule, isl_schedule_get_ctx,
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

std::string polly::stringFromIslObj(__isl_keep isl_multi_pw_aff *mpa) {
  return stringFromIslObjInternal(mpa, isl_multi_pw_aff_get_ctx,
                                  isl_printer_print_multi_pw_aff);
}

std::string polly::stringFromIslObj(__isl_keep isl_union_pw_multi_aff *upma) {
  return stringFromIslObjInternal(upma, isl_union_pw_multi_aff_get_ctx,
                                  isl_printer_print_union_pw_multi_aff);
}

std::string polly::stringFromIslObj(__isl_keep isl_aff *aff) {
  return stringFromIslObjInternal(aff, isl_aff_get_ctx, isl_printer_print_aff);
}

std::string polly::stringFromIslObj(__isl_keep isl_pw_aff *pwaff) {
  return stringFromIslObjInternal(pwaff, isl_pw_aff_get_ctx,
                                  isl_printer_print_pw_aff);
}

std::string polly::stringFromIslObj(__isl_keep isl_space *space) {
  return stringFromIslObjInternal(space, isl_space_get_ctx,
                                  isl_printer_print_space);
}

static void replace(std::string &str, const std::string &find,
                    const std::string &replace) {
  size_t pos = 0;
  while ((pos = str.find(find, pos)) != std::string::npos) {
    str.replace(pos, find.length(), replace);
    pos += replace.length();
  }
}

static void makeIslCompatible(std::string &str) {
  replace(str, ".", "_");
  replace(str, "\"", "_");
  replace(str, " ", "__");
  replace(str, "=>", "TO");
}

std::string polly::getIslCompatibleName(const std::string &Prefix,
                                        const std::string &Middle,
                                        const std::string &Suffix) {
  std::string S = Prefix + Middle + Suffix;
  makeIslCompatible(S);
  return S;
}

std::string polly::getIslCompatibleName(const std::string &Prefix,
                                        const Value *Val,
                                        const std::string &Suffix) {
  std::string ValStr;
  raw_string_ostream OS(ValStr);
  Val->printAsOperand(OS, false);
  ValStr = OS.str();
  // Remove the leading %
  ValStr.erase(0, 1);
  return getIslCompatibleName(Prefix, ValStr, Suffix);
}

void polly::foreachElt(const isl::map &Map,
                       const std::function<void(isl::basic_map)> &F) {
  isl_map_foreach_basic_map(
      Map.keep(),
      [](__isl_take isl_basic_map *BMap, void *User) -> isl_stat {
        auto &F =
            *static_cast<const std::function<void(isl::basic_map)> *>(User);
        F(give(BMap));
        return isl_stat_ok;
      },
      const_cast<void *>(static_cast<const void *>(&F)));
}

void polly::foreachElt(const isl::set &Set,
                       const std::function<void(isl::basic_set)> &F) {
  isl_set_foreach_basic_set(
      Set.keep(),
      [](__isl_take isl_basic_set *BSet, void *User) -> isl_stat {
        auto &F =
            *static_cast<const std::function<void(isl::basic_set)> *>(User);
        F(give(BSet));
        return isl_stat_ok;
      },
      const_cast<void *>(static_cast<const void *>(&F)));
}

void polly::foreachElt(const isl::union_map &UMap,
                       const std::function<void(isl::map Map)> &F) {
  isl_union_map_foreach_map(
      UMap.keep(),
      [](__isl_take isl_map *Map, void *User) -> isl_stat {
        auto &F = *static_cast<const std::function<void(isl::map)> *>(User);
        F(give(Map));
        return isl_stat_ok;
      },
      const_cast<void *>(static_cast<const void *>(&F)));
}

void polly::foreachElt(const isl::union_set &USet,
                       const std::function<void(isl::set Set)> &F) {
  isl_union_set_foreach_set(
      USet.keep(),
      [](__isl_take isl_set *Set, void *User) -> isl_stat {
        auto &F = *static_cast<const std::function<void(isl::set)> *>(User);
        F(give(Set));
        return isl_stat_ok;
      },
      const_cast<void *>(static_cast<const void *>(&F)));
}

void polly::foreachElt(const isl::union_pw_aff &UPwAff,
                       const std::function<void(isl::pw_aff)> &F) {
  isl_union_pw_aff_foreach_pw_aff(
      UPwAff.keep(),
      [](__isl_take isl_pw_aff *PwAff, void *User) -> isl_stat {
        auto &F = *static_cast<const std::function<void(isl::pw_aff)> *>(User);
        F(give(PwAff));
        return isl_stat_ok;
      },
      const_cast<void *>(static_cast<const void *>(&F)));
}

isl_stat
polly::foreachEltWithBreak(const isl::map &Map,
                           const std::function<isl_stat(isl::basic_map)> &F) {
  return isl_map_foreach_basic_map(
      Map.keep(),
      [](__isl_take isl_basic_map *BMap, void *User) -> isl_stat {
        auto &F =
            *static_cast<const std::function<isl_stat(isl::basic_map)> *>(User);
        return F(give(BMap));
      },
      const_cast<void *>(static_cast<const void *>(&F)));
}

isl_stat
polly::foreachEltWithBreak(const isl::union_map &UMap,
                           const std::function<isl_stat(isl::map Map)> &F) {
  return isl_union_map_foreach_map(
      UMap.keep(),
      [](__isl_take isl_map *Map, void *User) -> isl_stat {
        auto &F =
            *static_cast<const std::function<isl_stat(isl::map Map)> *>(User);
        return F(give(Map));
      },
      const_cast<void *>(static_cast<const void *>(&F)));
}

isl_stat polly::foreachPieceWithBreak(
    const isl::pw_aff &PwAff,
    const std::function<isl_stat(isl::set, isl::aff)> &F) {
  return isl_pw_aff_foreach_piece(
      PwAff.keep(),
      [](__isl_take isl_set *Domain, __isl_take isl_aff *Aff,
         void *User) -> isl_stat {
        auto &F =
            *static_cast<const std::function<isl_stat(isl::set, isl::aff)> *>(
                User);
        return F(give(Domain), give(Aff));
      },
      const_cast<void *>(static_cast<const void *>(&F)));
}
