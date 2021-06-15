//===- GmpConv.cpp - Recreate LLVM IR from the Scop.  ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Functions for converting between gmp objects and llvm::APInt.
//
//===----------------------------------------------------------------------===//

#include "polly/Support/GICHelper.h"
#include "llvm/ADT/APInt.h"
#include "isl/val.h"

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
                                                   ISL_PRINTER printer_fn,
                                                   std::string DefaultValue) {
  if (!isl_obj)
    return DefaultValue;
  isl_ctx *ctx = ctx_getter_fn(isl_obj);
  isl_printer *p = isl_printer_to_str(ctx);
  p = printer_fn(p, isl_obj);
  char *char_str = isl_printer_get_str(p);
  std::string string;
  if (char_str)
    string = char_str;
  else
    string = DefaultValue;
  free(char_str);
  isl_printer_free(p);
  return string;
}

#define ISL_C_OBJECT_TO_STRING(name)                                           \
  std::string polly::stringFromIslObj(__isl_keep isl_##name *Obj,              \
                                      std::string DefaultValue) {              \
    return stringFromIslObjInternal(Obj, isl_##name##_get_ctx,                 \
                                    isl_printer_print_##name, DefaultValue);   \
  }

ISL_C_OBJECT_TO_STRING(aff)
ISL_C_OBJECT_TO_STRING(ast_expr)
ISL_C_OBJECT_TO_STRING(ast_node)
ISL_C_OBJECT_TO_STRING(basic_map)
ISL_C_OBJECT_TO_STRING(basic_set)
ISL_C_OBJECT_TO_STRING(map)
ISL_C_OBJECT_TO_STRING(set)
ISL_C_OBJECT_TO_STRING(id)
ISL_C_OBJECT_TO_STRING(multi_aff)
ISL_C_OBJECT_TO_STRING(multi_pw_aff)
ISL_C_OBJECT_TO_STRING(multi_union_pw_aff)
ISL_C_OBJECT_TO_STRING(point)
ISL_C_OBJECT_TO_STRING(pw_aff)
ISL_C_OBJECT_TO_STRING(pw_multi_aff)
ISL_C_OBJECT_TO_STRING(schedule)
ISL_C_OBJECT_TO_STRING(schedule_node)
ISL_C_OBJECT_TO_STRING(space)
ISL_C_OBJECT_TO_STRING(union_access_info)
ISL_C_OBJECT_TO_STRING(union_flow)
ISL_C_OBJECT_TO_STRING(union_set)
ISL_C_OBJECT_TO_STRING(union_map)
ISL_C_OBJECT_TO_STRING(union_pw_aff)
ISL_C_OBJECT_TO_STRING(union_pw_multi_aff)

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
  replace(str, "+", "_");
}

std::string polly::getIslCompatibleName(const std::string &Prefix,
                                        const std::string &Middle,
                                        const std::string &Suffix) {
  std::string S = Prefix + Middle + Suffix;
  makeIslCompatible(S);
  return S;
}

std::string polly::getIslCompatibleName(const std::string &Prefix,
                                        const std::string &Name, long Number,
                                        const std::string &Suffix,
                                        bool UseInstructionNames) {
  std::string S = Prefix;

  if (UseInstructionNames)
    S += std::string("_") + Name;
  else
    S += std::to_string(Number);

  S += Suffix;

  makeIslCompatible(S);
  return S;
}

std::string polly::getIslCompatibleName(const std::string &Prefix,
                                        const Value *Val, long Number,
                                        const std::string &Suffix,
                                        bool UseInstructionNames) {
  std::string ValStr;

  if (UseInstructionNames && Val->hasName())
    ValStr = std::string("_") + std::string(Val->getName());
  else
    ValStr = std::to_string(Number);

  return getIslCompatibleName(Prefix, ValStr, Suffix);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
/// To call a inline dump() method in a debugger, at it must have been
/// instantiated in at least one translation unit. Because isl's dump() method
/// are meant to be called from a debugger only, but not from code, no such
/// instantiation would exist. We use this method to force an instantiation in
/// this translation unit. Because it has non-static linking, the compiler does
/// not know that it is never called, and therefore must ensure the existence of
/// the dump functions.
void neverCalled() {
  isl::aff().dump();
  isl::aff_list().dump();
  isl::ast_expr().dump();
  isl::ast_expr_list().dump();
  isl::ast_node().dump();
  isl::ast_node_list().dump();
  isl::basic_map().dump();
  isl::basic_map_list().dump();
  isl::basic_set().dump();
  isl::basic_set_list().dump();
  isl::constraint().dump();
  isl::constraint_list().dump();
  isl::id().dump();
  isl::id_list().dump();
  isl::id_to_ast_expr().dump();
  isl::local_space().dump();
  isl::map().dump();
  isl::map_list().dump();
  isl::multi_aff().dump();
  isl::multi_pw_aff().dump();
  isl::multi_union_pw_aff().dump();
  isl::multi_val().dump();
  isl::point().dump();
  isl::pw_aff().dump();
  isl::pw_aff_list().dump();
  isl::pw_multi_aff().dump();
  isl::pw_qpolynomial().dump();
  isl::qpolynomial().dump();
  isl::schedule().dump();
  isl::schedule_constraints().dump();
  isl::schedule_node().dump();
  isl::set().dump();
  isl::set_list().dump();
  isl::space().dump();
  isl::union_map().dump();
  isl::union_map_list().dump();
  isl::union_pw_aff().dump();
  isl::union_pw_aff_list().dump();
  isl::union_pw_multi_aff().dump();
  isl::union_pw_multi_aff_list().dump();
  isl::union_set().dump();
  isl::union_set_list().dump();
  isl::val().dump();
  isl::val_list().dump();
}
#endif
