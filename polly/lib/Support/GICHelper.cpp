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
  if (A.getSignificantBits() < A.getBitWidth())
    A = A.trunc(A.getSignificantBits());

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
#define ISL_DUMP_OBJECT_IMPL(NAME)                                             \
  void polly::dumpIslObj(const isl::NAME &Obj) {                               \
    isl_##NAME##_dump(Obj.get());                                              \
  }                                                                            \
  void polly::dumpIslObj(isl_##NAME *Obj) { isl_##NAME##_dump(Obj); }

ISL_DUMP_OBJECT_IMPL(aff)
ISL_DUMP_OBJECT_IMPL(aff_list)
ISL_DUMP_OBJECT_IMPL(ast_expr)
ISL_DUMP_OBJECT_IMPL(ast_node)
ISL_DUMP_OBJECT_IMPL(ast_node_list)
ISL_DUMP_OBJECT_IMPL(basic_map)
ISL_DUMP_OBJECT_IMPL(basic_map_list)
ISL_DUMP_OBJECT_IMPL(basic_set)
ISL_DUMP_OBJECT_IMPL(basic_set_list)
ISL_DUMP_OBJECT_IMPL(constraint)
ISL_DUMP_OBJECT_IMPL(id)
ISL_DUMP_OBJECT_IMPL(id_list)
ISL_DUMP_OBJECT_IMPL(id_to_ast_expr)
ISL_DUMP_OBJECT_IMPL(local_space)
ISL_DUMP_OBJECT_IMPL(map)
ISL_DUMP_OBJECT_IMPL(map_list)
ISL_DUMP_OBJECT_IMPL(multi_aff)
ISL_DUMP_OBJECT_IMPL(multi_pw_aff)
ISL_DUMP_OBJECT_IMPL(multi_union_pw_aff)
ISL_DUMP_OBJECT_IMPL(multi_val)
ISL_DUMP_OBJECT_IMPL(point)
ISL_DUMP_OBJECT_IMPL(pw_aff)
ISL_DUMP_OBJECT_IMPL(pw_aff_list)
ISL_DUMP_OBJECT_IMPL(pw_multi_aff)
ISL_DUMP_OBJECT_IMPL(schedule)
ISL_DUMP_OBJECT_IMPL(schedule_constraints)
ISL_DUMP_OBJECT_IMPL(schedule_node)
ISL_DUMP_OBJECT_IMPL(set)
ISL_DUMP_OBJECT_IMPL(set_list)
ISL_DUMP_OBJECT_IMPL(space)
ISL_DUMP_OBJECT_IMPL(union_map)
ISL_DUMP_OBJECT_IMPL(union_pw_aff)
ISL_DUMP_OBJECT_IMPL(union_pw_aff_list)
ISL_DUMP_OBJECT_IMPL(union_pw_multi_aff)
ISL_DUMP_OBJECT_IMPL(union_set)
ISL_DUMP_OBJECT_IMPL(union_set_list)
ISL_DUMP_OBJECT_IMPL(val)
ISL_DUMP_OBJECT_IMPL(val_list)

void polly::dumpIslObj(__isl_keep isl_schedule_node *node, raw_ostream &OS) {
  if (!node)
    return;

  isl_ctx *ctx = isl_schedule_node_get_ctx(node);
  isl_printer *p = isl_printer_to_str(ctx);
  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
  p = isl_printer_print_schedule_node(p, node);

  char *char_str = isl_printer_get_str(p);
  OS << char_str;

  free(char_str);
  isl_printer_free(p);
}

void polly::dumpIslObj(const isl::schedule_node &Node, raw_ostream &OS) {
  dumpIslObj(Node.get(), OS);
}

#endif
