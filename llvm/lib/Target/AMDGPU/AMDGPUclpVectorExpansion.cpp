//===- AMDGPUclpVectorExpansion.cpp ------------===//
//
// Copyright(c) 2016 Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief vector builtin expansion engine for clp.
//
//===----------------------------------------------------------------------===//
#include "AMDGPU.h"
#include "llvm/Config/config.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <sstream>

#define DEBUG_TYPE "amdclpvectorexpansion"

using namespace llvm;

// Copied from amd_ocl_builtindef.h
// descriptor for a parameter type or return type
enum an_typedes_t {
  tdInvalid = 0,

  // beginning of concret type
  tdInt,

  // beginning of abstract lead1 type
  tdAnyIntFloat,
  tdAnyFloat,     // half + float + double
  tdAnySingle,    // float
  tdAnyFloat_PLG, // pointer to half + float + double + private|local|global
  tdAnyInt,
  tdAnyIntk8_32,  //[u]char => [u]int
  tdAnyIntk32_32, //[u]int[n]
  tdAnySint,
  tdAnyUintk16_32_64,

  // beginning of abstract following type
  tdFollow,                   // follow lead1
  tdFollowKu,                 // the unsigned of the int type
  tdFollowKn,                 // char=>short, uchar=>ushort, etc
  tdFollowVsInt,              // follow the vector size => intn
  tdFollowVsShortIntLongn,    // half=>int halfn=>shortn, float/floatn => intn,
                              // double=>int, doublen=>longn
  tdFollowVsHalfFloatDoublen, // follow the vector size => half/float/doublen
  tdFollowPele, // follow the element type of the lead1 pointer type
};

#define MAX_NUM_PARAMS (3)

struct a_builtinfunc {
  const char *Name; // the name of the opencl builtin function
  char LeaderId;    // this is the id for the leading parameter, range from
                    // 1..LastParam,
  //-1 indicate nonoverloading
  char TypeDes[MAX_NUM_PARAMS +
               1]; // type descriptor for return type, and each parameter
};                 // a_builtinfunc;

#define NO_LEADS ((char)-1)
// 1 parameter
#define PTR_TYPE_1(r, p1)                                                      \
  { r, p1, tdInvalid, tdInvalid }
// 2 parameter
#define PTR_TYPE_2(r, p1, p2)                                                  \
  { r, p1, p2, tdInvalid }
// 3 parameter
#define PTR_TYPE_3(r, p1, p2, p3)                                              \
  { r, p1, p2, p3 }

#define END_FUNC_GROUPS                                                        \
  {                                                                            \
    nullptr, NO_LEADS, { tdInvalid, tdInvalid, tdInvalid, tdInvalid }          \
  }

// section 6.12.2 math functions
// the version with double type is controled via whether the type is available
// (NULL not available)
//
static const a_builtinfunc MathFunc[] = {
    {(const char *)"acos", 1, PTR_TYPE_1(tdFollow, tdAnyFloat)},
    {(const char *)"acosh", 1, PTR_TYPE_1(tdFollow, tdAnyFloat)},
    {(const char *)"acospi", 1, PTR_TYPE_1(tdFollow, tdAnyFloat)},
    {(const char *)"asin", 1, PTR_TYPE_1(tdFollow, tdAnyFloat)},
    {(const char *)"asinh", 1, PTR_TYPE_1(tdFollow, tdAnyFloat)},
    {(const char *)"asinpi", 1, PTR_TYPE_1(tdFollow, tdAnyFloat)},
    {(const char *)"atan", 1, PTR_TYPE_1(tdFollow, tdAnyFloat)},
    {(const char *)"atan2", 1, PTR_TYPE_2(tdFollow, tdAnyFloat, tdFollow)},
    {(const char *)"atanh", 1, PTR_TYPE_1(tdFollow, tdAnyFloat)},
    {(const char *)"atanpi", 1, PTR_TYPE_1(tdFollow, tdAnyFloat)},
    {(const char *)"atan2pi", 1, PTR_TYPE_2(tdFollow, tdAnyFloat, tdFollow)},

    {(const char *)"cbrt", 1, PTR_TYPE_1(tdFollow, tdAnyFloat)},
    {(const char *)"ceil", 1, PTR_TYPE_1(tdFollow, tdAnyFloat)},
    {(const char *)"copysign", 1, PTR_TYPE_2(tdFollow, tdAnyFloat, tdFollow)},
    {(const char *)"cos", 1, PTR_TYPE_1(tdFollow, tdAnyFloat)},
    {(const char *)"cosh", 1, PTR_TYPE_1(tdFollow, tdAnyFloat)},
    {(const char *)"cospi", 1, PTR_TYPE_1(tdFollow, tdAnyFloat)},
    {(const char *)"erfc", 1, PTR_TYPE_1(tdFollow, tdAnyFloat)},
    {(const char *)"erf", 1, PTR_TYPE_1(tdFollow, tdAnyFloat)},

    {(const char *)"exp", 1, PTR_TYPE_1(tdFollow, tdAnyFloat)},
    {(const char *)"exp2", 1, PTR_TYPE_1(tdFollow, tdAnyFloat)},
    {(const char *)"exp10", 1, PTR_TYPE_1(tdFollow, tdAnyFloat)},
    {(const char *)"expm1", 1, PTR_TYPE_1(tdFollow, tdAnyFloat)},
    {(const char *)"fabs", 1, PTR_TYPE_1(tdFollow, tdAnyFloat)},
    {(const char *)"fdim", 1, PTR_TYPE_2(tdFollow, tdAnyFloat, tdFollow)},
    {(const char *)"floor", 1, PTR_TYPE_1(tdFollow, tdAnyFloat)},

    {(const char *)"fma", 1,
     PTR_TYPE_3(tdFollow, tdAnyFloat, tdFollow, tdFollow)},
    {(const char *)"__builtin_fma", 1,
     PTR_TYPE_3(tdFollow, tdAnyFloat, tdFollow, tdFollow)},
    // in the spec, fmax/fmin is lised in three entry
    // we cover the last two entry by allowing scalar=>vector promotion
    {(const char *)"fmax", 1, PTR_TYPE_2(tdFollow, tdAnyFloat, tdFollow)},
    {(const char *)"fmin", 1, PTR_TYPE_2(tdFollow, tdAnyFloat, tdFollow)},
    {(const char *)"fmod", 1, PTR_TYPE_2(tdFollow, tdAnyFloat, tdFollow)},

    {(const char *)"fract", 2,
     PTR_TYPE_2(tdFollowPele, tdFollowPele, tdAnyFloat_PLG)},
    // fexp is described in group2
    {(const char *)"hypot", 1, PTR_TYPE_2(tdFollow, tdAnyFloat, tdFollow)},
    {(const char *)"ilogb", 1, PTR_TYPE_1(tdFollowVsInt, tdAnyFloat)},
    {(const char *)"ldexp", 1, PTR_TYPE_2(tdFollow, tdAnyFloat, tdFollowVsInt)},
    {(const char *)"lgamma", 1, PTR_TYPE_1(tdFollow, tdAnyFloat)},
    // lgamma_r is described in group2
    {(const char *)"log", 1, PTR_TYPE_1(tdFollow, tdAnyFloat)},
    {(const char *)"log2", 1, PTR_TYPE_1(tdFollow, tdAnyFloat)},
    {(const char *)"log10", 1, PTR_TYPE_1(tdFollow, tdAnyFloat)},
    {(const char *)"log1p", 1, PTR_TYPE_1(tdFollow, tdAnyFloat)},
    {(const char *)"logb", 1, PTR_TYPE_1(tdFollow, tdAnyFloat)},
    {(const char *)"mad", 1,
     PTR_TYPE_3(tdFollow, tdAnyFloat, tdFollow, tdFollow)},
    {(const char *)"maxmag", 1, PTR_TYPE_2(tdFollow, tdAnyFloat, tdFollow)},
    {(const char *)"minmag", 1, PTR_TYPE_2(tdFollow, tdAnyFloat, tdFollow)},
    {(const char *)"modf", 2,
     PTR_TYPE_2(tdFollowPele, tdFollowPele, tdAnyFloat_PLG)},
    {(const char *)"nan", 1,
     PTR_TYPE_1(tdFollowVsHalfFloatDoublen, tdAnyUintk16_32_64)},
    {(const char *)"nextafter", 1, PTR_TYPE_2(tdFollow, tdAnyFloat, tdFollow)},
    {(const char *)"pow", 1, PTR_TYPE_2(tdFollow, tdAnyFloat, tdFollow)},
    {(const char *)"pown", 1, PTR_TYPE_2(tdFollow, tdAnyFloat, tdFollowVsInt)},
    {(const char *)"powr", 1, PTR_TYPE_2(tdFollow, tdAnyFloat, tdFollow)},
    {(const char *)"remainder", 1, PTR_TYPE_2(tdFollow, tdAnyFloat, tdFollow)},
    // remquo is described in group2
    {(const char *)"rint", 1, PTR_TYPE_1(tdFollow, tdAnyFloat)},
    {(const char *)"rootn", 1, PTR_TYPE_2(tdFollow, tdAnyFloat, tdFollowVsInt)},
    {(const char *)"round", 1, PTR_TYPE_1(tdFollow, tdAnyFloat)},
    {(const char *)"rsqrt", 1, PTR_TYPE_1(tdFollow, tdAnyFloat)},
    {(const char *)"sin", 1, PTR_TYPE_1(tdFollow, tdAnyFloat)},
    {(const char *)"sincos", 2,
     PTR_TYPE_2(tdFollowPele, tdFollowPele, tdAnyFloat_PLG)},
    {(const char *)"sinh", 1, PTR_TYPE_1(tdFollow, tdAnyFloat)},
    {(const char *)"sinpi", 1, PTR_TYPE_1(tdFollow, tdAnyFloat)},
    {(const char *)"sqrt", 1, PTR_TYPE_1(tdFollow, tdAnyFloat)},
    {(const char *)"tan", 1, PTR_TYPE_1(tdFollow, tdAnyFloat)},
    {(const char *)"tanh", 1, PTR_TYPE_1(tdFollow, tdAnyFloat)},
    {(const char *)"tanpi", 1, PTR_TYPE_1(tdFollow, tdAnyFloat)},
    {(const char *)"tgamma", 1, PTR_TYPE_1(tdFollow, tdAnyFloat)},
    {(const char *)"trunc", 1, PTR_TYPE_1(tdFollow, tdAnyFloat)},

    {(const char *)"half_cos", 1, PTR_TYPE_1(tdFollow, tdAnySingle)},
    {(const char *)"half_divide", 1,
     PTR_TYPE_2(tdFollow, tdAnySingle, tdFollow)},
    {(const char *)"half_exp", 1, PTR_TYPE_1(tdFollow, tdAnySingle)},
    {(const char *)"half_exp2", 1, PTR_TYPE_1(tdFollow, tdAnySingle)},
    {(const char *)"half_exp10", 1, PTR_TYPE_1(tdFollow, tdAnySingle)},
    {(const char *)"half_log", 1, PTR_TYPE_1(tdFollow, tdAnySingle)},
    {(const char *)"half_log2", 1, PTR_TYPE_1(tdFollow, tdAnySingle)},
    {(const char *)"half_log10", 1, PTR_TYPE_1(tdFollow, tdAnySingle)},
    {(const char *)"half_powr", 1, PTR_TYPE_2(tdFollow, tdAnySingle, tdFollow)},
    {(const char *)"half_recip", 1, PTR_TYPE_1(tdFollow, tdAnySingle)},
    {(const char *)"half_rsqrt", 1, PTR_TYPE_1(tdFollow, tdAnySingle)},
    {(const char *)"half_sin", 1, PTR_TYPE_1(tdFollow, tdAnySingle)},
    {(const char *)"half_sqrt", 1, PTR_TYPE_1(tdFollow, tdAnySingle)},
    {(const char *)"half_tan", 1, PTR_TYPE_1(tdFollow, tdAnySingle)},
    {(const char *)"native_cos", 1, PTR_TYPE_1(tdFollow, tdAnySingle)},
    {(const char *)"native_divide", 1,
     PTR_TYPE_2(tdFollow, tdAnyFloat, tdFollow)},
    {(const char *)"native_exp", 1, PTR_TYPE_1(tdFollow, tdAnySingle)},
    {(const char *)"native_exp2", 1, PTR_TYPE_1(tdFollow, tdAnySingle)},
    {(const char *)"native_exp10", 1, PTR_TYPE_1(tdFollow, tdAnySingle)},
    {(const char *)"native_log", 1, PTR_TYPE_1(tdFollow, tdAnySingle)},
    {(const char *)"native_log2", 1, PTR_TYPE_1(tdFollow, tdAnySingle)},
    {(const char *)"native_log10", 1, PTR_TYPE_1(tdFollow, tdAnySingle)},
    {(const char *)"native_powr", 1,
     PTR_TYPE_2(tdFollow, tdAnySingle, tdFollow)},
    {(const char *)"native_recip", 1, PTR_TYPE_1(tdFollow, tdAnyFloat)},
    {(const char *)"native_rsqrt", 1, PTR_TYPE_1(tdFollow, tdAnyFloat)},
    {(const char *)"native_sin", 1, PTR_TYPE_1(tdFollow, tdAnySingle)},
    {(const char *)"native_sqrt", 1, PTR_TYPE_1(tdFollow, tdAnyFloat)},
    {(const char *)"native_tan", 1, PTR_TYPE_1(tdFollow, tdAnySingle)},
    END_FUNC_GROUPS};

// section 6.12.3 integer functions
static const a_builtinfunc IntegerFunc[] = {
    {(const char *)"abs", 1, PTR_TYPE_1(tdFollowKu, tdAnyInt)},
    {(const char *)"abs_diff", 1, PTR_TYPE_2(tdFollowKu, tdAnyInt, tdFollow)},
    {(const char *)"add_sat", 1, PTR_TYPE_2(tdFollow, tdAnyInt, tdFollow)},
    {(const char *)"hadd", 1, PTR_TYPE_2(tdFollow, tdAnyInt, tdFollow)},
    {(const char *)"rhadd", 1, PTR_TYPE_2(tdFollow, tdAnyInt, tdFollow)},
    {(const char *)"clamp", 1,
     PTR_TYPE_3(tdFollow, tdAnyIntFloat, tdFollow, tdFollow)},
    {(const char *)"clz", 1, PTR_TYPE_1(tdFollow, tdAnyInt)},
    {(const char *)"ctz", 1, PTR_TYPE_1(tdFollow, tdAnyInt)},
    {(const char *)"mad_hi", 1,
     PTR_TYPE_3(tdFollow, tdAnyInt, tdFollow, tdFollow)},
    {(const char *)"mad_sat", 1,
     PTR_TYPE_3(tdFollow, tdAnyInt, tdFollow, tdFollow)},
    {(const char *)"max", 1, PTR_TYPE_2(tdFollow, tdAnyIntFloat, tdFollow)},
    {(const char *)"min", 1, PTR_TYPE_2(tdFollow, tdAnyIntFloat, tdFollow)},
    {(const char *)"mul_hi", 1, PTR_TYPE_2(tdFollow, tdAnyInt, tdFollow)},
    {(const char *)"rotate", 1, PTR_TYPE_2(tdFollow, tdAnyInt, tdFollow)},
    {(const char *)"sub_sat", 1, PTR_TYPE_2(tdFollow, tdAnyInt, tdFollow)},
    {(const char *)"upsample", 1,
     PTR_TYPE_2(tdFollowKn, tdAnyIntk8_32, tdFollowKu)},
    {(const char *)"popcount", 1, PTR_TYPE_1(tdFollow, tdAnyInt)},
    {(const char *)"mad24", 1,
     PTR_TYPE_3(tdFollow, tdAnyIntk32_32, tdFollow, tdFollow)},
    {(const char *)"mul24", 1, PTR_TYPE_2(tdFollow, tdAnyIntk32_32, tdFollow)},
    END_FUNC_GROUPS};

// section 6.12.4 common functions
static const a_builtinfunc CommonFunc[] = {
    // clamp is described in integer function
    {(const char *)"degrees", 1, PTR_TYPE_1(tdFollow, tdAnyFloat)},
    // max, min is described in integer function
    {(const char *)"mix", 1,
     PTR_TYPE_3(tdFollow, tdAnyFloat, tdFollow, tdFollow)},
    {(const char *)"radians", 1, PTR_TYPE_1(tdFollow, tdAnyFloat)},
    {(const char *)"step", 1, PTR_TYPE_2(tdFollow, tdAnyFloat, tdFollow)},
    {(const char *)"smoothstep", 1,
     PTR_TYPE_3(tdFollow, tdAnyFloat, tdFollow, tdFollow)},
    {(const char *)"sign", 1, PTR_TYPE_1(tdFollow, tdAnyFloat)},
    END_FUNC_GROUPS};

// section 6.12.6 relational functions
// 1.2 spec has a problem in isequal(double, double) return type
static const a_builtinfunc relationalFunc[] = {
    {(const char *)"isequal", 1,
     PTR_TYPE_2(tdFollowVsShortIntLongn, tdAnyFloat, tdFollow)},
    {(const char *)"isnotequal", 1,
     PTR_TYPE_2(tdFollowVsShortIntLongn, tdAnyFloat, tdFollow)},
    {(const char *)"isgreater", 1,
     PTR_TYPE_2(tdFollowVsShortIntLongn, tdAnyFloat, tdFollow)},
    {(const char *)"isgreaterequal", 1,
     PTR_TYPE_2(tdFollowVsShortIntLongn, tdAnyFloat, tdFollow)},
    {(const char *)"isless", 1,
     PTR_TYPE_2(tdFollowVsShortIntLongn, tdAnyFloat, tdFollow)},
    {(const char *)"islessequal", 1,
     PTR_TYPE_2(tdFollowVsShortIntLongn, tdAnyFloat, tdFollow)},
    {(const char *)"islessgreater", 1,
     PTR_TYPE_2(tdFollowVsShortIntLongn, tdAnyFloat, tdFollow)},
    {(const char *)"isfinite", 1,
     PTR_TYPE_1(tdFollowVsShortIntLongn, tdAnyFloat)},
    {(const char *)"isinf", 1, PTR_TYPE_1(tdFollowVsShortIntLongn, tdAnyFloat)},
    {(const char *)"isnan", 1, PTR_TYPE_1(tdFollowVsShortIntLongn, tdAnyFloat)},
    {(const char *)"isnormal", 1,
     PTR_TYPE_1(tdFollowVsShortIntLongn, tdAnyFloat)},
    {(const char *)"isordered", 1,
     PTR_TYPE_2(tdFollowVsShortIntLongn, tdAnyFloat, tdFollow)},
    {(const char *)"isunordered", 1,
     PTR_TYPE_2(tdFollowVsShortIntLongn, tdAnyFloat, tdFollow)},
    {(const char *)"signbit", 1,
     PTR_TYPE_1(tdFollowVsShortIntLongn, tdAnyFloat)},
    {(const char *)"any", 1, PTR_TYPE_1(tdInt, tdAnySint)},
    {(const char *)"all", 1, PTR_TYPE_1(tdInt, tdAnySint)},
    {(const char *)"bitselect", 1,
     PTR_TYPE_3(tdFollow, tdAnyIntFloat, tdFollow, tdFollow)},
    // select is described in group2
    END_FUNC_GROUPS};

static const a_builtinfunc pragmaEnableFunc[] = {
    {(const char *)"popcnt", 1, PTR_TYPE_1(tdFollow, tdAnyInt)},
    END_FUNC_GROUPS};

//===----------------------------------------------------------------------===//
// info that describes what to expand
// it is currently a list of builtin names
// as we weak expand all vector size for all targets

// use the order of commonVectorExpansions.cl
//
static const char *NeedExpansionBuiltins[] = {
    "acos", "acosh", "acospi", "asin", "asinh", "asinpi", "atan", "atanpi",
    "atanh", "cos", "cosh", "cospi", "sin", "sinh", "sinpi", "tan", "tanh",
    "tanpi", "cbrt", "ceil",

    "erf", "erfc", "exp", "exp2", "exp10", "expm1", "fabs", "floor", "lgamma",
    "log", "log2", "log10", "log1p", "logb", "rint", "round", "rsqrt", "sqrt",
    "trunc", "tgamma",

    "half_cos", "half_exp", "half_exp2", "half_exp10", "half_log", "half_log2",
    "half_log10", "half_recip", "half_rsqrt", "half_sin", "half_sqrt",
    "half_tan", "native_cos", "native_exp", "native_exp2", "native_exp10",
    "native_log", "native_log2", "native_log10", "native_recip", "native_rsqrt",
    "native_sin", "native_sqrt", "native_tan",

    "half_divide", "half_powr", "native_divide", "native_powr",

    "sign", "degrees", "radians",

    "nan", "clz", "ctz", "popcnt", "popcount", "abs",

    "ilogb", "atan2", "atan2pi", "copysign", "fdim", "fmax", "fmin", "fmod",
    "hypot", "nextafter", "pow", "powr", "remainder",

    "mul24",

    "fract", "modf", "sincos",
    // todo: either reimplement it using lead1 and remove lead2 or handle lead2
    // ...
    //      "frexp",
    // todo: "lgamma_r",
    "upsample",

    "rootn", "ldexp", "pown", "max", "min", "add_sat", "hadd", "rhadd",
    "mul_hi", "sub_sat", "rotate", "abs_diff",

    "fma", "mad",

    "mad_hi", "mad_sat", "mad24",
    // todo: "remquo",
    "clamp", "mix", "step", "smoothstep",

    "maxmag", "minmag",

    nullptr};

//===----------------------------------------------------------------------===//

const an_typedes_t HandledLead1TypeDes[] = {
    // todo: fix the type name an=>a
    tdAnyIntFloat, tdAnyFloat,     tdAnySingle,        tdAnyInt,
    tdAnyIntk8_32, tdAnyIntk32_32, tdAnyUintk16_32_64, tdAnyFloat_PLG,
    tdInvalid // 0
};

// need to test popcnt
const an_typedes_t HandledOtherTypeDes[] = {
    tdFollow,
    tdFollowKu,
    tdFollowKn,                 // usample
    tdFollowVsInt,              // ilogb
    tdFollowVsHalfFloatDoublen, // nan
    tdFollowPele,
    tdInvalid // 0
};

static bool leadTypeDesIsPointer(an_typedes_t TypeDes) {
  return TypeDes == tdAnyFloat_PLG;
} // leadTypeDesIsPointer

#define InvalidASChar '\0'

static Type *getNextType(Type *CurTy, int CurVecSize, int NextVecSize,
                         bool IsPointer, char NextAddrSpace) {
  Type *CurVecTy = isa<PointerType>(CurTy)
                       ? cast<PointerType>(CurTy)->getElementType()
                       : CurTy;

  assert(isa<VectorType>(CurVecTy) && "problem with builtin description");
  assert(static_cast<int>(cast<VectorType>(CurVecTy)->getNumElements()) ==
             CurVecSize &&
         "problem with builtin description");

  Type *CurEleTy = cast<VectorType>(CurVecTy)->getElementType();

  Type *NextTy;
  if (NextVecSize == 1)
    NextTy = CurEleTy;
  else
    NextTy = VectorType::get(CurEleTy, NextVecSize);
  if (IsPointer) {
    assert('0' <= NextAddrSpace && NextAddrSpace <= '4' &&
           "incorrect addrspace");
    NextTy = PointerType::get(
        NextTy, NextAddrSpace == InvalidASChar ? 0 : (NextAddrSpace - '0'));
  }

  return NextTy;
} // getNextType

static Type *getNextLeadType(Type *CurLeadTy, an_typedes_t TypeDes,
                             int CurVecSize, int NextVecSize,
                             char NextAddrSpace) {
  assert(((leadTypeDesIsPointer(TypeDes) && isa<PointerType>(CurLeadTy)) ||
          isa<PointerType>(CurLeadTy) == false) &&
         "problem with builtin description");

  return getNextType(CurLeadTy, CurVecSize, NextVecSize,
                     leadTypeDesIsPointer(TypeDes), NextAddrSpace);
} // getNextLeadType

static Type *getNextOtherType(Type *CurTy, int CurVecSize, int NextVecSize) {
  return getNextType(CurTy, CurVecSize, NextVecSize, false, InvalidASChar);
} // getNextOtherType

static int getAndPassVecSize(StringRef &Str) {
  int Res = 0;
  while (Str[0] >= '1' && Str[0] <= '9') {
    Res = Res * 10 + Str[0] - '0';
    Str = Str.substr(1);
  }
  if (Res == 0)
    Res = 1;

  return Res;
} // getAndPassVecSize

static bool canHandled(an_typedes_t TypeDes, const an_typedes_t *Array) {
  int Idx = 0;
  while (Array[Idx] != 0 && Array[Idx] != TypeDes)
    ++Idx;
  return (Array[Idx] != tdInvalid);
} // canHandled

// data structure to collect use of builtin functions
struct a_funcuse_t {
  Function *LlvmFunc;
  const a_builtinfunc *Builtin;
  int VecSize; // number of vector elements
  a_funcuse_t() : LlvmFunc(0), Builtin(0), VecSize(0) {}
  a_funcuse_t(Function *f, int v, const a_builtinfunc *b)
      : LlvmFunc(f), Builtin(b), VecSize(v) {}
};

namespace {
class AMDGPUclpVectorExpansion : public ModulePass {
  // data structure to provide builtin names => builtin info mapping
  typedef StringMap<const a_builtinfunc *> an_expansionInfo_t;
  an_expansionInfo_t ExpansionInfo;
  std::unique_ptr<Module> TempModule;

  char getAddrSpaceCode(StringRef);
  std::string getNextFunctionName(StringRef, an_typedes_t, int, char);
  void checkAndAddToExpansion(Function *);
  std::string replaceSubstituteTypes(StringRef &, StringRef);
  StringRef passPointerAddrSpace(StringRef &);

public:
  AMDGPUclpVectorExpansion();
  bool runOnModule(Module &) override;

protected:
  void addBuiltinInfo(const a_builtinfunc *);
  void addNeedExpansion(const char **);

#if !defined(NDEBUG)
  void checkExpansionInfo();
  bool canHandlePattern(const a_builtinfunc *);
#endif

  void addFuncuseInfo(Function *, int vecSize, const a_builtinfunc *);
  const a_builtinfunc *getBuiltinInfo(StringRef &name);

  void checkAndExpand(a_funcuse_t *);
  Function *checkAndExpand(Function *, int, const a_builtinfunc *, int &,
                           char nextAddrSpace);

  Value *loadVectorSlice(int, int, Value *, BasicBlock *);
  Value *insertVectorSlice(int, int, Value *, Value *, BasicBlock *);

  Function *getNextFunction(Function *curFunc, int curVecSize,
                            const a_builtinfunc *builtin, int &nextVecSize,
                            char nextAddrSpace);
  Function *adjustFunctionImpl(Function *curFunc, int curVecSize,
                               const a_builtinfunc *builtin, int nextVecSize,
                               char);

public:
  static char ID;
}; // class AMDGPUclpVectorExpansion
}

char AMDGPUclpVectorExpansion::ID = 0;

INITIALIZE_PASS(AMDGPUclpVectorExpansion, "amdgpu-clp-vector-expansion",
                "Vector builtins expansion for clp", false, false)

char &llvm::AMDGPUclpVectorExpansionID = AMDGPUclpVectorExpansion::ID;

namespace llvm {
ModulePass *createAMDGPUclpVectorExpansionPass() {
  return new AMDGPUclpVectorExpansion();
}
}

static const char *OPENCL_VARG_BUILTIN_SPIR_PREFIX = "_Z";
static const int OPENCL_VARG_BUILTIN_PREFIX_LEN = 2;
static const int OPENCL_VARG_BUILTIN_ADDRSPACE_LEN = 6; // PU3ASN
static const char *OPENCL_VARG_BUILTIN_SPIR_VECTYPE = "Dv";
static const int OPENCL_VARG_BUILTIN_SPIR_VECTYPE_LEN = 2;
static inline int OCLTypeLength(char t) { return t == 'D' ? 2 : 1; }

StringRef AMDGPUclpVectorExpansion::passPointerAddrSpace(StringRef &Suffix) {
  assert(Suffix[0] == 'P' && "internal problem");
  StringRef AddrSpace;
  if (Suffix.rfind("PU") !=
      StringRef::npos) { // pointer to non-private address space
    AddrSpace = Suffix.slice(0, OPENCL_VARG_BUILTIN_ADDRSPACE_LEN);
    Suffix = Suffix.substr(OPENCL_VARG_BUILTIN_ADDRSPACE_LEN);
  } else { // pointer to private address space
    AddrSpace = Suffix.slice(0, 0);
    Suffix = Suffix.substr(1);
  }
  return AddrSpace;
} // passPointerAddrSpace

/// replace substitution mangled types
///
std::string
AMDGPUclpVectorExpansion::replaceSubstituteTypes(StringRef &Suffix,
                                                 StringRef ArgType) {
  std::ostringstream Oss;
  while (!Suffix.empty() && Suffix[0] == 'S') {
    Oss << ArgType.str();
    Suffix = Suffix.substr(Suffix.find_first_of('_') + 1);
  }
  return Oss.str();
} // replaceSubstituteTypes

/// extract the address space code from the current function name
///
char AMDGPUclpVectorExpansion::getAddrSpaceCode(StringRef CurFuncName) {
  char AddrSpaceCode = InvalidASChar;
  assert(CurFuncName.startswith(OPENCL_VARG_BUILTIN_SPIR_PREFIX));

  // pointer to non-private address space
  if (CurFuncName.rfind("PU") != StringRef::npos) {
    //<address-space-number> consists of 'AS' followed by the address space
    //<number>
    size_t SeparatorPos = CurFuncName.rfind("AS");
    assert(SeparatorPos != StringRef::npos);
    AddrSpaceCode = CurFuncName[SeparatorPos + 2];
    assert('0' <= AddrSpaceCode && AddrSpaceCode <= '4');
  }
  // make sure it's a pointer to private address space
  else {
    assert(CurFuncName.rfind('P') != StringRef::npos);
    AddrSpaceCode = (AMDGPUAS::PRIVATE_ADDRESS + '0');
  }
  return AddrSpaceCode;
} // AMDGPUclpVectorExpansion::getAddrSpaceCode

/// produce the next function name based on NextVecSize
///
std::string AMDGPUclpVectorExpansion::getNextFunctionName(StringRef CurFuncName,
                                                          an_typedes_t TypeDes,
                                                          int NextVecSize,
                                                          char NextAddrSpace) {
  assert(CurFuncName.startswith(OPENCL_VARG_BUILTIN_SPIR_PREFIX));

  // Dv is for Vector type followed by number of elements
  size_t SeparatorPos = CurFuncName.find(OPENCL_VARG_BUILTIN_SPIR_VECTYPE);
  assert(SeparatorPos > OPENCL_VARG_BUILTIN_PREFIX_LEN);

  StringRef ArgType;
  StringRef Suffix = CurFuncName.substr(SeparatorPos);

  std::ostringstream Oss;
  Oss << CurFuncName.slice(0, SeparatorPos).str();

  do {
    assert(Suffix.startswith(OPENCL_VARG_BUILTIN_SPIR_VECTYPE));
    Suffix = Suffix.substr(OPENCL_VARG_BUILTIN_SPIR_VECTYPE_LEN);
    getAndPassVecSize(Suffix);

    if (NextVecSize > 1) {
      Oss << OPENCL_VARG_BUILTIN_SPIR_VECTYPE << NextVecSize
          << Suffix.slice(0, OCLTypeLength(Suffix[1]) + 1).str();
      Suffix = Suffix.substr(OCLTypeLength(Suffix[1]) + 1);
      SeparatorPos = Suffix.find(OPENCL_VARG_BUILTIN_SPIR_VECTYPE);
      if (SeparatorPos == StringRef::npos && !Suffix.empty()) {
        Oss << Suffix.str();
        break;
      } else if (SeparatorPos != StringRef::npos && SeparatorPos > 0) {
        Oss << CurFuncName.slice(0, SeparatorPos).str();
        Suffix = Suffix.substr(SeparatorPos + 1);
      }
    } else {
      size_t ArgIndex = Suffix.find_first_of('_');
      assert(ArgIndex != StringRef::npos);
      ArgIndex++;
      ArgType = Suffix.slice(ArgIndex, OCLTypeLength(Suffix[ArgIndex]) + 1);
      Oss << ArgType.str();
      Suffix = Suffix.substr(ArgIndex + OCLTypeLength(Suffix[ArgIndex]));
      if (leadTypeDesIsPointer(TypeDes)) {
        // keep address space for pointers
        Oss << passPointerAddrSpace(Suffix).str();
      }
      Oss << replaceSubstituteTypes(Suffix, ArgType);
    }
  } while (!Suffix.empty());

  return Oss.str();
} // AMDGPUclpVectorExpansion::getNextFunctionName

/// process a function to collect information for funcuseInfo
///
void AMDGPUclpVectorExpansion::checkAndAddToExpansion(Function *TheFunc) {
  StringRef FuncName = TheFunc->getName();
  if (!FuncName.startswith(OPENCL_VARG_BUILTIN_SPIR_PREFIX))
    return;

  FuncName = FuncName.drop_front(OPENCL_VARG_BUILTIN_PREFIX_LEN);
  auto UnmangledLen = getAndPassVecSize(FuncName);

  // processing specific functions only
  if (!FuncName.substr(UnmangledLen)
           .startswith(OPENCL_VARG_BUILTIN_SPIR_VECTYPE))
    return;

  StringRef Suffix =
      FuncName.substr(UnmangledLen + OPENCL_VARG_BUILTIN_SPIR_VECTYPE_LEN);
  FuncName = FuncName.take_front(UnmangledLen);

  DEBUG(dbgs() << "check " << FuncName << "\n");
  const a_builtinfunc *Builtin = getBuiltinInfo(FuncName);
  if (Builtin) {
    int vecSize = getAndPassVecSize(Suffix);
    if (vecSize > 1) {
      addFuncuseInfo(TheFunc, vecSize, Builtin);
    }
  }
} // AMDGPUclpVectorExpansion::checkAndAddToExpansion

//===----------------------------------------------------------------------===//
#if !defined(NDEBUG)
/// check the consistency between need expansion and builtin info
///
void AMDGPUclpVectorExpansion::checkExpansionInfo() {
  DEBUG(dbgs() << "checkExpansionInfo\n");

  for (const auto &Iter : ExpansionInfo) {
    // check other info?
    DEBUG(dbgs() << Iter.getKey() << "\n");
    assert(Iter.second != nullptr && "missing builtin info");
    assert(canHandlePattern(Iter.second) && "can't handle builtin info");
  }

  DEBUG(dbgs() << "checkExpansionInfo end\n");
} // AMDGPUclpVectorExpansion::checkExpansionInfo

bool AMDGPUclpVectorExpansion::canHandlePattern(const a_builtinfunc *Table) {
  int Lead1 = Table->LeaderId;
  const char *TypeDes = Table->TypeDes;

  int I = 0;
  while (TypeDes[I]) {
    if (I == Lead1) {
      if (canHandled((an_typedes_t)TypeDes[I], HandledLead1TypeDes) == false)
        return false;
    } else {
      if (canHandled((an_typedes_t)TypeDes[I], HandledOtherTypeDes) == false)
        return false;
    }
    ++I;
  }
  return true;
} // AMDGPUclpVectorExpansion::canHandlePattern
#endif

/// loop through the input needExpansion builtin
///   and add the information to expansion Info
//
void AMDGPUclpVectorExpansion::addNeedExpansion(const char **Table) {
  int Idx = 0;
  DEBUG(dbgs() << "addNeedExpansion\n");
  while (Table[Idx]) {
    StringRef Name(Table[Idx]);
    DEBUG(dbgs() << Name << "\n");
    assert(ExpansionInfo.find(Name) == ExpansionInfo.end() &&
           "builtin is specified multiple times");
    ExpansionInfo[Name] = nullptr;
    ++Idx;
  }
  DEBUG(dbgs() << "addNeedExpansion end\n");
} // AMDGPUclpVectorExpansion::addNeedExpansion

/// loop through the input builtin info table
///    and match the information to the ExpansionInfo
///
void AMDGPUclpVectorExpansion::addBuiltinInfo(const a_builtinfunc *Table) {
  int Idx = 0;
  DEBUG(dbgs() << "addBuiltinInfo\n");
  while (Table[Idx].Name) {
    StringRef Name(Table[Idx].Name);
    auto Iter = ExpansionInfo.find(Name);
    if (Iter != ExpansionInfo.end()) {
      Iter->second = &Table[Idx];
      DEBUG(dbgs() << Name << " filled\n");
    }
    ++Idx;
  }
  DEBUG(dbgs() << "addBuiltinInfo end\n");
} // AMDGPUclpVectorExpansion::addBuiltinInfo

/// get the builtinInfo associated with an unmangled name
///
const a_builtinfunc *
AMDGPUclpVectorExpansion::getBuiltinInfo(StringRef &FuncName) {
  auto Iter = ExpansionInfo.find(FuncName);
  if (Iter != ExpansionInfo.end()) {
    assert(Iter->second != nullptr && "missing builtin info");
    return Iter->second;
  }
  return nullptr;
} // AMDGPUclpVectorExpansion::getBuiltinInfo

/// Add a function declared to funcuseInfo which is a collection
///   of builtin used in the compilation unit
///
void AMDGPUclpVectorExpansion::addFuncuseInfo(Function *TheFunc, int VecSize,
                                              const a_builtinfunc *builtin) {
  a_funcuse_t Funcuse(TheFunc, VecSize, builtin);
  checkAndExpand(&Funcuse);
} // AMDGPUclpVectorExpansion::addFuncuseInfo

//===----------------------------------------------------------------------===//
Function *AMDGPUclpVectorExpansion::adjustFunctionImpl(
    Function *CurFunc, int CurVecSize, const a_builtinfunc *Table,
    int NextVecSize, char NextAddrSpace) {
  int Lead1 = Table->LeaderId;
  const char *TypeDes = Table->TypeDes;

  FunctionType *CurFuncTy = CurFunc->getFunctionType();
  assert(CurFuncTy->getNumParams() == strlen(TypeDes + 1) &&
         "mismatching llvm function and builtin description");

  std::string NextFuncName =
      getNextFunctionName(CurFunc->getName(), (an_typedes_t)TypeDes[Lead1],
                          NextVecSize, NextAddrSpace);
  Function *NextFunc = TempModule->getFunction(NextFuncName);
  if (NextFunc) {
    return NextFunc;
  }

  Type *CurLeadTy = CurFuncTy->getParamType(Lead1 - 1);
  Type *NextLeadTy = getNextLeadType(CurLeadTy, (an_typedes_t)TypeDes[Lead1],
                                     CurVecSize, NextVecSize, NextAddrSpace);

  Type *CurRetTy = CurFuncTy->getReturnType();
  Type *NextRetTy = getNextOtherType(CurRetTy, CurVecSize, NextVecSize);

  SmallVector<Type *, 4> NextFuncArgs;
  int Idx = 1;
  for (FunctionType::param_iterator PI = CurFuncTy->param_begin(),
                                    PE = CurFuncTy->param_end();
       PI != PE; ++PI, ++Idx) {
    Type *CurTy = *PI;
    Type *NextTy;
    if (Idx == Lead1)
      NextTy = NextLeadTy;
    else
      NextTy = getNextOtherType(CurTy, CurVecSize, NextVecSize);

    NextFuncArgs.push_back(NextTy);
  }

  FunctionType *NextFuncTy = FunctionType::get(NextRetTy, NextFuncArgs, false);

  NextFunc = Function::Create(
      NextFuncTy,
      Function::ExternalLinkage, // set to Function::WeakAnyLinkage when
                                 // definition is created
      NextFuncName, TempModule.get());

  NextFunc->setCallingConv(CurFunc->getCallingConv());
  NextFunc->setAttributes(CurFunc->getAttributes());

  return NextFunc;
} // AMDGPUclpVectorExpansion::adjustFunctionImpl

Function *AMDGPUclpVectorExpansion::getNextFunction(Function *CurFunc,
                                                    int CurVecSize,
                                                    const a_builtinfunc *Table,
                                                    int &NextVecSize,
                                                    char NextAddrSpace) {

  if (CurVecSize == 3) {
    NextVecSize = 2;
  } else {
    NextVecSize = CurVecSize / 2;
    assert(NextVecSize * 2 == CurVecSize);
  }

  return adjustFunctionImpl(CurFunc, CurVecSize, Table, NextVecSize,
                            NextAddrSpace);
} // AMDGPUclpVectorExpansion::getNextFunction

void AMDGPUclpVectorExpansion::checkAndExpand(a_funcuse_t *Funcuse) {
  Function *CurFunc = Funcuse->LlvmFunc;
  int CurVecSize = Funcuse->VecSize;
  const a_builtinfunc *Builtin = Funcuse->Builtin;

  an_typedes_t LeadTypeDes =
      (an_typedes_t)Builtin->TypeDes[static_cast<int>(Builtin->LeaderId)];
  char CurASChar = InvalidASChar;
  bool IsFirst = true;

  if (leadTypeDesIsPointer(LeadTypeDes))
    CurASChar = getAddrSpaceCode(CurFunc->getName());

  while (CurVecSize > 1) {
    Function *NextFunc = nullptr;
    int NextVecSize = 0;

    if (leadTypeDesIsPointer(LeadTypeDes)) {
      if (IsFirst == false) {
        CurFunc = adjustFunctionImpl(CurFunc, CurVecSize, Builtin, CurVecSize,
                                     CurASChar);
        NextFunc = checkAndExpand(CurFunc, CurVecSize, Builtin, NextVecSize,
                                  CurASChar);
      }
      CurFunc = adjustFunctionImpl(CurFunc, CurVecSize, Builtin, CurVecSize,
                                   CurASChar);
      NextFunc =
          checkAndExpand(CurFunc, CurVecSize, Builtin, NextVecSize, CurASChar);
    } else {
      CurFunc = adjustFunctionImpl(CurFunc, CurVecSize, Builtin, CurVecSize,
                                   CurASChar);
      NextFunc =
          checkAndExpand(CurFunc, CurVecSize, Builtin, NextVecSize, CurASChar);
    }

    IsFirst = false;
    CurVecSize = NextVecSize;
    CurFunc = NextFunc;

  } // while CurVecSize > 1

} // AMDGPUclpVectorExpansion::checkAndExpand

Function *AMDGPUclpVectorExpansion::checkAndExpand(Function *CurFunc,
                                                   int CurVecSize,
                                                   const a_builtinfunc *Builtin,
                                                   int &NextVecSize,
                                                   char NextAddrSpace) {
  Function *NextFunc =
      getNextFunction(CurFunc, CurVecSize, Builtin, NextVecSize, NextAddrSpace);
  if (CurFunc->isDeclaration() == false) {
    return NextFunc;
  }

  Function *NextFuncHi =
      (CurVecSize == 3)
          ? adjustFunctionImpl(NextFunc, 2, Builtin, 1, NextAddrSpace)
          : NextFunc;

  // create an entry block,
  BasicBlock *EntryBlk =
      BasicBlock::Create(CurFunc->getContext(), "entry", CurFunc);

  FunctionType *CurFuncTy = CurFunc->getFunctionType();
  FunctionType *NextFuncTy = NextFunc->getFunctionType();

  Type *RetTy = CurFuncTy->getReturnType();
  Value *ResVal = nullptr;
  if (RetTy->isVoidTy() == false) {
    ResVal = UndefValue::get(RetTy);
  }

  int Lead1 = Builtin->LeaderId;
  PointerType *NextLeadTy =
      dyn_cast<PointerType>(NextFuncTy->getParamType(Lead1 - 1));

  // split the input parameter into two arg lists
  SmallVector<Value *, 8> HiArgs;
  SmallVector<Value *, 8> LoArgs;
  int Idx = 1;
  LLVMContext &TheContext = CurFunc->getContext();
  for (Function::arg_iterator AI = CurFunc->arg_begin(),
                              AE = CurFunc->arg_end();
       AI != AE; ++AI, ++Idx) {
    Value *CurArgVal = &*AI;
    std::ostringstream Oss;
    Oss << "_p" << Idx;
    CurArgVal->setName(Oss.str());
    if (Idx != Lead1 || !NextLeadTy) { // input
      Value *NextArgVal = loadVectorSlice(0, NextVecSize, CurArgVal, EntryBlk);
      LoArgs.push_back(NextArgVal);
      NextArgVal =
          loadVectorSlice(NextVecSize, CurVecSize, CurArgVal, EntryBlk);
      HiArgs.push_back(NextArgVal);
    } else { // output
      // if there is another part of the result is stored in a ptr parameter
      // get a pointer to pass for the partial result
      Value *PtrResLoc =
          new BitCastInst(CurArgVal, NextLeadTy, "ptrres1", EntryBlk);
      LoArgs.push_back(PtrResLoc);
      Type *Int32Ty = Type::getInt32Ty(TheContext);
      Value *IndexVal = ConstantInt::get(Int32Ty, (CurVecSize != 3) ? 1 : 2);
      Value *Index[] = {IndexVal};
      if (CurVecSize == 3) {
        unsigned AS = NextLeadTy->getPointerAddressSpace();
        Type *NextEltTy = NextLeadTy->getElementType()->getVectorElementType();
        Type *ScalarPtrTy = PointerType::get(NextEltTy, AS);
        PtrResLoc = new BitCastInst(CurArgVal, ScalarPtrTy, "", EntryBlk);
      }
      PtrResLoc = GetElementPtrInst::CreateInBounds(PtrResLoc, Index, "ptrres2",
                                                    EntryBlk);
      HiArgs.push_back(PtrResLoc);
    }
  }

  // process the lower part of the vector
  CallInst *CallInst = CallInst::Create(NextFunc, LoArgs, "lo.call", EntryBlk);
  CallInst->setCallingConv(NextFunc->getCallingConv());
  CallInst->setAttributes(NextFunc->getAttributes());
  if (ResVal) {
    ResVal = insertVectorSlice(0, NextVecSize, CallInst, ResVal, EntryBlk);
  }

  // process the higher part of the vector
  CallInst = CallInst::Create(NextFuncHi, HiArgs, "hi.call", EntryBlk);
  CallInst->setCallingConv(NextFunc->getCallingConv());
  CallInst->setAttributes(NextFunc->getAttributes());
  if (ResVal) {
    ResVal =
        insertVectorSlice(NextVecSize, CurVecSize, CallInst, ResVal, EntryBlk);
  }

  if (ResVal) {
    ReturnInst::Create(TheContext, ResVal, EntryBlk);
  }

  CurFunc->setLinkage(Function::WeakAnyLinkage);

// verify that the function is well formed.
#if !defined(NDEBUG)
  verifyFunction(*CurFunc);
#endif

  return NextFunc;
} // AMDGPUclpVectorExpansion::CheckAndExpand

Value *AMDGPUclpVectorExpansion::loadVectorSlice(int SrcStartIdx,
                                                 int SrcPassIdx, Value *SrcVal,
                                                 BasicBlock *BB) {
  int NumEle = SrcPassIdx - SrcStartIdx;
  assert(NumEle >= 1);

  Type *SrcTy = SrcVal->getType();
  assert(isa<VectorType>(SrcTy));

  Value *DstVal = nullptr;
  if (NumEle == 1) {
    Value *SrcIdx =
        ConstantInt::get(Type::getInt32Ty(BB->getContext()), SrcStartIdx);
    DstVal = ExtractElementInst::Create(SrcVal, SrcIdx, "", BB);
  } else {
    Type *EleTy = cast<VectorType>(SrcTy)->getElementType();
    Type *DstTy = VectorType::get(EleTy, NumEle);
    DstVal = UndefValue::get(DstTy);
    for (int i = SrcStartIdx; i < SrcPassIdx; ++i) {
      Value *SrcIdx = ConstantInt::get(Type::getInt32Ty(BB->getContext()), i);
      Value *TmpVal = ExtractElementInst::Create(SrcVal, SrcIdx, "", BB);
      Value *DstIdx =
          ConstantInt::get(Type::getInt32Ty(BB->getContext()), i - SrcStartIdx);
      DstVal = InsertElementInst::Create(DstVal, TmpVal, DstIdx, "", BB);
    }
  }

  return DstVal;
} // AMDGPUclpVectorExpansion::loadVectorSlice

Value *AMDGPUclpVectorExpansion::insertVectorSlice(int DstStartIdx,
                                                   int DstPassIdx,
                                                   Value *SrcVal, Value *DstVal,
                                                   BasicBlock *BB) {
  int NumSrcEle = DstPassIdx - DstStartIdx;
  for (int Idx = DstStartIdx; Idx < DstPassIdx; ++Idx) {
    Value *TmpVal = nullptr;

    if (NumSrcEle == 1) {
      TmpVal = SrcVal;
    } else {
      Value *SrcIdx = ConstantInt::get(Type::getInt32Ty(BB->getContext()),
                                       Idx - DstStartIdx);
      TmpVal = ExtractElementInst::Create(SrcVal, SrcIdx, "", BB);
    }
    Value *DstIdx = ConstantInt::get(Type::getInt32Ty(BB->getContext()), Idx);
    DstVal = InsertElementInst::Create(DstVal, TmpVal, DstIdx, "", BB);
  }

  return DstVal;
} // AMDGPUclpVectorExpansion::insertVectorSlice

AMDGPUclpVectorExpansion::AMDGPUclpVectorExpansion()
    : ModulePass(ID), TempModule(nullptr) {
  initializeAMDGPUclpVectorExpansionPass(*PassRegistry::getPassRegistry());
  addNeedExpansion(NeedExpansionBuiltins);
  addBuiltinInfo(MathFunc);
  addBuiltinInfo(IntegerFunc);
  addBuiltinInfo(CommonFunc);
  addBuiltinInfo(relationalFunc);
  addBuiltinInfo(pragmaEnableFunc);

#if !defined(NDEBUG)
  checkExpansionInfo();
#endif
} // AMDGPUclpVectorExpansion::AMDGPUclpVectorExpansion

/// process an LLVM module to perform expand vector builtin calls
///
bool AMDGPUclpVectorExpansion::runOnModule(Module &TheModule) {
  TempModule.reset(
      new Module("__opencllib_vectorexpansion", TheModule.getContext()));
  TempModule->setDataLayout(TheModule.getDataLayout());
  // loop through all Function to collect funcuseInfo
  for (auto &func : TheModule) {
    // Function must be a prototype and is used.
    if (func.isDeclaration() && func.use_empty() == false) {
      checkAndAddToExpansion(&func);
    }
  }
  if (!TempModule->empty()) {
    return !Linker::linkModules(TheModule, std::move(TempModule));
  }
  return false;
} // AMDGPUclpVectorExpansion::runOnModule

//===----------------------------------------------------------------------===//
// end-of-file newline
