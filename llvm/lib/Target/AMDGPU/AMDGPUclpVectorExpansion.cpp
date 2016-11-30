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
typedef enum {
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
} an_typedes_t;

#define MaxNumPara (3)

struct a_builtinfunc {
  const char *name;    // the name of the opencl builtin function
  char leaderId; // this is the id for the leading parameter, range from
                 // 1..LastParam,
  //-1 indicate nonoverloading
  char typeDes[MaxNumPara +
               1]; // type descriptor for return type, and each parameter
};                 // a_builtinfunc;

#define noLeads ((char)-1)
// 1 parameter
#define prttype1(r, p1)                                                        \
  { r, p1, tdInvalid, tdInvalid }
// 2 parameter
#define prttype2(r, p1, p2)                                                    \
  { r, p1, p2, tdInvalid }
// 3 parameter
#define prttype3(r, p1, p2, p3)                                                \
  { r, p1, p2, p3 }

#define endFuncGroups                                                          \
  {                                                                            \
    nullptr, noLeads, { tdInvalid, tdInvalid, tdInvalid, tdInvalid }           \
  }

typedef struct a_builtinfunc a_builtinfunc_t;

// section 6.12.2 math functions
// the version with double type is controled via whether the type is available
// (NULL not available)
//
static a_builtinfunc_t mathFunc[] = {
    {(const char *)"acos", 1, prttype1(tdFollow, tdAnyFloat)},
    {(const char *)"acosh", 1, prttype1(tdFollow, tdAnyFloat)},
    {(const char *)"acospi", 1, prttype1(tdFollow, tdAnyFloat)},
    {(const char *)"asin", 1, prttype1(tdFollow, tdAnyFloat)},
    {(const char *)"asinh", 1, prttype1(tdFollow, tdAnyFloat)},
    {(const char *)"asinpi", 1, prttype1(tdFollow, tdAnyFloat)},
    {(const char *)"atan", 1, prttype1(tdFollow, tdAnyFloat)},
    {(const char *)"atan2", 1, prttype2(tdFollow, tdAnyFloat, tdFollow)},
    {(const char *)"atanh", 1, prttype1(tdFollow, tdAnyFloat)},
    {(const char *)"atanpi", 1, prttype1(tdFollow, tdAnyFloat)},
    {(const char *)"atan2pi", 1, prttype2(tdFollow, tdAnyFloat, tdFollow)},

    {(const char *)"cbrt", 1, prttype1(tdFollow, tdAnyFloat)},
    {(const char *)"ceil", 1, prttype1(tdFollow, tdAnyFloat)},
    {(const char *)"copysign", 1, prttype2(tdFollow, tdAnyFloat, tdFollow)},
    {(const char *)"cos", 1, prttype1(tdFollow, tdAnyFloat)},
    {(const char *)"cosh", 1, prttype1(tdFollow, tdAnyFloat)},
    {(const char *)"cospi", 1, prttype1(tdFollow, tdAnyFloat)},
    {(const char *)"erfc", 1, prttype1(tdFollow, tdAnyFloat)},
    {(const char *)"erf", 1, prttype1(tdFollow, tdAnyFloat)},

    {(const char *)"exp", 1, prttype1(tdFollow, tdAnyFloat)},
    {(const char *)"exp2", 1, prttype1(tdFollow, tdAnyFloat)},
    {(const char *)"exp10", 1, prttype1(tdFollow, tdAnyFloat)},
    {(const char *)"expm1", 1, prttype1(tdFollow, tdAnyFloat)},
    {(const char *)"fabs", 1, prttype1(tdFollow, tdAnyFloat)},
    {(const char *)"fdim", 1, prttype2(tdFollow, tdAnyFloat, tdFollow)},
    {(const char *)"floor", 1, prttype1(tdFollow, tdAnyFloat)},

    {(const char *)"fma", 1, prttype3(tdFollow, tdAnyFloat, tdFollow, tdFollow)},
    {(const char *)"__builtin_fma", 1,
     prttype3(tdFollow, tdAnyFloat, tdFollow, tdFollow)},
    // in the spec, fmax/fmin is lised in three entry
    // we cover the last two entry by allowing scalar=>vector promotion
    {(const char *)"fmax", 1, prttype2(tdFollow, tdAnyFloat, tdFollow)},
    {(const char *)"fmin", 1, prttype2(tdFollow, tdAnyFloat, tdFollow)},
    {(const char *)"fmod", 1, prttype2(tdFollow, tdAnyFloat, tdFollow)},

    {(const char *)"fract", 2, prttype2(tdFollowPele, tdFollowPele, tdAnyFloat_PLG)},
    // fexp is described in group2
    {(const char *)"hypot", 1, prttype2(tdFollow, tdAnyFloat, tdFollow)},
    {(const char *)"ilogb", 1, prttype1(tdFollowVsInt, tdAnyFloat)},
    {(const char *)"ldexp", 1, prttype2(tdFollow, tdAnyFloat, tdFollowVsInt)},
    {(const char *)"lgamma", 1, prttype1(tdFollow, tdAnyFloat)},
    // lgamma_r is described in group2
    {(const char *)"log", 1, prttype1(tdFollow, tdAnyFloat)},
    {(const char *)"log2", 1, prttype1(tdFollow, tdAnyFloat)},
    {(const char *)"log10", 1, prttype1(tdFollow, tdAnyFloat)},
    {(const char *)"log1p", 1, prttype1(tdFollow, tdAnyFloat)},
    {(const char *)"logb", 1, prttype1(tdFollow, tdAnyFloat)},
    {(const char *)"mad", 1, prttype3(tdFollow, tdAnyFloat, tdFollow, tdFollow)},
    {(const char *)"maxmag", 1, prttype2(tdFollow, tdAnyFloat, tdFollow)},
    {(const char *)"minmag", 1, prttype2(tdFollow, tdAnyFloat, tdFollow)},
    {(const char *)"modf", 2, prttype2(tdFollowPele, tdFollowPele, tdAnyFloat_PLG)},
    {(const char *)"nan", 1,
     prttype1(tdFollowVsHalfFloatDoublen, tdAnyUintk16_32_64)},
    {(const char *)"nextafter", 1, prttype2(tdFollow, tdAnyFloat, tdFollow)},
    {(const char *)"pow", 1, prttype2(tdFollow, tdAnyFloat, tdFollow)},
    {(const char *)"pown", 1, prttype2(tdFollow, tdAnyFloat, tdFollowVsInt)},
    {(const char *)"powr", 1, prttype2(tdFollow, tdAnyFloat, tdFollow)},
    {(const char *)"remainder", 1, prttype2(tdFollow, tdAnyFloat, tdFollow)},
    // remquo is described in group2
    {(const char *)"rint", 1, prttype1(tdFollow, tdAnyFloat)},
    {(const char *)"rootn", 1, prttype2(tdFollow, tdAnyFloat, tdFollowVsInt)},
    {(const char *)"round", 1, prttype1(tdFollow, tdAnyFloat)},
    {(const char *)"rsqrt", 1, prttype1(tdFollow, tdAnyFloat)},
    {(const char *)"sin", 1, prttype1(tdFollow, tdAnyFloat)},
    {(const char *)"sincos", 2, prttype2(tdFollowPele, tdFollowPele, tdAnyFloat_PLG)},
    {(const char *)"sinh", 1, prttype1(tdFollow, tdAnyFloat)},
    {(const char *)"sinpi", 1, prttype1(tdFollow, tdAnyFloat)},
    {(const char *)"sqrt", 1, prttype1(tdFollow, tdAnyFloat)},
    {(const char *)"tan", 1, prttype1(tdFollow, tdAnyFloat)},
    {(const char *)"tanh", 1, prttype1(tdFollow, tdAnyFloat)},
    {(const char *)"tanpi", 1, prttype1(tdFollow, tdAnyFloat)},
    {(const char *)"tgamma", 1, prttype1(tdFollow, tdAnyFloat)},
    {(const char *)"trunc", 1, prttype1(tdFollow, tdAnyFloat)},

    {(const char *)"half_cos", 1, prttype1(tdFollow, tdAnySingle)},
    {(const char *)"half_divide", 1, prttype2(tdFollow, tdAnySingle, tdFollow)},
    {(const char *)"half_exp", 1, prttype1(tdFollow, tdAnySingle)},
    {(const char *)"half_exp2", 1, prttype1(tdFollow, tdAnySingle)},
    {(const char *)"half_exp10", 1, prttype1(tdFollow, tdAnySingle)},
    {(const char *)"half_log", 1, prttype1(tdFollow, tdAnySingle)},
    {(const char *)"half_log2", 1, prttype1(tdFollow, tdAnySingle)},
    {(const char *)"half_log10", 1, prttype1(tdFollow, tdAnySingle)},
    {(const char *)"half_powr", 1, prttype2(tdFollow, tdAnySingle, tdFollow)},
    {(const char *)"half_recip", 1, prttype1(tdFollow, tdAnySingle)},
    {(const char *)"half_rsqrt", 1, prttype1(tdFollow, tdAnySingle)},
    {(const char *)"half_sin", 1, prttype1(tdFollow, tdAnySingle)},
    {(const char *)"half_sqrt", 1, prttype1(tdFollow, tdAnySingle)},
    {(const char *)"half_tan", 1, prttype1(tdFollow, tdAnySingle)},
    {(const char *)"native_cos", 1, prttype1(tdFollow, tdAnySingle)},
    {(const char *)"native_divide", 1, prttype2(tdFollow, tdAnyFloat, tdFollow)},
    {(const char *)"native_exp", 1, prttype1(tdFollow, tdAnySingle)},
    {(const char *)"native_exp2", 1, prttype1(tdFollow, tdAnySingle)},
    {(const char *)"native_exp10", 1, prttype1(tdFollow, tdAnySingle)},
    {(const char *)"native_log", 1, prttype1(tdFollow, tdAnySingle)},
    {(const char *)"native_log2", 1, prttype1(tdFollow, tdAnySingle)},
    {(const char *)"native_log10", 1, prttype1(tdFollow, tdAnySingle)},
    {(const char *)"native_powr", 1, prttype2(tdFollow, tdAnySingle, tdFollow)},
    {(const char *)"native_recip", 1, prttype1(tdFollow, tdAnyFloat)},
    {(const char *)"native_rsqrt", 1, prttype1(tdFollow, tdAnyFloat)},
    {(const char *)"native_sin", 1, prttype1(tdFollow, tdAnySingle)},
    {(const char *)"native_sqrt", 1, prttype1(tdFollow, tdAnyFloat)},
    {(const char *)"native_tan", 1, prttype1(tdFollow, tdAnySingle)},
    endFuncGroups};

// section 6.12.3 integer functions
static a_builtinfunc_t integerFunc[] = {
    {(const char *)"abs", 1, prttype1(tdFollowKu, tdAnyInt)},
    {(const char *)"abs_diff", 1, prttype2(tdFollowKu, tdAnyInt, tdFollow)},
    {(const char *)"add_sat", 1, prttype2(tdFollow, tdAnyInt, tdFollow)},
    {(const char *)"hadd", 1, prttype2(tdFollow, tdAnyInt, tdFollow)},
    {(const char *)"rhadd", 1, prttype2(tdFollow, tdAnyInt, tdFollow)},
    {(const char *)"clamp", 1, prttype3(tdFollow, tdAnyIntFloat, tdFollow, tdFollow)},
    {(const char *)"clz", 1, prttype1(tdFollow, tdAnyInt)},
    {(const char *)"ctz", 1, prttype1(tdFollow, tdAnyInt)},
    {(const char *)"mad_hi", 1, prttype3(tdFollow, tdAnyInt, tdFollow, tdFollow)},
    {(const char *)"mad_sat", 1, prttype3(tdFollow, tdAnyInt, tdFollow, tdFollow)},
    {(const char *)"max", 1, prttype2(tdFollow, tdAnyIntFloat, tdFollow)},
    {(const char *)"min", 1, prttype2(tdFollow, tdAnyIntFloat, tdFollow)},
    {(const char *)"mul_hi", 1, prttype2(tdFollow, tdAnyInt, tdFollow)},
    {(const char *)"rotate", 1, prttype2(tdFollow, tdAnyInt, tdFollow)},
    {(const char *)"sub_sat", 1, prttype2(tdFollow, tdAnyInt, tdFollow)},
    {(const char *)"upsample", 1, prttype2(tdFollowKn, tdAnyIntk8_32, tdFollowKu)},
    {(const char *)"popcount", 1, prttype1(tdFollow, tdAnyInt)},
    {(const char *)"mad24", 1,
     prttype3(tdFollow, tdAnyIntk32_32, tdFollow, tdFollow)},
    {(const char *)"mul24", 1, prttype2(tdFollow, tdAnyIntk32_32, tdFollow)},
    endFuncGroups};

// section 6.12.4 common functions
static a_builtinfunc_t commonFunc[] = {
    // clamp is described in integer function
    {(const char *)"degrees", 1, prttype1(tdFollow, tdAnyFloat)},
    // max, min is described in integer function
    {(const char *)"mix", 1, prttype3(tdFollow, tdAnyFloat, tdFollow, tdFollow)},
    {(const char *)"radians", 1, prttype1(tdFollow, tdAnyFloat)},
    {(const char *)"step", 1, prttype2(tdFollow, tdAnyFloat, tdFollow)},
    {(const char *)"smoothstep", 1,
     prttype3(tdFollow, tdAnyFloat, tdFollow, tdFollow)},
    {(const char *)"sign", 1, prttype1(tdFollow, tdAnyFloat)},
    endFuncGroups};

// section 6.12.6 relational functions
// 1.2 spec has a problem in isequal(double, double) return type
static a_builtinfunc_t relationalFunc[] = {
    {(const char *)"isequal", 1,
     prttype2(tdFollowVsShortIntLongn, tdAnyFloat, tdFollow)},
    {(const char *)"isnotequal", 1,
     prttype2(tdFollowVsShortIntLongn, tdAnyFloat, tdFollow)},
    {(const char *)"isgreater", 1,
     prttype2(tdFollowVsShortIntLongn, tdAnyFloat, tdFollow)},
    {(const char *)"isgreaterequal", 1,
     prttype2(tdFollowVsShortIntLongn, tdAnyFloat, tdFollow)},
    {(const char *)"isless", 1,
     prttype2(tdFollowVsShortIntLongn, tdAnyFloat, tdFollow)},
    {(const char *)"islessequal", 1,
     prttype2(tdFollowVsShortIntLongn, tdAnyFloat, tdFollow)},
    {(const char *)"islessgreater", 1,
     prttype2(tdFollowVsShortIntLongn, tdAnyFloat, tdFollow)},
    {(const char *)"isfinite", 1, prttype1(tdFollowVsShortIntLongn, tdAnyFloat)},
    {(const char *)"isinf", 1, prttype1(tdFollowVsShortIntLongn, tdAnyFloat)},
    {(const char *)"isnan", 1, prttype1(tdFollowVsShortIntLongn, tdAnyFloat)},
    {(const char *)"isnormal", 1, prttype1(tdFollowVsShortIntLongn, tdAnyFloat)},
    {(const char *)"isordered", 1,
     prttype2(tdFollowVsShortIntLongn, tdAnyFloat, tdFollow)},
    {(const char *)"isunordered", 1,
     prttype2(tdFollowVsShortIntLongn, tdAnyFloat, tdFollow)},
    {(const char *)"signbit", 1, prttype1(tdFollowVsShortIntLongn, tdAnyFloat)},
    {(const char *)"any", 1, prttype1(tdInt, tdAnySint)},
    {(const char *)"all", 1, prttype1(tdInt, tdAnySint)},
    {(const char *)"bitselect", 1,
     prttype3(tdFollow, tdAnyIntFloat, tdFollow, tdFollow)},
    // select is described in group2
    endFuncGroups};

static a_builtinfunc_t pragmaEnableFunc[] = {
    {(const char *)"popcnt", 1, prttype1(tdFollow, tdAnyInt)}, endFuncGroups};

//===----------------------------------------------------------------------===//
// info that describes what to expand
// it is currently a list of builtin names
// as we weak expand all vector size for all targets

// use the order of commonVectorExpansions.cl
//
const char *needExpansionBuiltins[] = {
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

an_typedes_t handledLead1TypeDes[] = {
    // todo: fix the type name an=>a
    tdAnyIntFloat, tdAnyFloat,     tdAnySingle,        tdAnyInt,
    tdAnyIntk8_32, tdAnyIntk32_32, tdAnyUintk16_32_64, tdAnyFloat_PLG,
    tdInvalid // 0
};

// need to test popcnt
an_typedes_t handledOtherTypeDes[] = {
    tdFollow,
    tdFollowKu,
    tdFollowKn,                 // usample
    tdFollowVsInt,              // ilogb
    tdFollowVsHalfFloatDoublen, // nan
    tdFollowPele,
    tdInvalid // 0
};

static bool leadTypeDesIsPointer(an_typedes_t typeDes) {
  return typeDes == tdAnyFloat_PLG;
} // leadTypeDesIsPointer

#define InvalidASChar '\0'

static Type *getNextTypeImp(Type *curTy, int curVecSize, int nextVecSize,
                            bool isPointer, char nextAddrSpace) {
  Type *curVecTy = isa<PointerType>(curTy)
                       ? cast<PointerType>(curTy)->getElementType()
                       : curTy;

  assert(isa<VectorType>(curVecTy) && "problem with builtin description");
  assert(static_cast<int>(cast<VectorType>(curVecTy)->getNumElements()) ==
             curVecSize &&
         "problem with builtin description");

  Type *curEleTy = cast<VectorType>(curVecTy)->getElementType();

  Type *nextTy;
  if (nextVecSize == 1)
    nextTy = curEleTy;
  else
    nextTy = VectorType::get(curEleTy, nextVecSize);
  if (isPointer) {
    assert('0' <= nextAddrSpace && nextAddrSpace <= '4' &&
           "incorrect addrspace");
    nextTy = PointerType::get(
        nextTy, nextAddrSpace == InvalidASChar ? 0 : (nextAddrSpace - '0'));
  }

  return nextTy;
} // getNextTypeImp

static Type *getNextLeadType(Type *curLeadTy, an_typedes_t typeDes,
                             int curVecSize, int nextVecSize,
                             char nextAddrSpace) {
  assert(((leadTypeDesIsPointer(typeDes) && isa<PointerType>(curLeadTy)) ||
         isa<PointerType>(curLeadTy) == false) &&
             "problem with builtin description");

  return getNextTypeImp(curLeadTy, curVecSize, nextVecSize,
                        leadTypeDesIsPointer(typeDes), nextAddrSpace);
} // getNextLeadType

static Type *getNextOtherType(Type *curTy, an_typedes_t typeDes, int curVecSize,
                              int nextVecSize) {
  return getNextTypeImp(curTy, curVecSize, nextVecSize, false, InvalidASChar);
} // getNextOtherType

static int getAndPassVecSize(StringRef &str) {
  int res = 0;
  while (str[0] >= '1' && str[0] <= '9') {
    res = res * 10 + str[0] - '0';
    str = str.substr(1);
  }
  if (res == 0)
    res = 1;

  return res;
} // getAndPassVecSize

static bool canHandled(an_typedes_t typeDes, an_typedes_t *parray) {
  int i = 0;
  while (parray[i] != 0 && parray[i] != typeDes)
    ++i;
  return (parray[i] != tdInvalid);
} // canHandled

// data structure to collect use of builtin functions
struct a_funcuse_t {
  Function *llvmFunc;
  a_builtinfunc_t *builtin;
  int vecSize; // number of vector elements
  a_funcuse_t() : llvmFunc(0), builtin(0), vecSize(0) {}
  a_funcuse_t(Function *f, int v, a_builtinfunc_t *b)
      : llvmFunc(f), builtin(b), vecSize(v) {}
};

namespace llvm {
class AMDGPUclpVectorExpansion : public ModulePass {
  // data structure to provide builtin names => builtin info mapping
  typedef StringMap<a_builtinfunc_t *> an_expansionInfo_t;
  an_expansionInfo_t expansionInfo;
  std::unique_ptr<Module> tmpModule;

  char getAddrSpaceCode(StringRef);
  std::string getNextFunctionName(StringRef, an_typedes_t, int, int, char);
  void checkAndAddToExpansion(Function *);
  std::string replaceSubstituteTypes(StringRef &, StringRef);
  StringRef passPointerAddrSpace(StringRef &);

public:
  AMDGPUclpVectorExpansion();
  bool runOnModule(Module &) override;

protected:
  void addBuiltinInfo(a_builtinfunc_t *);
  void addNeedExpansion(const char **);

#if !defined(NDEBUG)
  void checkExpansionInfo();
  bool canHandlePattern(a_builtinfunc_t *);
#endif

  void addFuncuseInfo(Function *, StringRef &name, int vecSize,
                      a_builtinfunc_t *);
  a_builtinfunc_t *getBuiltinInfo(StringRef &name);

  void checkAndExpand(a_funcuse_t *);
  Function *checkAndExpand(Function *, int, a_builtinfunc_t *, int &,
                           char nextAddrSpace);

  Value *loadVectorSlice(int, int, Value *, BasicBlock *);
  Value *insertVectorSlice(int, int, Value *, Value *, BasicBlock *);

  Function *getNextFunction(Function *curFunc, int curVecSize,
                            a_builtinfunc_t *builtin, int &nextVecSize,
                            char nextAddrSpace);
  Function *adjustFunctionImpl(Function *curFunc, int curVecSize,
                               a_builtinfunc_t *builtin, int nextVecSize, char);

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

StringRef AMDGPUclpVectorExpansion::passPointerAddrSpace(StringRef &suffix) {
  assert(suffix[0] == 'P' && "internal problem");
  StringRef addrSpace;
  if (suffix.rfind("PU") !=
      StringRef::npos) { // pointer to non-private address space
    addrSpace = suffix.slice(0, OPENCL_VARG_BUILTIN_ADDRSPACE_LEN);
    suffix = suffix.substr(OPENCL_VARG_BUILTIN_ADDRSPACE_LEN);
  } else { // pointer to private address space
    addrSpace = suffix.slice(0, 0);
    suffix = suffix.substr(1);
  }
  return addrSpace;
} // passPointerAddrSpace

/// replace substitution mangled types
///
std::string
AMDGPUclpVectorExpansion::replaceSubstituteTypes(StringRef &suffix,
                                                 StringRef argType) {
  std::ostringstream oss;
  while (!suffix.empty() && suffix[0] == 'S') {
    oss << argType.str();
    suffix = suffix.substr(suffix.find_first_of('_') + 1);
  }
  return oss.str();
} // replaceSubstituteTypes

/// extract the address space code from the current function name
///
char AMDGPUclpVectorExpansion::getAddrSpaceCode(StringRef curFuncName) {
  char addrSpaceCode = InvalidASChar;
  assert(curFuncName.startswith(OPENCL_VARG_BUILTIN_SPIR_PREFIX));

  // pointer to non-private address space
  if (curFuncName.rfind("PU") != StringRef::npos) {
    //<address-space-number> consists of 'AS' followed by the address space
    //<number>
    size_t separatorPos = curFuncName.rfind("AS");
    assert(separatorPos != StringRef::npos);
    addrSpaceCode = curFuncName[separatorPos + 2];
    assert('0' <= addrSpaceCode && addrSpaceCode <= '4');
  }
  // make sure it's a pointer to private address space
  else {
    assert(curFuncName.rfind('P') != StringRef::npos);
    addrSpaceCode = (AMDGPUAS::PRIVATE_ADDRESS + '0');
  }
  return addrSpaceCode;
} // AMDGPUclpVectorExpansion::getAddrSpaceCode

/// produce the next function name based on nextVecSize
///
std::string AMDGPUclpVectorExpansion::getNextFunctionName(StringRef curFuncName,
                                                          an_typedes_t typeDes,
                                                          int curVecSize,
                                                          int nextVecSize,
                                                          char nextAddrSpace) {
  assert(curFuncName.startswith(OPENCL_VARG_BUILTIN_SPIR_PREFIX));

  // Dv is for Vector type followed by number of elements
  size_t separatorPos = curFuncName.find(OPENCL_VARG_BUILTIN_SPIR_VECTYPE);
  assert(separatorPos > OPENCL_VARG_BUILTIN_PREFIX_LEN);

  StringRef argType;
  StringRef suffix = curFuncName.substr(separatorPos);
  //  llvm::raw_ostream oss1;
  std::ostringstream oss;
  oss << curFuncName.slice(0, separatorPos).str();

  do {
    assert(suffix.startswith(OPENCL_VARG_BUILTIN_SPIR_VECTYPE));
    suffix = suffix.substr(OPENCL_VARG_BUILTIN_SPIR_VECTYPE_LEN);
    int actVecSize = getAndPassVecSize(suffix);
    assert(curVecSize == actVecSize);

    if (nextVecSize > 1) {
      oss << OPENCL_VARG_BUILTIN_SPIR_VECTYPE << nextVecSize
          << suffix.slice(0, OCLTypeLength(suffix[1]) + 1).str();
      suffix = suffix.substr(OCLTypeLength(suffix[1]) + 1);
      separatorPos = suffix.find(OPENCL_VARG_BUILTIN_SPIR_VECTYPE);
      if (separatorPos == StringRef::npos && !suffix.empty()) {
        oss << suffix.str();
        break;
      } else if (separatorPos != StringRef::npos && separatorPos > 0) {
        oss << curFuncName.slice(0, separatorPos).str();
        suffix = suffix.substr(separatorPos + 1);
      }
    } else {
      size_t argIndex = suffix.find_first_of('_');
      assert(argIndex != StringRef::npos);
      argIndex++;
      argType = suffix.slice(argIndex, OCLTypeLength(suffix[argIndex]) + 1);
      oss << argType.str();
      suffix = suffix.substr(argIndex + OCLTypeLength(suffix[argIndex]));
      if (leadTypeDesIsPointer(typeDes)) {
        // keep address space for pointers
        oss << passPointerAddrSpace(suffix).str();
      }
      oss << replaceSubstituteTypes(suffix, argType);
    }
  } while (!suffix.empty());

  return oss.str();
} // AMDGPUclpVectorExpansion::getNextFunctionName

/// process a function to collect information for funcuseInfo
///
void AMDGPUclpVectorExpansion::checkAndAddToExpansion(Function *theFunc) {
  StringRef funcName = theFunc->getName();
  size_t separatorPos = funcName.find(OPENCL_VARG_BUILTIN_SPIR_VECTYPE);
  if (funcName.startswith(OPENCL_VARG_BUILTIN_SPIR_PREFIX) &&
      separatorPos != StringRef::npos) {
    StringRef subStr =
        funcName.slice(OPENCL_VARG_BUILTIN_PREFIX_LEN, separatorPos);

    // drop function length
    assert(subStr[0] >= '0' && subStr[0] <= '9');
    while (subStr[0] >= '0' && subStr[0] <= '9') {
      subStr = subStr.substr(1);
    }

    DEBUG(dbgs() << "check " << subStr << "\n");

    a_builtinfunc_t *builtin = getBuiltinInfo(subStr);
    if (builtin) {
      StringRef suffix =
          funcName.substr(separatorPos + OPENCL_VARG_BUILTIN_SPIR_VECTYPE_LEN);
      int vecSize = getAndPassVecSize(suffix);
      if (vecSize > 1) {
        addFuncuseInfo(theFunc, subStr, vecSize, builtin);
      }
    }
  }
} // AMDGPUclpVectorExpansion::checkAndAddToExpansion

//===----------------------------------------------------------------------===//
#if !defined(NDEBUG)
/// check the consistency between need expansion and builtin info
///
void AMDGPUclpVectorExpansion::checkExpansionInfo() {
  DEBUG(dbgs() << "checkExpansionInfo\n");

  for (const auto &iter : expansionInfo) {
    // check other info?
    DEBUG(dbgs() << iter.getKey() << "\n");
    assert(iter.second != nullptr && "missing builtin info");
    assert(canHandlePattern(iter.second) && "can't handle builtin info");
  }

  DEBUG(dbgs() << "checkExpansionInfo end\n");
} // AMDGPUclpVectorExpansion::checkExpansionInfo

bool AMDGPUclpVectorExpansion::canHandlePattern(a_builtinfunc_t *tbl) {
  int lead1 = tbl->leaderId;
  char *typeDes = tbl->typeDes;

  int i = 0;
  while (typeDes[i]) {
    if (i == lead1) {
      if (canHandled((an_typedes_t)typeDes[i], handledLead1TypeDes) == false)
        return false;
    } else {
      if (canHandled((an_typedes_t)typeDes[i], handledOtherTypeDes) == false)
        return false;
    }
    ++i;
  }
  return true;
} // AMDGPUclpVectorExpansion::canHandlePattern
#endif

/// loop through the input needExpansion builtin
///   and add the information to expansion Info
//
void AMDGPUclpVectorExpansion::addNeedExpansion(const char **table) {
  int idx = 0;
  DEBUG(dbgs() << "addNeedExpansion\n");
  while (table[idx]) {
    StringRef name(table[idx]);
    DEBUG(dbgs() << name << "\n");
    assert(expansionInfo.find(name) == expansionInfo.end() &&
           "builtin is specified multiple times");
    expansionInfo[name] = nullptr;
    ++idx;
  }
  DEBUG(dbgs() << "addNeedExpansion end\n");
} // AMDGPUclpVectorExpansion::addNeedExpansion

/// loop through the input builtin info table
///    and match the information to the expansionInfo
///
void AMDGPUclpVectorExpansion::addBuiltinInfo(a_builtinfunc_t *table) {
  int idx = 0;
  DEBUG(dbgs() << "addBuiltinInfo\n");
  while (table[idx].name) {
    StringRef name(table[idx].name);
    auto iter = expansionInfo.find(name);
    if (iter != expansionInfo.end()) {
      iter->second = &table[idx];
      DEBUG(dbgs() << name << " filled\n");
    }
    ++idx;
  }
  DEBUG(dbgs() << "addBuiltinInfo end\n");
} // AMDGPUclpVectorExpansion::addBuiltinInfo

/// get the builtinInfo associated with an unmangled name
///
a_builtinfunc_t *AMDGPUclpVectorExpansion::getBuiltinInfo(StringRef &name) {
  auto iter = expansionInfo.find(name);
  if (iter != expansionInfo.end()) {
    assert(iter->second != nullptr && "missing builtin info");
    return iter->second;
  }
  return nullptr;
} // AMDGPUclpVectorExpansion::getBuiltinInfo

/// Add a function declared to funcuseInfo which is a collection
///   of builtin used in the compilation unit
///
void AMDGPUclpVectorExpansion::addFuncuseInfo(Function *theFunc,
                                              StringRef &name, int vecSize,
                                              a_builtinfunc_t *builtin) {
  DEBUG(dbgs() << "addFuncuseInfo " << name);
  a_funcuse_t funcuse(theFunc, vecSize, builtin);
  checkAndExpand(&funcuse);
} // AMDGPUclpVectorExpansion::addFuncuseInfo

//===----------------------------------------------------------------------===//
Function *AMDGPUclpVectorExpansion::adjustFunctionImpl(Function *curFunc,
                                                       int curVecSize,
                                                       a_builtinfunc_t *tbl,
                                                       int nextVecSize,
                                                       char nextAddrSpace) {
  int lead1 = tbl->leaderId;
  char *typeDes = tbl->typeDes;

  FunctionType *curFuncTy = curFunc->getFunctionType();
  assert(curFuncTy->getNumParams() == strlen(typeDes + 1) &&
         "mismatching llvm function and builtin description");

  std::string nextFuncName =
      getNextFunctionName(curFunc->getName(), (an_typedes_t)typeDes[lead1],
                          curVecSize, nextVecSize, nextAddrSpace);
  Function *nextFunc = tmpModule->getFunction(nextFuncName);
  if (nextFunc) {
    return nextFunc;
  }

  Type *curLeadTy = curFuncTy->getParamType(lead1 - 1);
  Type *nextLeadTy = getNextLeadType(curLeadTy, (an_typedes_t)typeDes[lead1],
                                     curVecSize, nextVecSize, nextAddrSpace);

  Type *curRetTy = curFuncTy->getReturnType();
  Type *nextRetTy = getNextOtherType(curRetTy, (an_typedes_t)typeDes[0],
                                     curVecSize, nextVecSize);

  SmallVector<Type *, 4> nextFuncArgs;
  int i = 1;
  for (FunctionType::param_iterator pi = curFuncTy->param_begin(),
                                    pe = curFuncTy->param_end();
       pi != pe; ++pi, ++i) {
    Type *curTy = *pi;
    Type *nextTy;
    if (i == lead1)
      nextTy = nextLeadTy;
    else
      nextTy = getNextOtherType(curTy, (an_typedes_t)typeDes[i], curVecSize,
                                nextVecSize);

    nextFuncArgs.push_back(nextTy);
  }

  FunctionType *nextFuncTy = FunctionType::get(nextRetTy, nextFuncArgs, false);

  nextFunc = Function::Create(
      nextFuncTy,
      Function::ExternalLinkage, // set to Function::WeakAnyLinkage when
                                 // definition is created
      nextFuncName, tmpModule.get());

  nextFunc->setCallingConv(curFunc->getCallingConv());
  nextFunc->setAttributes(curFunc->getAttributes());

  return nextFunc;
} // AMDGPUclpVectorExpansion::adjustFunctionImpl

Function *AMDGPUclpVectorExpansion::getNextFunction(Function *curFunc,
                                                    int curVecSize,
                                                    a_builtinfunc_t *tbl,
                                                    int &nextVecSize,
                                                    char nextAddrSpace) {

  if (curVecSize == 3) {
    nextVecSize = 2;
  } else {
    nextVecSize = curVecSize / 2;
    assert(nextVecSize * 2 == curVecSize);
  }

  return adjustFunctionImpl(curFunc, curVecSize, tbl, nextVecSize,
                            nextAddrSpace);
} // AMDGPUclpVectorExpansion::getNextFunction

void AMDGPUclpVectorExpansion::checkAndExpand(a_funcuse_t *funcuse) {
  Function *curFunc = funcuse->llvmFunc;
  int curVecSize = funcuse->vecSize;
  a_builtinfunc_t *builtin = funcuse->builtin;

  an_typedes_t leadTypeDes =
      (an_typedes_t)builtin->typeDes[static_cast<int>(builtin->leaderId)];
  char curASChar = InvalidASChar;
  bool isFirst = true;

  if (leadTypeDesIsPointer(leadTypeDes))
    curASChar = getAddrSpaceCode(curFunc->getName());

  while (curVecSize > 1) {
    Function *nextFunc = nullptr;
    int nextVecSize = 0;

    if (leadTypeDesIsPointer(leadTypeDes)) {
      if (isFirst == false) {
        curFunc = adjustFunctionImpl(curFunc, curVecSize, builtin, curVecSize,
                                     curASChar);
        nextFunc = checkAndExpand(curFunc, curVecSize, builtin, nextVecSize,
                                  curASChar);
      }
      curFunc = adjustFunctionImpl(curFunc, curVecSize, builtin, curVecSize,
                                   curASChar);
      nextFunc =
          checkAndExpand(curFunc, curVecSize, builtin, nextVecSize, curASChar);
    } else {
      curFunc = adjustFunctionImpl(curFunc, curVecSize, builtin, curVecSize,
                                   curASChar);
      nextFunc =
          checkAndExpand(curFunc, curVecSize, builtin, nextVecSize, curASChar);
    }

    isFirst = false;
    curVecSize = nextVecSize;
    curFunc = nextFunc;

  } // while curVecSize > 1

} // AMDGPUclpVectorExpansion::checkAndExpand

Function *AMDGPUclpVectorExpansion::checkAndExpand(Function *curFunc,
                                                   int curVecSize,
                                                   a_builtinfunc_t *builtin,
                                                   int &nextVecSize,
                                                   char nextAddrSpace) {
  Function *nextFunc =
      getNextFunction(curFunc, curVecSize, builtin, nextVecSize, nextAddrSpace);
  if (curFunc->isDeclaration() == false) {
    return nextFunc;
  }

  Function *nextFuncHi =
      (curVecSize == 3)
          ? adjustFunctionImpl(nextFunc, 2, builtin, 1, nextAddrSpace)
          : nextFunc;

  // create an entry block,
  BasicBlock *entryBlk =
      BasicBlock::Create(curFunc->getContext(), "entry", curFunc);
  // TerminatorInst* lastInstr =curFunc->begin()->getTerminator();

  FunctionType *curFuncTy = curFunc->getFunctionType();
  FunctionType *nextFuncTy = nextFunc->getFunctionType();

  Type *retTy = curFuncTy->getReturnType();
  Value *resVal = nullptr;
  if (retTy->isVoidTy() == false) {
    resVal = UndefValue::get(retTy);
  }

  int lead1 = builtin->leaderId;
  PointerType *nextLeadTy =
      dyn_cast<PointerType>(nextFuncTy->getParamType(lead1 - 1));

  // split the input parameter into two arg lists
  SmallVector<Value *, 8> hiArgs;
  SmallVector<Value *, 8> loArgs;
  int i = 1;
  LLVMContext &theContext = curFunc->getContext();
  for (Function::arg_iterator ai = curFunc->arg_begin(),
                              ae = curFunc->arg_end();
       ai != ae; ++ai, ++i) {
    Value *curArgVal = &*ai;
    std::ostringstream oss;
    oss << "_p" << i;
    curArgVal->setName(oss.str());
    if (i != lead1 || !nextLeadTy) { // input
      Value *nextArgVal = loadVectorSlice(0, nextVecSize, curArgVal, entryBlk);
      loArgs.push_back(nextArgVal);
      nextArgVal =
          loadVectorSlice(nextVecSize, curVecSize, curArgVal, entryBlk);
      hiArgs.push_back(nextArgVal);
    } else { // output
      // if there is another part of the result is stored in a ptr parameter
      // get a pointer to pass for the partial result
      Value *ptrResLoc =
          new BitCastInst(curArgVal, nextLeadTy, "ptrres1", entryBlk);
      loArgs.push_back(ptrResLoc);
      Type *Int32Ty = Type::getInt32Ty(theContext);
      Value *IndexVal = ConstantInt::get(Int32Ty, (curVecSize != 3) ? 1 : 2);
      Value *Index[] = {IndexVal};
      if (curVecSize == 3) {
        unsigned as = nextLeadTy->getPointerAddressSpace();
        Type *nextEltTy = nextLeadTy->getElementType()->getVectorElementType();
        Type *scalarPtrTy = PointerType::get(nextEltTy, as);
        ptrResLoc = new BitCastInst(curArgVal, scalarPtrTy, "", entryBlk);
      }
      ptrResLoc = GetElementPtrInst::CreateInBounds(ptrResLoc, Index, "ptrres2",
                                                    entryBlk);
      hiArgs.push_back(ptrResLoc);
    }
  }

  // process the lower part of the vector
  CallInst *callInst = CallInst::Create(nextFunc, loArgs, "lo.call", entryBlk);
  callInst->setCallingConv(nextFunc->getCallingConv());
  callInst->setAttributes(nextFunc->getAttributes());
  if (resVal) {
    resVal = insertVectorSlice(0, nextVecSize, callInst, resVal, entryBlk);
  }

  // process the higher part of the vector
  callInst = CallInst::Create(nextFuncHi, hiArgs, "hi.call", entryBlk);
  callInst->setCallingConv(nextFunc->getCallingConv());
  callInst->setAttributes(nextFunc->getAttributes());
  if (resVal) {
    resVal =
        insertVectorSlice(nextVecSize, curVecSize, callInst, resVal, entryBlk);
  }

  if (resVal) {
    ReturnInst::Create(theContext, resVal, entryBlk);
  }

  curFunc->setLinkage(Function::WeakAnyLinkage);

// verify that the function is well formed.
#if !defined(NDEBUG)
  verifyFunction(*curFunc);
#endif

  return nextFunc;
} // AMDGPUclpVectorExpansion::CheckAndExpand

Value *AMDGPUclpVectorExpansion::loadVectorSlice(int srcStartIdx,
                                                 int srcPassIdx, Value *srcVal,
                                                 BasicBlock *blk) {
  int numEle = srcPassIdx - srcStartIdx;
  assert(numEle >= 1);

  Type *srcTy = srcVal->getType();
  assert(isa<VectorType>(srcTy));

  Value *dstVal = nullptr;
  if (numEle == 1) {
    Value *srcIdx =
        ConstantInt::get(Type::getInt32Ty(blk->getContext()), srcStartIdx);
    dstVal = ExtractElementInst::Create(srcVal, srcIdx, "tmp", blk);
  } else {
    Type *eleTy = cast<VectorType>(srcTy)->getElementType();
    Type *dstTy = VectorType::get(eleTy, numEle);
    dstVal = UndefValue::get(dstTy);
    for (int i = srcStartIdx; i < srcPassIdx; ++i) {
      Value *srcIdx = ConstantInt::get(Type::getInt32Ty(blk->getContext()), i);
      Value *tmpVal = ExtractElementInst::Create(srcVal, srcIdx, "tmp", blk);
      Value *dstIdx = ConstantInt::get(Type::getInt32Ty(blk->getContext()),
                                       i - srcStartIdx);
      dstVal = InsertElementInst::Create(dstVal, tmpVal, dstIdx, "tmp", blk);
    }
  }

  return dstVal;
} // AMDGPUclpVectorExpansion::loadVectorSlice

Value *AMDGPUclpVectorExpansion::insertVectorSlice(int dstStartIdx,
                                                   int dstPassIdx,
                                                   Value *srcVal, Value *dstVal,
                                                   BasicBlock *blk) {
  int numSrcEle = dstPassIdx - dstStartIdx;
  for (int i = dstStartIdx; i < dstPassIdx; ++i) {
    Value *tmpVal = nullptr;

    if (numSrcEle == 1) {
      tmpVal = srcVal;
    } else {
      Value *srcIdx = ConstantInt::get(Type::getInt32Ty(blk->getContext()),
                                       i - dstStartIdx);
      tmpVal = ExtractElementInst::Create(srcVal, srcIdx, "tmp", blk);
    }
    Value *dstIdx = ConstantInt::get(Type::getInt32Ty(blk->getContext()), i);
    dstVal = InsertElementInst::Create(dstVal, tmpVal, dstIdx, "tmp", blk);
  }

  return dstVal;
} // AMDGPUclpVectorExpansion::insertVectorSlice

AMDGPUclpVectorExpansion::AMDGPUclpVectorExpansion()
    : ModulePass(ID), tmpModule(nullptr) {
  initializeAMDGPUclpVectorExpansionPass(*PassRegistry::getPassRegistry());
  addNeedExpansion(needExpansionBuiltins);
  addBuiltinInfo(mathFunc);
  addBuiltinInfo(integerFunc);
  addBuiltinInfo(commonFunc);
  addBuiltinInfo(relationalFunc);
  addBuiltinInfo(pragmaEnableFunc);

#if !defined(NDEBUG)
  checkExpansionInfo();
#endif
} // AMDGPUclpVectorExpansion::AMDGPUclpVectorExpansion

/// process an LLVM module to perform expand vector builtin calls
///
bool AMDGPUclpVectorExpansion::runOnModule(Module &theModule) {
  tmpModule.reset(
      new Module("__opencllib_vectorexpansion", theModule.getContext()));
  tmpModule->setDataLayout(theModule.getDataLayout());
  // loop through all Function to collect funcuseInfo
  for (auto &func : theModule) {
    // Function must be a prototype and is used.
    if (func.isDeclaration() && func.use_empty() == false) {
      checkAndAddToExpansion(&func);
    }
  }
  if (!tmpModule->empty()) {
    return !Linker::linkModules(theModule, std::move(tmpModule));
  }
  return false;
} // AMDGPUclpVectorExpansion::runOnModule

//===----------------------------------------------------------------------===//
// end-of-file newline
