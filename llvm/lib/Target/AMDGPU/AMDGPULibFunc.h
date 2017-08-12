//===-- AMDGPULibFunc.h ---------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef _AMDGPU_LIBFUNC_H_
#define _AMDGPU_LIBFUNC_H_

#include "llvm/ADT/StringRef.h"

namespace llvm {

class FunctionType;
class Function;
class Module;

class AMDGPULibFunc {
public:
  enum EFuncId {
    EI_NONE,

    // IMPORTANT: enums below should go in ascending by 1 value order
    // because they are used as indexes in the mangling rules table.
    // don't use explicit value assignment.
    EI_ABS,
    EI_ABS_DIFF,
    EI_ACOS,
    EI_ACOSH,
    EI_ACOSPI,
    EI_ADD_SAT,
    EI_ALL,
    EI_ANY,
    EI_ASIN,
    EI_ASINH,
    EI_ASINPI,
    EI_ASYNC_WORK_GROUP_COPY,
    EI_ASYNC_WORK_GROUP_STRIDED_COPY,
    EI_ATAN,
    EI_ATAN2,
    EI_ATAN2PI,
    EI_ATANH,
    EI_ATANPI,
    EI_ATOMIC_ADD,
    EI_ATOMIC_AND,
    EI_ATOMIC_CMPXCHG,
    EI_ATOMIC_DEC,
    EI_ATOMIC_INC,
    EI_ATOMIC_MAX,
    EI_ATOMIC_MIN,
    EI_ATOMIC_OR,
    EI_ATOMIC_SUB,
    EI_ATOMIC_XCHG,
    EI_ATOMIC_XOR,
    EI_BITSELECT,
    EI_CBRT,
    EI_CEIL,
    EI_CLAMP,
    EI_CLZ,
    EI_COMMIT_READ_PIPE,
    EI_COMMIT_WRITE_PIPE,
    EI_COPYSIGN,
    EI_COS,
    EI_COSH,
    EI_COSPI,
    EI_CROSS,
    EI_CTZ,
    EI_DEGREES,
    EI_DISTANCE,
    EI_DIVIDE,
    EI_DOT,
    EI_ERF,
    EI_ERFC,
    EI_EXP,
    EI_EXP10,
    EI_EXP2,
    EI_EXPM1,
    EI_FABS,
    EI_FAST_DISTANCE,
    EI_FAST_LENGTH,
    EI_FAST_NORMALIZE,
    EI_FDIM,
    EI_FLOOR,
    EI_FMA,
    EI_FMAX,
    EI_FMIN,
    EI_FMOD,
    EI_FRACT,
    EI_FREXP,
    EI_GET_IMAGE_ARRAY_SIZE,
    EI_GET_IMAGE_CHANNEL_DATA_TYPE,
    EI_GET_IMAGE_CHANNEL_ORDER,
    EI_GET_IMAGE_DIM,
    EI_GET_IMAGE_HEIGHT,
    EI_GET_IMAGE_WIDTH,
    EI_GET_PIPE_MAX_PACKETS,
    EI_GET_PIPE_NUM_PACKETS,
    EI_HADD,
    EI_HYPOT,
    EI_ILOGB,
    EI_ISEQUAL,
    EI_ISFINITE,
    EI_ISGREATER,
    EI_ISGREATEREQUAL,
    EI_ISINF,
    EI_ISLESS,
    EI_ISLESSEQUAL,
    EI_ISLESSGREATER,
    EI_ISNAN,
    EI_ISNORMAL,
    EI_ISNOTEQUAL,
    EI_ISORDERED,
    EI_ISUNORDERED,
    EI_LDEXP,
    EI_LENGTH,
    EI_LGAMMA,
    EI_LGAMMA_R,
    EI_LOG,
    EI_LOG10,
    EI_LOG1P,
    EI_LOG2,
    EI_LOGB,
    EI_MAD,
    EI_MAD24,
    EI_MAD_HI,
    EI_MAD_SAT,
    EI_MAX,
    EI_MAXMAG,
    EI_MIN,
    EI_MINMAG,
    EI_MIX,
    EI_MODF,
    EI_MUL24,
    EI_MUL_HI,
    EI_NAN,
    EI_NEXTAFTER,
    EI_NORMALIZE,
    EI_POPCOUNT,
    EI_POW,
    EI_POWN,
    EI_POWR,
    EI_PREFETCH,
    EI_RADIANS,
    EI_READ_PIPE,
    EI_RECIP,
    EI_REMAINDER,
    EI_REMQUO,
    EI_RESERVE_READ_PIPE,
    EI_RESERVE_WRITE_PIPE,
    EI_RHADD,
    EI_RINT,
    EI_ROOTN,
    EI_ROTATE,
    EI_ROUND,
    EI_RSQRT,
    EI_SELECT,
    EI_SHUFFLE,
    EI_SHUFFLE2,
    EI_SIGN,
    EI_SIGNBIT,
    EI_SIN,
    EI_SINCOS,
    EI_SINH,
    EI_SINPI,
    EI_SMOOTHSTEP,
    EI_SQRT,
    EI_STEP,
    EI_SUB_GROUP_BROADCAST,
    EI_SUB_GROUP_COMMIT_READ_PIPE,
    EI_SUB_GROUP_COMMIT_WRITE_PIPE,
    EI_SUB_GROUP_REDUCE_ADD,
    EI_SUB_GROUP_REDUCE_MAX,
    EI_SUB_GROUP_REDUCE_MIN,
    EI_SUB_GROUP_RESERVE_READ_PIPE,
    EI_SUB_GROUP_RESERVE_WRITE_PIPE,
    EI_SUB_GROUP_SCAN_EXCLUSIVE_ADD,
    EI_SUB_GROUP_SCAN_EXCLUSIVE_MAX,
    EI_SUB_GROUP_SCAN_EXCLUSIVE_MIN,
    EI_SUB_GROUP_SCAN_INCLUSIVE_ADD,
    EI_SUB_GROUP_SCAN_INCLUSIVE_MAX,
    EI_SUB_GROUP_SCAN_INCLUSIVE_MIN,
    EI_SUB_SAT,
    EI_TAN,
    EI_TANH,
    EI_TANPI,
    EI_TGAMMA,
    EI_TRUNC,
    EI_UPSAMPLE,
    EI_VEC_STEP,
    EI_VSTORE,
    EI_VSTORE16,
    EI_VSTORE2,
    EI_VSTORE3,
    EI_VSTORE4,
    EI_VSTORE8,
    EI_WORK_GROUP_COMMIT_READ_PIPE,
    EI_WORK_GROUP_COMMIT_WRITE_PIPE,
    EI_WORK_GROUP_REDUCE_ADD,
    EI_WORK_GROUP_REDUCE_MAX,
    EI_WORK_GROUP_REDUCE_MIN,
    EI_WORK_GROUP_RESERVE_READ_PIPE,
    EI_WORK_GROUP_RESERVE_WRITE_PIPE,
    EI_WORK_GROUP_SCAN_EXCLUSIVE_ADD,
    EI_WORK_GROUP_SCAN_EXCLUSIVE_MAX,
    EI_WORK_GROUP_SCAN_EXCLUSIVE_MIN,
    EI_WORK_GROUP_SCAN_INCLUSIVE_ADD,
    EI_WORK_GROUP_SCAN_INCLUSIVE_MAX,
    EI_WORK_GROUP_SCAN_INCLUSIVE_MIN,
    EI_WRITE_IMAGEF,
    EI_WRITE_IMAGEI,
    EI_WRITE_IMAGEUI,
    EI_WRITE_PIPE,
    EI_NCOS,
    EI_NEXP2,
    EI_NFMA,
    EI_NLOG2,
    EI_NRCP,
    EI_NRSQRT,
    EI_NSIN,
    EI_NSQRT,
    EI_FTZ,
    EI_FLDEXP,
    EI_CLASS,
    EI_RCBRT,

    EX_INTRINSICS_COUNT
  };

  enum ENamePrefix {
    NOPFX,
    NATIVE,
    HALF
  };

  enum EType {
    B8  = 1,
    B16 = 2,
    B32 = 3,
    B64 = 4,
    SIZE_MASK = 7,
    FLOAT = 0x10,
    INT   = 0x20,
    UINT  = 0x30,
    BASE_TYPE_MASK = 0x30,
    U8  =  UINT | B8,
    U16 =  UINT | B16,
    U32 =  UINT | B32,
    U64 =  UINT | B64,
    I8  =   INT | B8,
    I16 =   INT | B16,
    I32 =   INT | B32,
    I64 =   INT | B64,
    F16 = FLOAT | B16,
    F32 = FLOAT | B32,
    F64 = FLOAT | B64,
    IMG1DA = 0x80,
    IMG1DB,
    IMG2DA,
    IMG1D,
    IMG2D,
    IMG3D,
    SAMPLER,
    EVENT,
    DUMMY
  };

  enum EPtrKind {
    BYVALUE = 0,
    PRIVATE,
    GLOBAL,
    READONLY,
    LOCAL,
    GENERIC,
    OTHER,

    ADDR_SPACE = 0xF,
    CONST      = 0x10,
    VOLATILE   = 0x20
  };

  struct Param {
    unsigned char ArgType;
    unsigned char VectorSize;
    unsigned char PtrKind;

    unsigned char Reserved;

    void reset() {
      ArgType = 0;
      VectorSize = 1;
      PtrKind = 0;
    }
    Param() { reset(); }

    template <typename Stream>
    void mangleItanium(Stream& os);
  };

public:
  static bool      parse(StringRef mangledName, AMDGPULibFunc &iInfo);

  AMDGPULibFunc();
  AMDGPULibFunc(EFuncId id, const AMDGPULibFunc& copyFrom);

  ENamePrefix   getPrefix() const { return FKind; }
  EFuncId  getId() const { return FuncId; }

  std::string   getName() const;
  unsigned      getNumArgs() const;

  FunctionType* getFunctionType(Module& M) const;

  std::string   mangle() const;

  void setPrefix(ENamePrefix pfx) { FKind = pfx; }
  void setId(EFuncId id) { FuncId = id; }

  static Function* getFunction(llvm::Module *M, const AMDGPULibFunc& fInfo);

  static Function* getOrInsertFunction(llvm::Module *M,
                                       const AMDGPULibFunc& fInfo);

  static StringRef getUnmangledName(const StringRef& mangledName);

  Param         Leads[2];

private:
  EFuncId       FuncId;
  ENamePrefix   FKind;
  std::string   Name;

  void          reset();

  std::string   mangleNameItanium() const;
  bool          parseItanuimName(StringRef& mangledName);

  std::string   mangleName(const StringRef& name) const;
  bool          parseName(const StringRef& mangledName);

  template <typename Stream>
  void          writeName(Stream& OS) const;
};

}
#endif // _AMDGPU_LIBFUNC_H_
