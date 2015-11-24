/*===-- InstrProfData.inc - instr profiling runtime structures -----------=== *\
|*
|*                     The LLVM Compiler Infrastructure
|*
|* This file is distributed under the University of Illinois Open Source
|* License. See LICENSE.TXT for details.
|*
\*===----------------------------------------------------------------------===*/
/*
 * This is the master file that defines all the data structure, signature,
 * constant literals that are shared across profiling runtime library,
 * compiler (instrumentation), and host tools (reader/writer). The entities
 * defined in this file affect the profile runtime ABI, the raw profile format,
 * or both.
 *
 * The file has two identical copies. The master copy lives in LLVM and
 * the other one  sits in compiler-rt/lib/profile directory. To make changes
 * in this file, first modify the master copy and copy it over to compiler-rt.
 * Testing of any change in this file can start only after the two copies are
 * synced up.
 *
 * The first part of the file includes macros that defines types, names, and
 * initializers for the member fields of the core data structures. The field
 * declarations for one structure is enabled by defining the field activation
 * macro associated with that structure. Only one field activation record
 * can be defined at one time and the rest definitions will be filtered out by
 * the preprocessor.
 *
 * Examples of how the template is used to instantiate structure definition:
 * 1. To declare a structure:
 * 
 * struct ProfData {
 * #define INSTR_PROF_DATA(Type, LLVMType, Name, Initializer) \
 *    Type Name;
 * #include "llvm/ProfileData/InstrProfData.inc"
 * };
 *
 * 2. To construct LLVM type arrays for the struct type:
 *
 * Type *DataTypes[] = {
 * #define INSTR_PROF_DATA(Type, LLVMType, Name, Initializer) \
 *   LLVMType,
 * #include "llvm/ProfileData/InstrProfData.inc"
 * };
 *
 * 4. To construct constant array for the initializers:
 * #define INSTR_PROF_DATA(Type, LLVMType, Name, Initializer) \
 *   Initializer,
 * Constant *ConstantVals[] = {
 * #include "llvm/ProfileData/InstrProfData.inc"
 * };
 *
 *
 * The second part of the file includes definitions all other entities that
 * are related to runtime ABI and format. When no field activation macro is
 * defined, this file can be included to introduce the definitions.
 *
\*===----------------------------------------------------------------------===*/

/* INSTR_PROF_DATA start. */
/* Definition of member fields of the per-function control structure. */
#ifndef INSTR_PROF_DATA
#define INSTR_PROF_DATA(Type, LLVMType, Name, Initializer)
#else
#define INSTR_PROF_DATA_DEFINED
#endif

INSTR_PROF_DATA(const uint32_t, llvm::Type::getInt32Ty(Ctx), NameSize, \
                ConstantInt::get(llvm::Type::getInt32Ty(Ctx), \
                NamePtr->getType()->getPointerElementType()->getArrayNumElements()))
INSTR_PROF_DATA(const uint32_t, llvm::Type::getInt32Ty(Ctx), NumCounters, \
                ConstantInt::get(llvm::Type::getInt32Ty(Ctx), NumCounters))
INSTR_PROF_DATA(const uint64_t, llvm::Type::getInt64Ty(Ctx), FuncHash, \
                ConstantInt::get(llvm::Type::getInt64Ty(Ctx), \
                Inc->getHash()->getZExtValue()))
INSTR_PROF_DATA(const IntPtrT, llvm::Type::getInt8PtrTy(Ctx), NamePtr, \
                ConstantExpr::getBitCast(NamePtr, llvm::Type::getInt8PtrTy(Ctx)))
INSTR_PROF_DATA(const IntPtrT, llvm::Type::getInt64PtrTy(Ctx), CounterPtr, \
                ConstantExpr::getBitCast(CounterPtr, \
                llvm::Type::getInt64PtrTy(Ctx)))
INSTR_PROF_DATA(const IntPtrT, llvm::Type::getInt8PtrTy(Ctx), FunctionPointer, \
                FunctionAddr)
INSTR_PROF_DATA(IntPtrT, llvm::Type::getInt8PtrTy(Ctx), Values, \
                ConstantPointerNull::get(Int8PtrTy))
INSTR_PROF_DATA(const uint16_t, Int16ArrayTy, NumValueSites[IPVK_Last+1], \
                ConstantArray::get(Int16ArrayTy, Int16ArrayVals))
#undef INSTR_PROF_DATA
/* INSTR_PROF_DATA end. */

/* INSTR_PROF_RAW_HEADER  start */
/* Definition of member fields of the raw profile header data structure. */
#ifndef INSTR_PROF_RAW_HEADER
#define INSTR_PROF_RAW_HEADER(Type, Name, Initializer)
#else
#define INSTR_PROF_DATA_DEFINED
#endif
INSTR_PROF_RAW_HEADER(uint64_t, Magic, __llvm_profile_get_magic())
INSTR_PROF_RAW_HEADER(uint64_t, Version, __llvm_profile_get_version())
INSTR_PROF_RAW_HEADER(uint64_t, DataSize, DataSize)
INSTR_PROF_RAW_HEADER(uint64_t, CountersSize, CountersSize)
INSTR_PROF_RAW_HEADER(uint64_t, NamesSize,  NamesSize)
INSTR_PROF_RAW_HEADER(uint64_t, CountersDelta, (uintptr_t)CountersBegin)
INSTR_PROF_RAW_HEADER(int64_t, NamesDelta, (uintptr_t)NamesBegin)
INSTR_PROF_RAW_HEADER(uint64_t, ValueKindLast, IPVK_Last)
INSTR_PROF_RAW_HEADER(uint64_t, ValueDataSize, ValueDataSize)
INSTR_PROF_RAW_HEADER(uint64_t, ValueDataDelta, (uintptr_t)ValueDataBegin)
#undef INSTR_PROF_RAW_HEADER
/* INSTR_PROF_RAW_HEADER  end */

/* VALUE_PROF_FUNC_PARAM start */
/* Definition of parameter types of the runtime API used to do value profiling
 * for a given value site.
 */
#ifndef VALUE_PROF_FUNC_PARAM
#define VALUE_PROF_FUNC_PARAM(ArgType, ArgName, ArgLLVMType)
#define INSTR_PROF_COMMA
#else
#define INSTR_PROF_DATA_DEFINED
#define INSTR_PROF_COMMA ,
#endif
VALUE_PROF_FUNC_PARAM(uint64_t, TargetValue, Type::getInt64Ty(Ctx)) \
                      INSTR_PROF_COMMA
VALUE_PROF_FUNC_PARAM(void *, Data, Type::getInt8PtrTy(Ctx)) INSTR_PROF_COMMA
VALUE_PROF_FUNC_PARAM(uint32_t, CounterIndex, Type::getInt32Ty(Ctx))
#undef VALUE_PROF_FUNC_PARAM
#undef INSTR_PROF_COMMA
/* VALUE_PROF_FUNC_PARAM end */

/* VALUE_PROF_KIND start */
#ifndef VALUE_PROF_KIND
#define VALUE_PROF_KIND(Enumerator, Value)
#else
#define INSTR_PROF_DATA_DEFINED
#endif
VALUE_PROF_KIND(IPVK_IndirectCallTarget, 0)
/* These two kinds must be the last to be
 * declared. This is to make sure the string
 * array created with the template can be
 * indexed with the kind value.
 */
VALUE_PROF_KIND(IPVK_First, IPVK_IndirectCallTarget)
VALUE_PROF_KIND(IPVK_Last, IPVK_IndirectCallTarget)

#undef VALUE_PROF_KIND
/* VALUE_PROF_KIND end */

/* COVMAP_FUNC_RECORD start */
/* Definition of member fields of the function record structure in coverage
 * map.
 */
#ifndef COVMAP_FUNC_RECORD
#define COVMAP_FUNC_RECORD(Type, LLVMType, Name, Initializer)
#else
#define INSTR_PROF_DATA_DEFINED
#endif
COVMAP_FUNC_RECORD(const IntPtrT, llvm::Type::getInt8PtrTy(Ctx), \
                   NamePtr, llvm::ConstantExpr::getBitCast(NamePtr, \
                   llvm::Type::getInt8PtrTy(Ctx))) 
COVMAP_FUNC_RECORD(const uint32_t, llvm::Type::getInt32Ty(Ctx), NameSize, \
                   llvm::ConstantInt::get(llvm::Type::getInt32Ty(Ctx),\
                   NameValue.size()))
COVMAP_FUNC_RECORD(const uint32_t, llvm::Type::getInt32Ty(Ctx), DataSize, \
                   llvm::ConstantInt::get(llvm::Type::getInt32Ty(Ctx),\
                   CoverageMapping.size()))
COVMAP_FUNC_RECORD(const uint64_t, llvm::Type::getInt64Ty(Ctx), FuncHash, \
                   llvm::ConstantInt::get(llvm::Type::getInt64Ty(Ctx), FuncHash))
#undef COVMAP_FUNC_RECORD
/* COVMAP_FUNC_RECORD end.  */



/*============================================================================*/


#ifndef INSTR_PROF_DATA_DEFINED

#ifndef INSTR_PROF_DATA_INC_
#define INSTR_PROF_DATA_INC_

/* Helper macros.  */
#define INSTR_PROF_SIMPLE_QUOTE(x) #x
#define INSTR_PROF_QUOTE(x) INSTR_PROF_SIMPLE_QUOTE(x)
#define INSTR_PROF_SIMPLE_CONCAT(x,y) x ## y
#define INSTR_PROF_CONCAT(x,y) INSTR_PROF_SIMPLE_CONCAT(x,y)

/* Magic number to detect file format and endianness.
 * Use 255 at one end, since no UTF-8 file can use that character.  Avoid 0,
 * so that utilities, like strings, don't grab it as a string.  129 is also
 * invalid UTF-8, and high enough to be interesting.
 * Use "lprofr" in the centre to stand for "LLVM Profile Raw", or "lprofR"
 * for 32-bit platforms.
 */
#define INSTR_PROF_RAW_MAGIC_64 (uint64_t)255 << 56 | (uint64_t)'l' << 48 | \
       (uint64_t)'p' << 40 | (uint64_t)'r' << 32 | (uint64_t)'o' << 24 |  \
        (uint64_t)'f' << 16 | (uint64_t)'r' << 8 | (uint64_t)129
#define INSTR_PROF_RAW_MAGIC_32 (uint64_t)255 << 56 | (uint64_t)'l' << 48 | \
       (uint64_t)'p' << 40 | (uint64_t)'r' << 32 | (uint64_t)'o' << 24 |  \
        (uint64_t)'f' << 16 | (uint64_t)'R' << 8 | (uint64_t)129

/* Raw profile format version. */
#define INSTR_PROF_RAW_VERSION 2

/* Runtime section names and name strings.  */
#define INSTR_PROF_DATA_SECT_NAME __llvm_prf_data
#define INSTR_PROF_NAME_SECT_NAME __llvm_prf_names
#define INSTR_PROF_CNTS_SECT_NAME __llvm_prf_cnts

#define INSTR_PROF_DATA_SECT_NAME_STR \
        INSTR_PROF_QUOTE(INSTR_PROF_DATA_SECT_NAME)
#define INSTR_PROF_NAME_SECT_NAME_STR \
        INSTR_PROF_QUOTE(INSTR_PROF_NAME_SECT_NAME)
#define INSTR_PROF_CNTS_SECT_NAME_STR \
        INSTR_PROF_QUOTE(INSTR_PROF_CNTS_SECT_NAME)

/* Macros to define start/stop section symbol for a given
 * section on Linux. For instance
 * INSTR_PROF_SECT_START(INSTR_PROF_DATA_SECT_NAME) will
 * expand to __start___llvm_prof_data
 */
#define INSTR_PROF_SECT_START(Sect) \
        INSTR_PROF_CONCAT(__start_,Sect)
#define INSTR_PROF_SECT_STOP(Sect) \
        INSTR_PROF_CONCAT(__stop_,Sect)

/* Value Profiling API linkage name.  */
#define INSTR_PROF_VALUE_PROF_FUNC __llvm_profile_instrument_target
#define INSTR_PROF_VALUE_PROF_FUNC_STR \
        INSTR_PROF_QUOTE(INSTR_PROF_VALUE_PROF_FUNC)

/* InstrProfile per-function control data alignment.  */
#define INSTR_PROF_DATA_ALIGNMENT 8

/* The data structure that represents a tracked value by the
 * value profiler.
 */
typedef struct InstrProfValueData {
  /* Profiled value. */
  uint64_t Value;
  /* Number of times the value appears in the training run. */
  uint64_t Count;
} InstrProfValueData;

/* This is an internal data structure used by value profiler. It
 * is defined here to allow serialization code sharing by LLVM
 * to be used in unit test.
 */
typedef struct ValueProfNode {
  InstrProfValueData VData;
  struct ValueProfNode *Next;
} ValueProfNode;

#endif /* INSTR_PROF_DATA_INC_ */

#else
#undef INSTR_PROF_DATA_DEFINED
#endif

