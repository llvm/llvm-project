//===-- X86InstrFMA3Info.cpp - X86 FMA3 Instruction Information -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the classes providing information
// about existing X86 FMA3 opcodes, classifying and grouping them.
//
//===----------------------------------------------------------------------===//

#include "X86InstrFMA3Info.h"
#include "X86InstrInfo.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Threading.h"
#include <cassert>
#include <cstdint>

using namespace llvm;

/// This flag is used in the method llvm::call_once() used below to make the
/// initialization of the map 'OpcodeToGroup' thread safe.
static llvm::once_flag InitGroupsOnceFlag;

static ManagedStatic<X86InstrFMA3Info> X86InstrFMA3InfoObj;
X86InstrFMA3Info *X86InstrFMA3Info::getX86InstrFMA3Info() {
  return &*X86InstrFMA3InfoObj;
}

void X86InstrFMA3Info::initRMGroup(const uint16_t *RegOpcodes,
                                   const uint16_t *MemOpcodes, unsigned Attr) {
  // Create a new instance of this class that would hold a group of FMA opcodes.
  X86InstrFMA3Group *G = new X86InstrFMA3Group(RegOpcodes, MemOpcodes, Attr);

  // Add the references from indvidual opcodes to the group holding them.
  assert((!OpcodeToGroup[RegOpcodes[0]] && !OpcodeToGroup[RegOpcodes[1]] &&
          !OpcodeToGroup[RegOpcodes[2]] && !OpcodeToGroup[MemOpcodes[0]] &&
          !OpcodeToGroup[MemOpcodes[1]] && !OpcodeToGroup[MemOpcodes[2]]) &&
         "Duplication or rewrite of elements in OpcodeToGroup.");
  OpcodeToGroup[RegOpcodes[0]] = G;
  OpcodeToGroup[RegOpcodes[1]] = G;
  OpcodeToGroup[RegOpcodes[2]] = G;
  OpcodeToGroup[MemOpcodes[0]] = G;
  OpcodeToGroup[MemOpcodes[1]] = G;
  OpcodeToGroup[MemOpcodes[2]] = G;
}

void X86InstrFMA3Info::initRGroup(const uint16_t *RegOpcodes, unsigned Attr) {
  // Create a new instance of this class that would hold a group of FMA opcodes.
  X86InstrFMA3Group *G = new X86InstrFMA3Group(RegOpcodes, nullptr, Attr);

  // Add the references from indvidual opcodes to the group holding them.
  assert((!OpcodeToGroup[RegOpcodes[0]] && !OpcodeToGroup[RegOpcodes[1]] &&
          !OpcodeToGroup[RegOpcodes[2]]) &&
         "Duplication or rewrite of elements in OpcodeToGroup.");
  OpcodeToGroup[RegOpcodes[0]] = G;
  OpcodeToGroup[RegOpcodes[1]] = G;
  OpcodeToGroup[RegOpcodes[2]] = G;
}

void X86InstrFMA3Info::initMGroup(const uint16_t *MemOpcodes, unsigned Attr) {
  // Create a new instance of this class that would hold a group of FMA opcodes.
  X86InstrFMA3Group *G = new X86InstrFMA3Group(nullptr, MemOpcodes, Attr);

  // Add the references from indvidual opcodes to the group holding them.
  assert((!OpcodeToGroup[MemOpcodes[0]] && !OpcodeToGroup[MemOpcodes[1]] &&
          !OpcodeToGroup[MemOpcodes[2]]) &&
         "Duplication or rewrite of elements in OpcodeToGroup.");
  OpcodeToGroup[MemOpcodes[0]] = G;
  OpcodeToGroup[MemOpcodes[1]] = G;
  OpcodeToGroup[MemOpcodes[2]] = G;
}

#define FMA3RM(R132, R213, R231, M132, M213, M231)                             \
  static const uint16_t Reg##R132[3] = {X86::R132, X86::R213, X86::R231};      \
  static const uint16_t Mem##R132[3] = {X86::M132, X86::M213, X86::M231};      \
  initRMGroup(Reg##R132, Mem##R132);

#define FMA3RMA(R132, R213, R231, M132, M213, M231, Attrs)                     \
  static const uint16_t Reg##R132[3] = {X86::R132, X86::R213, X86::R231};      \
  static const uint16_t Mem##R132[3] = {X86::M132, X86::M213, X86::M231};      \
  initRMGroup(Reg##R132, Mem##R132, (Attrs));

#define FMA3R(R132, R213, R231)                                                \
  static const uint16_t Reg##R132[3] = {X86::R132, X86::R213, X86::R231};      \
  initRGroup(Reg##R132);

#define FMA3RA(R132, R213, R231, Attrs)                                        \
  static const uint16_t Reg##R132[3] = {X86::R132, X86::R213, X86::R231};      \
  initRGroup(Reg##R132, (Attrs));

#define FMA3M(M132, M213, M231)                                                \
  static const uint16_t Mem##M132[3] = {X86::M132, X86::M213, X86::M231};      \
  initMGroup(Mem##M132);

#define FMA3MA(M132, M213, M231, Attrs)                                        \
  static const uint16_t Mem##M132[3] = {X86::M132, X86::M213, X86::M231};      \
  initMGroup(Mem##M132, (Attrs));

#define FMA3_AVX2_VECTOR_GROUP(Name)                                           \
  FMA3RM(Name##132PSr, Name##213PSr, Name##231PSr,                             \
         Name##132PSm, Name##213PSm, Name##231PSm);                            \
  FMA3RM(Name##132PDr, Name##213PDr, Name##231PDr,                             \
         Name##132PDm, Name##213PDm, Name##231PDm);                            \
  FMA3RM(Name##132PSYr, Name##213PSYr, Name##231PSYr,                          \
         Name##132PSYm, Name##213PSYm, Name##231PSYm);                         \
  FMA3RM(Name##132PDYr, Name##213PDYr, Name##231PDYr,                          \
         Name##132PDYm, Name##213PDYm, Name##231PDYm);

#define FMA3_AVX2_SCALAR_GROUP(Name)                                           \
  FMA3RM(Name##132SSr, Name##213SSr, Name##231SSr,                             \
         Name##132SSm, Name##213SSm, Name##231SSm);                            \
  FMA3RM(Name##132SDr, Name##213SDr, Name##231SDr,                             \
         Name##132SDm, Name##213SDm, Name##231SDm);                            \
  FMA3RMA(Name##132SSr_Int, Name##213SSr_Int, Name##231SSr_Int,                \
          Name##132SSm_Int, Name##213SSm_Int, Name##231SSm_Int,                \
          X86InstrFMA3Group::X86FMA3Intrinsic);                                \
  FMA3RMA(Name##132SDr_Int, Name##213SDr_Int, Name##231SDr_Int,                \
          Name##132SDm_Int, Name##213SDm_Int, Name##231SDm_Int,                \
          X86InstrFMA3Group::X86FMA3Intrinsic);

#define FMA3_AVX2_FULL_GROUP(Name)                                             \
  FMA3_AVX2_VECTOR_GROUP(Name);                                                \
  FMA3_AVX2_SCALAR_GROUP(Name);

#define FMA3_AVX512_VECTOR_GROUP(Name)                                         \
  FMA3RM(Name##132PSZ128r, Name##213PSZ128r, Name##231PSZ128r,                 \
         Name##132PSZ128m, Name##213PSZ128m, Name##231PSZ128m);                \
  FMA3RM(Name##132PDZ128r, Name##213PDZ128r, Name##231PDZ128r,                 \
         Name##132PDZ128m, Name##213PDZ128m, Name##231PDZ128m);                \
  FMA3RM(Name##132PSZ256r, Name##213PSZ256r, Name##231PSZ256r,                 \
         Name##132PSZ256m, Name##213PSZ256m, Name##231PSZ256m);                \
  FMA3RM(Name##132PDZ256r, Name##213PDZ256r, Name##231PDZ256r,                 \
         Name##132PDZ256m, Name##213PDZ256m, Name##231PDZ256m);                \
  FMA3RM(Name##132PSZr,    Name##213PSZr,    Name##231PSZr,                    \
         Name##132PSZm,    Name##213PSZm,    Name##231PSZm);                   \
  FMA3RM(Name##132PDZr,    Name##213PDZr,    Name##231PDZr,                    \
         Name##132PDZm,    Name##213PDZm,    Name##231PDZm);                   \
  FMA3RMA(Name##132PSZ128rk, Name##213PSZ128rk, Name##231PSZ128rk,             \
          Name##132PSZ128mk, Name##213PSZ128mk, Name##231PSZ128mk,             \
          X86InstrFMA3Group::X86FMA3KMergeMasked);                             \
  FMA3RMA(Name##132PDZ128rk, Name##213PDZ128rk, Name##231PDZ128rk,             \
          Name##132PDZ128mk, Name##213PDZ128mk, Name##231PDZ128mk,             \
          X86InstrFMA3Group::X86FMA3KMergeMasked);                             \
  FMA3RMA(Name##132PSZ256rk, Name##213PSZ256rk, Name##231PSZ256rk,             \
          Name##132PSZ256mk, Name##213PSZ256mk, Name##231PSZ256mk,             \
          X86InstrFMA3Group::X86FMA3KMergeMasked);                             \
  FMA3RMA(Name##132PDZ256rk, Name##213PDZ256rk, Name##231PDZ256rk,             \
          Name##132PDZ256mk, Name##213PDZ256mk, Name##231PDZ256mk,             \
          X86InstrFMA3Group::X86FMA3KMergeMasked);                             \
  FMA3RMA(Name##132PSZrk,    Name##213PSZrk,    Name##231PSZrk,                \
          Name##132PSZmk,    Name##213PSZmk,    Name##231PSZmk,                \
          X86InstrFMA3Group::X86FMA3KMergeMasked);                             \
  FMA3RMA(Name##132PDZrk,    Name##213PDZrk,    Name##231PDZrk,                \
          Name##132PDZmk,    Name##213PDZmk,    Name##231PDZmk,                \
          X86InstrFMA3Group::X86FMA3KMergeMasked);                             \
  FMA3RMA(Name##132PSZ128rkz, Name##213PSZ128rkz, Name##231PSZ128rkz,          \
          Name##132PSZ128mkz, Name##213PSZ128mkz, Name##231PSZ128mkz,          \
          X86InstrFMA3Group::X86FMA3KZeroMasked);                              \
  FMA3RMA(Name##132PDZ128rkz, Name##213PDZ128rkz, Name##231PDZ128rkz,          \
          Name##132PDZ128mkz, Name##213PDZ128mkz, Name##231PDZ128mkz,          \
          X86InstrFMA3Group::X86FMA3KZeroMasked);                              \
  FMA3RMA(Name##132PSZ256rkz, Name##213PSZ256rkz, Name##231PSZ256rkz,          \
          Name##132PSZ256mkz, Name##213PSZ256mkz, Name##231PSZ256mkz,          \
          X86InstrFMA3Group::X86FMA3KZeroMasked);                              \
  FMA3RMA(Name##132PDZ256rkz, Name##213PDZ256rkz, Name##231PDZ256rkz,          \
          Name##132PDZ256mkz, Name##213PDZ256mkz, Name##231PDZ256mkz,          \
          X86InstrFMA3Group::X86FMA3KZeroMasked);                              \
  FMA3RMA(Name##132PSZrkz,    Name##213PSZrkz,    Name##231PSZrkz,             \
          Name##132PSZmkz,    Name##213PSZmkz,    Name##231PSZmkz,             \
          X86InstrFMA3Group::X86FMA3KZeroMasked);                              \
  FMA3RMA(Name##132PDZrkz,    Name##213PDZrkz,    Name##231PDZrkz,             \
          Name##132PDZmkz,    Name##213PDZmkz,    Name##231PDZmkz,             \
          X86InstrFMA3Group::X86FMA3KZeroMasked);                              \
  FMA3R(Name##132PSZrb, Name##213PSZrb, Name##231PSZrb);                       \
  FMA3R(Name##132PDZrb, Name##213PDZrb, Name##231PDZrb);                       \
  FMA3RA(Name##132PSZrbk, Name##213PSZrbk, Name##231PSZrbk,                    \
         X86InstrFMA3Group::X86FMA3KMergeMasked);                              \
  FMA3RA(Name##132PDZrbk, Name##213PDZrbk, Name##231PDZrbk,                    \
         X86InstrFMA3Group::X86FMA3KMergeMasked);                              \
  FMA3RA(Name##132PSZrbkz, Name##213PSZrbkz, Name##231PSZrbkz,                 \
         X86InstrFMA3Group::X86FMA3KZeroMasked);                               \
  FMA3RA(Name##132PDZrbkz, Name##213PDZrbkz, Name##231PDZrbkz,                 \
         X86InstrFMA3Group::X86FMA3KZeroMasked);                               \
  FMA3M(Name##132PSZ128mb, Name##213PSZ128mb, Name##231PSZ128mb);              \
  FMA3M(Name##132PDZ128mb, Name##213PDZ128mb, Name##231PDZ128mb);              \
  FMA3M(Name##132PSZ256mb, Name##213PSZ256mb, Name##231PSZ256mb);              \
  FMA3M(Name##132PDZ256mb, Name##213PDZ256mb, Name##231PDZ256mb);              \
  FMA3M(Name##132PSZmb, Name##213PSZmb, Name##231PSZmb);                       \
  FMA3M(Name##132PDZmb, Name##213PDZmb, Name##231PDZmb);                       \
  FMA3MA(Name##132PSZ128mbk, Name##213PSZ128mbk, Name##231PSZ128mbk,           \
         X86InstrFMA3Group::X86FMA3KMergeMasked);                              \
  FMA3MA(Name##132PDZ128mbk, Name##213PDZ128mbk, Name##231PDZ128mbk,           \
         X86InstrFMA3Group::X86FMA3KMergeMasked);                              \
  FMA3MA(Name##132PSZ256mbk, Name##213PSZ256mbk, Name##231PSZ256mbk,           \
         X86InstrFMA3Group::X86FMA3KMergeMasked);                              \
  FMA3MA(Name##132PDZ256mbk, Name##213PDZ256mbk, Name##231PDZ256mbk,           \
         X86InstrFMA3Group::X86FMA3KMergeMasked);                              \
  FMA3MA(Name##132PSZmbk,    Name##213PSZmbk,    Name##231PSZmbk,              \
         X86InstrFMA3Group::X86FMA3KMergeMasked);                              \
  FMA3MA(Name##132PDZmbk,    Name##213PDZmbk,    Name##231PDZmbk,              \
         X86InstrFMA3Group::X86FMA3KMergeMasked);                              \
  FMA3MA(Name##132PSZ128mbkz, Name##213PSZ128mbkz, Name##231PSZ128mbkz,        \
         X86InstrFMA3Group::X86FMA3KZeroMasked);                               \
  FMA3MA(Name##132PDZ128mbkz, Name##213PDZ128mbkz, Name##231PDZ128mbkz,        \
         X86InstrFMA3Group::X86FMA3KZeroMasked);                               \
  FMA3MA(Name##132PSZ256mbkz, Name##213PSZ256mbkz, Name##231PSZ256mbkz,        \
         X86InstrFMA3Group::X86FMA3KZeroMasked);                               \
  FMA3MA(Name##132PDZ256mbkz, Name##213PDZ256mbkz, Name##231PDZ256mbkz,        \
         X86InstrFMA3Group::X86FMA3KZeroMasked);                               \
  FMA3MA(Name##132PSZmbkz, Name##213PSZmbkz, Name##231PSZmbkz,                 \
         X86InstrFMA3Group::X86FMA3KZeroMasked);                               \
  FMA3MA(Name##132PDZmbkz, Name##213PDZmbkz, Name##231PDZmbkz,                 \
         X86InstrFMA3Group::X86FMA3KZeroMasked);

#define FMA3_AVX512_SCALAR_GROUP(Name)                                         \
  FMA3RM(Name##132SSZr,      Name##213SSZr,     Name##231SSZr,                 \
         Name##132SSZm,      Name##213SSZm,     Name##231SSZm);                \
  FMA3RM(Name##132SDZr,      Name##213SDZr,     Name##231SDZr,                 \
         Name##132SDZm,      Name##213SDZm,     Name##231SDZm);                \
  FMA3RMA(Name##132SSZr_Int, Name##213SSZr_Int, Name##231SSZr_Int,             \
          Name##132SSZm_Int, Name##213SSZm_Int, Name##231SSZm_Int,             \
          X86InstrFMA3Group::X86FMA3Intrinsic);                                \
  FMA3RMA(Name##132SDZr_Int, Name##213SDZr_Int, Name##231SDZr_Int,             \
          Name##132SDZm_Int, Name##213SDZm_Int, Name##231SDZm_Int,             \
          X86InstrFMA3Group::X86FMA3Intrinsic);                                \
  FMA3RMA(Name##132SSZr_Intk, Name##213SSZr_Intk, Name##231SSZr_Intk,          \
          Name##132SSZm_Intk, Name##213SSZm_Intk, Name##231SSZm_Intk,          \
          X86InstrFMA3Group::X86FMA3Intrinsic |                                \
              X86InstrFMA3Group::X86FMA3KMergeMasked);                         \
  FMA3RMA(Name##132SDZr_Intk, Name##213SDZr_Intk, Name##231SDZr_Intk,          \
          Name##132SDZm_Intk, Name##213SDZm_Intk, Name##231SDZm_Intk,          \
          X86InstrFMA3Group::X86FMA3Intrinsic |                                \
              X86InstrFMA3Group::X86FMA3KMergeMasked);                         \
  FMA3RMA(Name##132SSZr_Intkz, Name##213SSZr_Intkz, Name##231SSZr_Intkz,       \
          Name##132SSZm_Intkz, Name##213SSZm_Intkz, Name##231SSZm_Intkz,       \
          X86InstrFMA3Group::X86FMA3Intrinsic |                                \
              X86InstrFMA3Group::X86FMA3KZeroMasked);                          \
  FMA3RMA(Name##132SDZr_Intkz, Name##213SDZr_Intkz, Name##231SDZr_Intkz,       \
          Name##132SDZm_Intkz, Name##213SDZm_Intkz, Name##231SDZm_Intkz,       \
          X86InstrFMA3Group::X86FMA3Intrinsic |                                \
              X86InstrFMA3Group::X86FMA3KZeroMasked);                          \
  FMA3RA(Name##132SSZrb_Int, Name##213SSZrb_Int, Name##231SSZrb_Int,           \
         X86InstrFMA3Group::X86FMA3Intrinsic);                                 \
  FMA3RA(Name##132SDZrb_Int, Name##213SDZrb_Int, Name##231SDZrb_Int,           \
         X86InstrFMA3Group::X86FMA3Intrinsic);                                 \
  FMA3RA(Name##132SSZrb_Intk, Name##213SSZrb_Intk, Name##231SSZrb_Intk,        \
         X86InstrFMA3Group::X86FMA3Intrinsic |                                 \
             X86InstrFMA3Group::X86FMA3KMergeMasked);                          \
  FMA3RA(Name##132SDZrb_Intk, Name##213SDZrb_Intk, Name##231SDZrb_Intk,        \
         X86InstrFMA3Group::X86FMA3Intrinsic |                                 \
             X86InstrFMA3Group::X86FMA3KMergeMasked);                          \
  FMA3RA(Name##132SSZrb_Intkz, Name##213SSZrb_Intkz, Name##231SSZrb_Intkz,     \
         X86InstrFMA3Group::X86FMA3Intrinsic |                                 \
             X86InstrFMA3Group::X86FMA3KZeroMasked);                           \
  FMA3RA(Name##132SDZrb_Intkz, Name##213SDZrb_Intkz, Name##231SDZrb_Intkz,     \
         X86InstrFMA3Group::X86FMA3Intrinsic |                                 \
             X86InstrFMA3Group::X86FMA3KZeroMasked);

#define FMA3_AVX512_FULL_GROUP(Name)                                           \
  FMA3_AVX512_VECTOR_GROUP(Name);                                              \
  FMA3_AVX512_SCALAR_GROUP(Name);

void X86InstrFMA3Info::initGroupsOnceImpl() {
  FMA3_AVX2_FULL_GROUP(VFMADD);
  FMA3_AVX2_FULL_GROUP(VFMSUB);
  FMA3_AVX2_FULL_GROUP(VFNMADD);
  FMA3_AVX2_FULL_GROUP(VFNMSUB);

  FMA3_AVX2_VECTOR_GROUP(VFMADDSUB);
  FMA3_AVX2_VECTOR_GROUP(VFMSUBADD);

  FMA3_AVX512_FULL_GROUP(VFMADD);
  FMA3_AVX512_FULL_GROUP(VFMSUB);
  FMA3_AVX512_FULL_GROUP(VFNMADD);
  FMA3_AVX512_FULL_GROUP(VFNMSUB);

  FMA3_AVX512_VECTOR_GROUP(VFMADDSUB);
  FMA3_AVX512_VECTOR_GROUP(VFMSUBADD);
}

void X86InstrFMA3Info::initGroupsOnce() {
  llvm::call_once(InitGroupsOnceFlag,
                  []() { getX86InstrFMA3Info()->initGroupsOnceImpl(); });
}
