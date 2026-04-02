//===-- ConnexISelLowering.cpp - Connex DAG Lowering Implementation  ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that Connex uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
//
// See http://llvm.org/docs/doxygen/html/classllvm_1_1DILocation.html
#include "llvm/IR/DebugInfoMetadata.h"
//
#include "Connex.h"
#include "ConnexISelLowering.h"
#include "ConnexISelMisc.h"
#include "ConnexSubtarget.h"
#include "ConnexTargetMachine.h"

using namespace llvm;

#define DEBUG_TYPE "connex-lower"

//#define DO_F16_EMULATION_IN_ISEL_LOWERING
#ifdef DO_F16_EMULATION_IN_ISEL_LOWERING
#define DO_MUL_F16_EMULATION_IN_ISEL_LOWERING
#define DO_ADD_F16_EMULATION_IN_ISEL_LOWERING
#endif

static void fail(const SDLoc &DL, SelectionDAG &DAG, const char *Msg) {
  MachineFunction &MF = DAG.getMachineFunction();
  DAG.getContext()->diagnose(
      DiagnosticInfoUnsupported(MF.getFunction(), Msg, DL.getDebugLoc()));
}

static void fail(const SDLoc &DL, SelectionDAG &DAG, const char *Msg,
                 SDValue Val) {
  MachineFunction &MF = DAG.getMachineFunction();
  std::string Str;
  raw_string_ostream OS(Str);
  OS << Msg;
  Val->print(OS);
  OS.flush();
  DAG.getContext()->diagnose(
      DiagnosticInfoUnsupported(MF.getFunction(), Str, DL.getDebugLoc()));
}

// See MipsTargetLowering::getTargetNode()
SDValue ConnexTargetLowering::getTargetNode(ConstantPoolSDNode *N, EVT Ty,
                                            SelectionDAG &DAG,
                                            unsigned Flag) const {
  return DAG.getTargetConstantPool(N->getConstVal(), Ty, N->getAlign(),
                                   N->getOffset(), Flag);
}

// Enable vector (inspired from Mips MSA) support for the given integer
//    type and Register class.
void ConnexTargetLowering::addVectorIntType(MVT::SimpleValueType aType,
                                            const TargetRegisterClass *RC) {
  LLVM_DEBUG(dbgs() << "Entered addVectorIntType(aType = " << aType << ")\n");
  // LLVM_DEBUG(dbgs() << "addVectorIntType(): "; RC->dump(); dbgs() << "\n");

  addRegisterClass(aType, RC);

  // Expand all builtin opcodes.
  for (unsigned Opc = 0; Opc < ISD::BUILTIN_OP_END; ++Opc)
    setOperationAction(Opc, aType, Expand);

  // Original code:
  // setOperationAction(ISD::BITCAST, aType, Legal);
  setOperationAction(ISD::BITCAST, aType, Custom);
  /*
  setOperationAction(ISD::BITCAST, aType, Promote);
  // setOperationAction(ISD::BITCAST, TYPE_VECTOR_I32, Promote);
  // Inspired from book Cardoso_2014, page 152
  AddPromotedToType(ISD::BITCAST, TYPE_VECTOR_I16, TYPE_VECTOR_I32);
  */

  // This is found in include/llvm/Target/TargetSelectionDAG.td
  setOperationAction(ISD::NON_EXTLOAD, aType, Legal);
  setOperationAction(ISD::EXTLOAD, aType, Legal);

  setOperationAction(ISD::LOAD, aType, Legal);
  setOperationAction(ISD::STORE, aType, Legal);

  setOperationAction(ISD::ABS, TYPE_VECTOR_I32 /*MVT::v4i32*/, Legal);

  /* Important: NONE of these seem to be required anymore after the last
      changes of the TableGen spec in ConnexInstrInfo_vec.td.

  // This is to help instruction selection of masked_gather:
  //addVectorIntType(TYPE_VECTOR_I64, &Connex::VectorHRegClass);
  //
  //setOperationAction(ISD::MGATHER, aType, Legal);
  //setOperationAction(ISD::MGATHER, aType, Custom);
  */
  /* Required if we work with index vector that is not zeroinitializer,
   * or if it is LD256[<unknown>]
   * NOT with something like LD256[%B] */
  // setOperationAction(ISD::MGATHER, TYPE_VECTOR_I32, Custom);
  // We require this to call replaceAddI32UseWithADDVH()
  setOperationAction(ISD::MGATHER, aType, Custom);
  // setOperationAction(ISD::MGATHER, aType, Legal);
  // setOperationAction(ISD::MGATHER, aType, Legal);
  /*
  setOperationAction(ISD::MGATHER, aType, Legal);
  setOperationAction(ISD::MGATHER, TYPE_VECTOR_I64, Legal);
  */

  // Failing to put this line gives the following STRANGE error - can't explain
  //       why this happens:
  //   include/llvm/CodeGen/ValueTypes.h:249:
  //    unsigned int llvm::EVT::getVectorNumElements() const:
  //    Assertion `isVector() && "Invalid vector type!"' failed.
  // setOperationAction(ISD::MSCATTER, aType, Legal);
  //
  setOperationAction(ISD::MSCATTER, aType, Custom);
  /*
  setOperationAction(ISD::MSCATTER, MVT::v64i32, Expand);
  AddPromotedToType(ISD::MSCATTER, TYPE_VECTOR_I32, TYPE_VECTOR_I16);
  */
  setOperationAction(ISD::EXTRACT_VECTOR_ELT, aType, Custom);

  // TODO!!!!: do a call to addVectorIntType(MVT::i32) instead of this
  /*
  setOperationAction(ISD::INSERT_VECTOR_ELT, MVT::i16, Legal);
  setOperationAction(ISD::INSERT_VECTOR_ELT, MVT::i32, Legal);
  setOperationAction(ISD::INSERT_VECTOR_ELT, aType, Legal);
  */
  setOperationAction(ISD::INSERT_VECTOR_ELT, MVT::i16, Custom);
  setOperationAction(ISD::INSERT_VECTOR_ELT, MVT::i32, Custom);
  setOperationAction(ISD::INSERT_VECTOR_ELT, aType, Custom);

  setOperationAction(ISD::BUILD_VECTOR, aType, Custom);

  // TODO: do a call to addVectorIntType(MVT::i32) instead of this
  setOperationAction(ISD::ConstantPool, MVT::i32, Custom);
  setOperationAction(ISD::ConstantPool, aType, Custom);

  setOperationAction(ISD::ADD, aType, Legal);
  setOperationAction(ISD::AND, aType, Legal);
  setOperationAction(ISD::CTLZ, aType, Legal);
  setOperationAction(ISD::CTPOP, aType, Legal);
  setOperationAction(ISD::MUL, aType, Legal);
  setOperationAction(ISD::OR, aType, Legal);

  // setOperationAction(ISD::SDIV, aType, Custom);
  setOperationAction(ISD::SDIV, aType, Legal);

  // setOperationAction(ISD::SREM, aType, Custom);
  setOperationAction(ISD::SREM, aType, Legal);

  setOperationAction(ISD::SHL, aType, Legal);

  setOperationAction(ISD::SRA, aType, Legal);
  // setOperationAction(ISD::SRA, aType, Custom);

  setOperationAction(ISD::SRL, aType, Legal);
  setOperationAction(ISD::SUB, aType, Legal);

  // setOperationAction(ISD::UDIV, aType, Custom);
  setOperationAction(ISD::UDIV, aType, Legal);

  // setOperationAction(ISD::UREM, aType, Custom);
  setOperationAction(ISD::UREM, aType, Legal);

  setOperationAction(ISD::VECTOR_SHUFFLE, aType, Custom);
  setOperationAction(ISD::VSELECT, aType, Legal);
  setOperationAction(ISD::XOR, aType, Legal);

  /*
  if (aType == MVT::v4i32 || aType == MVT::v2i64) {
    setOperationAction(ISD::FP_TO_SINT, aType, Legal);
    setOperationAction(ISD::FP_TO_UINT, aType, Legal);
    setOperationAction(ISD::SINT_TO_FP, aType, Legal);
    setOperationAction(ISD::UINT_TO_FP, aType, Legal);
  }
  */

  // changed
  setOperationAction(ISD::SETCC, aType, Legal);
  /*
  // Following advice Bruno Cardoso - see email Jun 7, 2016
  setOperationAction(ISD::SETCC, aType, Custom); // Expand, Promote or Legal
  */

  setCondCodeAction(ISD::SETEQ, aType, Legal);
  setCondCodeAction(ISD::SETNE, aType, Expand);
  setCondCodeAction(ISD::SETGE, aType, Expand);
  setCondCodeAction(ISD::SETGT, aType, Expand);
  setCondCodeAction(ISD::SETUGE, aType, Expand);
  setCondCodeAction(ISD::SETUGT, aType, Expand);
}

// Inspired from lib/Target/Mips/MipsSEISelLowering.cpp, addMSAFloatType()
// Enable support for the given floating-point type and Register class.
void ConnexTargetLowering::addVectorFloatType(MVT::SimpleValueType aType,
                                              const TargetRegisterClass *RC) {
  LLVM_DEBUG(dbgs() << "Entered addVectorFloatType(aType = " << aType << ")\n");
  addRegisterClass(aType, RC);

  // Expand all builtin opcodes.
  for (unsigned Opc = 0; Opc < ISD::BUILTIN_OP_END; ++Opc)
    setOperationAction(Opc, aType, Expand);

  setOperationAction(ISD::LOAD, aType, Legal);
  setOperationAction(ISD::STORE, aType, Legal);
  setOperationAction(ISD::BITCAST, aType, Legal);
  setOperationAction(ISD::EXTRACT_VECTOR_ELT, aType, Legal);
  setOperationAction(ISD::INSERT_VECTOR_ELT, aType, Legal);
  setOperationAction(ISD::BUILD_VECTOR, aType, Custom);

  setOperationAction(ISD::MGATHER, aType, Custom);
  // setOperationAction(ISD::MSCATTER, aType, Legal);
  setOperationAction(ISD::MSCATTER, aType, Custom);
  // TODO: only if we use f32, f64 I guess: setOperationAction(ISD::MSCATTER,
  //   aType, Custom);

  // if (Ty != MVT::v8f16) {
  setOperationAction(ISD::FABS, aType, Legal);
#ifdef DO_ADD_F16_EMULATION_IN_ISEL_LOWERING
  setOperationAction(ISD::FADD, aType, Custom);
#else
  setOperationAction(ISD::FADD, aType, Legal);
#endif
  //
  setOperationAction(ISD::FDIV, aType, Legal);
  setOperationAction(ISD::FEXP2, aType, Legal);
  setOperationAction(ISD::FLOG2, aType, Legal);
  setOperationAction(ISD::FMA, aType, Legal);
#ifdef DO_MUL_F16_EMULATION_IN_ISEL_LOWERING
  setOperationAction(ISD::FMUL, aType, Custom);
#else
  setOperationAction(ISD::FMUL, aType, Legal);
#endif
  setOperationAction(ISD::FRINT, aType, Legal);
  setOperationAction(ISD::FSQRT, aType, Legal);
  setOperationAction(ISD::FSUB, aType, Legal);
  setOperationAction(ISD::FNEG, aType, Legal);
  setOperationAction(ISD::FABS, aType, Legal);
  setOperationAction(ISD::VSELECT, aType, Legal);

  setOperationAction(ISD::SETCC, aType, Legal);
  setCondCodeAction(ISD::SETOGE, aType, Expand);
  setCondCodeAction(ISD::SETOGT, aType, Expand);
  setCondCodeAction(ISD::SETUGE, aType, Expand);
  setCondCodeAction(ISD::SETUGT, aType, Expand);
  setCondCodeAction(ISD::SETGE, aType, Expand);
  setCondCodeAction(ISD::SETGT, aType, Expand);
  //}
}

// Inspired from llvm/lib/Target/Mips/MipsSEISelLowering.cpp
bool ConnexTargetLowering::allowsMisalignedMemoryAccesses(EVT VT, unsigned,
                                                          unsigned,
                                                          bool *Fast) const {
  // MVT::SimpleValueType SVT = VT.getSimpleVT().SimpleTy;

  //  if (Subtarget.systemSupportsUnalignedAccess()) {
  // MIPS32r6/MIPS64r6 is required to support unaligned access. It's
  // implementation defined whether this is handled by hardware, software, or
  // a hybrid of the two but it's expected that most implementations will
  // handle the majority of cases in hardware.
  if (Fast)
    *Fast = true;
  return true;
  //  }

  /*
  switch (SVT) {
  case MVT::i64:
  case MVT::i32:
    if (Fast)
      *Fast = true;
    return true;
  default:
    return false;
  }
  */
}

ConnexTargetLowering::ConnexTargetLowering(const TargetMachine &TM,
                                           const ConnexSubtarget &STI)
    : TargetLowering(TM) {

  LLVM_DEBUG(
      dbgs() << "Entered ConnexTargetLowering::ConnexTargetLowering(): \n");

 #if 0
  // MEGA-MEGA-TODO: find a way to register memmove&memset. See also a bit
  //   https://discourse.llvm.org/t/lib-call-selection/88653 (not really useful)

  // Inspired from lib/Target/ARM/ARMISelLowering.cpp,
  // ARMTargetLowering::ARMTargetLowering()
  static const struct {
    const RTLIB::Libcall Op;
    const char *const Name;
    const CallingConv::ID CC;
    const ISD::CondCode Cond;
  } MemOpsLibraryCalls[] = {
      // Memory operations
      // RTABI chapter 4.3.4
      /*
      // NOTE: CallingConv::ARM_AAPCS is defined in
      //     http://llvm.org/docs/doxygen/html/namespacellvm_1_1CallingConv.html
      { RTLIB::MEMCPY,  "__aeabi_memcpy",  CallingConv::ARM_AAPCS,
                                           ISD::SETCC_INVALID },
      { RTLIB::MEMMOVE, "__aeabi_memmove", CallingConv::ARM_AAPCS,
                                           ISD::SETCC_INVALID },
      { RTLIB::MEMSET,  "__aeabi_memset",  CallingConv::ARM_AAPCS,
                                           ISD::SETCC_INVALID },
      */
      {RTLIB::MEMMOVE, "memmove", CallingConv::C, ISD::SETCC_INVALID},
      {RTLIB::MEMSET, "memset", CallingConv::C, ISD::SETCC_INVALID},
  };

  for (const auto &LC : MemOpsLibraryCalls) {
    LLVM_DEBUG(dbgs() << "ConnexTargetLowering::ConnexTargetLowering(): "
                         "registering RT-Libcall LC.name = "
                      << LC.Name << "\n");

    setLibcallName(LC.Op, LC.Name);
    setLibcallCallingConv(LC.Op, LC.CC);
    if (LC.Cond != ISD::SETCC_INVALID) {
      setCmpLibcallCC(LC.Op, LC.Cond);
    }
  }
 #endif

  // Set up the register classes.
  // TODO_CHANGE_BACKEND:
  // addRegisterClass(MVT::i64, &Connex::GPRRegClass);
  addRegisterClass(TYPE_SCALAR_ELEMENT, &Connex::GPRRegClass);

  // Taken from llvm/lib/Target/Mips/MipsSEISelLowering.cpp
  // if (Subtarget.hasDSP() || Subtarget.hasMSA()) {
  // Expand all truncating stores and extending loads.
  for (MVT VT0 : MVT::vector_valuetypes()) {
    for (MVT VT1 : MVT::vector_valuetypes()) {
      /*
      LLVM_DEBUG(dbgs() << "VT0.getSizeInBits() = "
                        << VT0.getSizeInBits() << "\n");
      LLVM_DEBUG(dbgs() << "VT1.getSizeInBits() = "
                        << VT1.getSizeInBits() << "\n");
      */
      setTruncStoreAction(VT0, VT1, Expand);
      // This is WRONG - it was added by me and caused llc to give core dump:
      //      setLoadExtAction(ISD::STORE, VT0, VT1, Expand);

      setLoadExtAction(ISD::SEXTLOAD, VT0, VT1, Expand);
      // setLoadExtAction(ISD::SEXTLOAD, VT0, VT1, Legal);
      setLoadExtAction(ISD::ZEXTLOAD, VT0, VT1, Expand);
      setLoadExtAction(ISD::EXTLOAD, VT0, VT1, Expand);
    }
  }
  //}

  // As said in [Pandey_2015], page 152:
  //   "The legalize phase can also instruct the kind of classes of registers
  //   supported for given data."

  // Taken from llvm/lib/Target/Mips/MipsSEISelLowering.cpp
  // if (Subtarget.hasMSA()) {
  /*
  addVectorIntType(MVT::v16i8, &Connex::MSA128BRegClass);
  addVectorIntType(MVT::v8i16, &Connex::VectorHRegClass);
  addVectorIntType(MVT::v4i32, &Connex::MSA128WRegClass);
  addVectorIntType(MVT::v2i64, &Connex::VectorHRegClass);
  */

  /*
  // TODO to add these reg classes in the end
  addVectorIntType(MVT::v64i8, &Connex::MSA128BRegClass);
  addVectorIntType(MVT::v32i16, &Connex::VectorHRegClass);
  addVectorIntType(MVT::v16i32, &Connex::MSA128WRegClass);
  */
  // TODO_CHANGE_BACKEND:
  // This is to help instruction selection of masked_gather:
  // addVectorIntType(MVT::v8i64, &Connex::VectorHRegClass);
  // To prevent error: <<Assertion `memvt.getStoreSize() <= MMO->getSize() &&
  //        "Size mismatch!"' failed.>>:
  addVectorIntType(MVT::v8i64, &Connex::VectorHRegClass);
  //
  addVectorIntType(TYPE_VECTOR_I16, &Connex::VectorHRegClass);

  // NEW32
  // addVectorIntType(TYPE_VECTOR_I32, &Connex::MSA128WRegClass);
  addVectorIntType(TYPE_VECTOR_I32, &Connex::VectorHRegClass);
  //

  /*
  // These are not useful since we already gave
  //     addVectorIntType(TYPE_VECTOR_I32) above
  // NEW32
  LLVM_DEBUG(dbgs() << "Calling addRegisterClass(TYPE_VECTOR_I32,
                                                 &Connex::MSA128WRegClass)\n");
  addRegisterClass(TYPE_VECTOR_I32, &Connex::MSA128WRegClass);
  */

  /*
  LLVM_DEBUG(dbgs() << "Calling setOperationAction(ISD::ADD, Custom)\n");
  setOperationAction(ISD::ADD, TYPE_VECTOR_I32, Custom);
  */

  /*
  LLVM_DEBUG(dbgs() << "Calling setOperationAction(ISD::ADD, Expand)\n");
  setOperationAction(ISD::ADD, TYPE_VECTOR_I32, Expand);
  AddPromotedToType(ISD::ADD, TYPE_VECTOR_I32, TYPE_VECTOR_I16);
  */

  /* SPECIAL_BITCAST_PROMOTE_EXPAND
  // NEW32
  // This normally results in having at I-sel something like:
  // Legally typed node: t35: v64i32,ch = masked_gather<
  //                                     LD256[inttoptr (i32 51 to i32*)]
  //                                      (alias.scope=<0xbe0aa0>)> t21,
  //                                     undef:v64i32, t37, Constant:i64<51>,
  t23
  // Promote integer result: t535: i32 = extract_vector_elt t35, Constant:i64<0>
  // Legally typed node: t727: i64 = extract_vector_elt t35, Constant:i64<0>
  // Promote integer result: t538: i32 = extract_vector_elt t35, Constant:i64<1>
  // Legally typed node: t728: i64 = extract_vector_elt t35, Constant:i64<1>

  // Inspired from book Cardoso_2014, page 152
  //
  LLVM_DEBUG(dbgs() << "Calling setOperationAction(ISD::OR, Expand)\n");
  setOperationAction(ISD::OR, TYPE_VECTOR_I32,
                              Expand); // Promote
  AddPromotedToType(ISD::OR,
                    TYPE_VECTOR_I32, // src
                    TYPE_VECTOR_I16); // dst


  LLVM_DEBUG(dbgs() << "ISD::BITCAST - we use "
                       "setOperationAction(..., Expand).\n");
  setOperationAction(ISD::BITCAST, TYPE_VECTOR_I16,
                     Expand); // Promote
  AddPromotedToType(ISD::BITCAST,
                    TYPE_VECTOR_I16, // src
                    TYPE_VECTOR_I32); // dst
  setOperationAction(ISD::BITCAST, TYPE_VECTOR_I32,
                     Expand); // Promote
  AddPromotedToType(ISD::BITCAST,
                    TYPE_VECTOR_I32, // src
                    TYPE_VECTOR_I16); // dst

  LLVM_DEBUG(dbgs() << "ISD::ADD - we use setOperationAction(..., Expand).\n");
  setOperationAction(ISD::ADD, TYPE_VECTOR_I16,
                     Expand); // Promote
  AddPromotedToType(ISD::ADD,
                    TYPE_VECTOR_I16, // src
                    TYPE_VECTOR_I32); // dst
  setOperationAction(ISD::ADD, TYPE_VECTOR_I32,
                     Expand); // Promote
  AddPromotedToType(ISD::ADD,
                    TYPE_VECTOR_I32, // src
                    TYPE_VECTOR_I16); // dst
  */

  // addVectorFloatType(MVT::v128f16, &Connex::VectorHRegClass);
  addVectorFloatType(TYPE_VECTOR_F16, &Connex::VectorHRegClass);

  /*
  addVectorFloatType(MVT::v8f16, &Mips::VectorHRegClass);
  addVectorFloatType(MVT::v4f32, &Mips::MSA128WRegClass);
  addVectorFloatType(MVT::v2f64, &Mips::VectorHRegClass);
  */
  /*
  From http://llvm.org/docs/doxygen/html/classllvm_1_1TargetLoweringBase.html:
    void llvm::TargetLoweringBase::setTargetDAGCombine(ISD::NodeType NT)
                                                            [inline, protected]
      <<Targets should invoke this method for each target independent
        node that they want to provide a custom DAG combiner for by
        implementing the PerformDAGCombine virtual method.>>
  */
  setTargetDAGCombine(ISD::AND);
  setTargetDAGCombine(ISD::OR);
  setTargetDAGCombine(ISD::SRA);
  setTargetDAGCombine(ISD::XOR);
  //}

  /* Very Important: this is why I've spent ~5 days of debugging
   * - the computeRegisterProperties() function is called at the end of the
   * constructor in lib/Target/Mips/MipsSEISelLowering.cpp (or
   * Mips16SEILoweing.cpp; note that ARM/ARMISelLowering.cpp is somewhat similar
   * to our case - computeRegisterProperties() is called AFTER all
   * addRegisterClass() calls).
   * But here it is called in the "middle", after the types are being
   * declared - i.e., addRegisterClass() has to be called BEFORE
   * computeRegisterProperties() - THIS IS Very Important.
   */
  // Compute derived properties from the register classes
  computeRegisterProperties(STI.getRegisterInfo());

  setStackPointerRegisterToSaveRestore(Connex::R11);

#if 0 // NEW_BIGGER_OPS
  /*
  setOperationAction(ISD::DIV, MVT::u16, Custom);
  setOperationAction(ISD::DIV, MVT::i16, Custom);
  */

  if (MVT::i16 != TYPE_SCALAR_ELEMENT) {
      setOperationAction(ISD::MUL, MVT::i16, Custom);
  }
  if (MVT::i32 != TYPE_SCALAR_ELEMENT) {
      setOperationAction(ISD::ADD, MVT::i32, Custom);
      setOperationAction(ISD::SUB, MVT::i32, Custom);
      setOperationAction(ISD::MUL, MVT::i32, Custom);
  }
#endif
  /*
  setOperationAction(ISD::~~~~VLOAD, TYPE_VECTOR_I32, Custom);
  setOperationAction(ISD::MGATHER, TYPE_VECTOR_I32, Custom);
  */

  /*
   From http://llvm.org/docs/doxygen/html/classllvm_1_1TargetLoweringBase.html
     void setOperationAction(unsigned Op, MVT VT, LegalizeAction Action)
      Indicate that the specified operation does not work with the specified
        type and indicate what to do about it.

   // From
   //   llvm.org/docs/WritingAnLLVMBackend.html#the-selectiondag-legalize-phase
    "For some operations, simple type promotion or operation expansion may be
      insufficient.
    [...]
    In the LowerOperation method, for each Custom operation, a case statement
      should be added to indicate what function to call.
    "
  */
  // TODO_CHANGE_BACKEND:
  setOperationAction(ISD::BR_CC, TYPE_SCALAR_ELEMENT, Custom);

  setOperationAction(ISD::BR_JT, MVT::Other, Expand);
  setOperationAction(ISD::BRIND, MVT::Other, Expand);
  setOperationAction(ISD::BRCOND, MVT::Other, Expand);

  setOperationAction(ISD::SETCC, TYPE_SCALAR_ELEMENT, Expand);

  setOperationAction(ISD::SELECT, TYPE_SCALAR_ELEMENT, Expand);

  setOperationAction(ISD::SELECT_CC, TYPE_SCALAR_ELEMENT, Custom);

  setOperationAction(ISD::GlobalAddress, TYPE_SCALAR_ELEMENT, Custom);

  setOperationAction(ISD::DYNAMIC_STACKALLOC, TYPE_SCALAR_ELEMENT, Custom);

  setOperationAction(ISD::STACKSAVE, MVT::Other, Expand);
  setOperationAction(ISD::STACKRESTORE, MVT::Other, Expand);

  setOperationAction(ISD::SDIVREM, TYPE_SCALAR_ELEMENT, Expand);
  setOperationAction(ISD::UDIVREM, TYPE_SCALAR_ELEMENT, Expand);

  setOperationAction(ISD::SREM, TYPE_SCALAR_ELEMENT, Expand);
  setOperationAction(ISD::UREM, TYPE_SCALAR_ELEMENT, Expand);

  setOperationAction(ISD::MULHU, TYPE_SCALAR_ELEMENT, Expand);
  setOperationAction(ISD::MULHS, TYPE_SCALAR_ELEMENT, Expand);

  setOperationAction(ISD::UMUL_LOHI, TYPE_SCALAR_ELEMENT, Expand);
  setOperationAction(ISD::SMUL_LOHI, TYPE_SCALAR_ELEMENT, Expand);

  setOperationAction(ISD::ADDC, TYPE_SCALAR_ELEMENT, Expand);
  setOperationAction(ISD::ADDE, TYPE_SCALAR_ELEMENT, Expand);

  setOperationAction(ISD::SUBC, TYPE_SCALAR_ELEMENT, Expand);
  setOperationAction(ISD::SUBE, TYPE_SCALAR_ELEMENT, Expand);

  setOperationAction(ISD::ROTR, TYPE_SCALAR_ELEMENT, Expand);
  setOperationAction(ISD::ROTL, TYPE_SCALAR_ELEMENT, Expand);

  setOperationAction(ISD::SHL_PARTS, TYPE_SCALAR_ELEMENT, Expand);
  setOperationAction(ISD::SRL_PARTS, TYPE_SCALAR_ELEMENT, Expand);
  setOperationAction(ISD::SRA_PARTS, TYPE_SCALAR_ELEMENT, Expand);

  setOperationAction(ISD::CTTZ, TYPE_SCALAR_ELEMENT, Custom);
  setOperationAction(ISD::CTLZ, TYPE_SCALAR_ELEMENT, Custom);
  //
  setOperationAction(ISD::CTTZ_ZERO_UNDEF, TYPE_SCALAR_ELEMENT, Custom);
  setOperationAction(ISD::CTLZ_ZERO_UNDEF, TYPE_SCALAR_ELEMENT, Custom);

  setOperationAction(ISD::CTPOP, TYPE_SCALAR_ELEMENT, Expand);

  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1, Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i8, Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i16, Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i32, Expand);
  // setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i64, Legal);
  //
  // setOperationAction(ISD::SIGN_EXTEND, MVT::i64, Legal);
  // Used ONLY for MGATHER and MSCATTER
  setOperationAction(ISD::SIGN_EXTEND, TYPE_VECTOR_I16_EXT_I64, Legal);
  setOperationAction(ISD::SIGN_EXTEND, TYPE_VECTOR_I32_EXT_I64, Legal);
  setOperationAction(ISD::SIGN_EXTEND, MVT::v4i32, Legal);
  // Very Important: the following lines legalize v4i64 = sext v4i32 used ONLY
  //     by MGATHER and MSCATTER (llvm.masked.gather/scatter).
  //   Seems to be the only one that does legalize this well...
  // 2021_01_27: ValueTypeActions.setTypeAction(MVT::v8i64, TypeLegal);
  // 2021_02_02_TODO check if OK since const does NOT allow this call anymore
  //   getValueTypeActions().setTypeAction(MVT::v8i64, TypeLegal);
  // 2021_01_27: ValueTypeActions.setTypeAction(MVT::v4i64, TypeLegal);
  // 2021_02_02_TODO check if OK since const does NOT allow this call anymore
  // getValueTypeActions().setTypeAction(MVT::v4i64, TypeLegal);
  //  ValueTypeActions.setTypeAction(MVT::v8f16, TypeCustom);
  // setOperationAction(ISD::SIGN_EXTEND, TYPE_VECTOR_I32, Legal);
  // setOperationAction(ISD::SIGN_EXTEND, TYPE_VECTOR_I64, Legal);

  // Inspired from llvm/lib/Target/X86/X86ISelLowering.cpp.
  // Important: From
  //  llvm.org/svn/llvm-project/llvm/trunk/include/llvm/Target/TargetLowering.h
  /// Convenience method to set an operation to Promote and specify the type
  /// in a single call.
  // void setOperationPromotedToType(unsigned Opc, MVT OrigVT, MVT DestVT) {
  //   setOperationAction(Opc, OrigVT, Promote);
  //   AddPromotedToType(Opc, OrigVT, DestVT);
  // }

  setOperationPromotedToType(ISD::ConstantPool, MVT::i32, TYPE_SCALAR_ELEMENT);
  setOperationPromotedToType(ISD::Constant, MVT::i32, TYPE_SCALAR_ELEMENT);
  setOperationPromotedToType(ISD::ADD, MVT::i32, TYPE_SCALAR_ELEMENT);

  setOperationPromotedToType(ISD::ADD, MVT::i16, TYPE_SCALAR_ELEMENT);
  setOperationPromotedToType(ISD::ADD, MVT::i64, MVT::i32);

  // Inspired from AMDGPU/AMDGPUISelLowering.cpp
  // Need DAG EVT LegalVT = getTypeToTransformTo(*DAG.getContext(), MVT::i32);
  // LLVM_DEBUG(dbgs() << "addVectorIntType(): LegalVT " << LegalVT << "\n");

  // Extended load operations for i1 types must be promoted
  for (MVT VT : MVT::integer_valuetypes()) {
    setLoadExtAction(ISD::EXTLOAD, VT, MVT::i1, Promote);
    setLoadExtAction(ISD::ZEXTLOAD, VT, MVT::i1, Promote);
    setLoadExtAction(ISD::SEXTLOAD, VT, MVT::i1, Promote);

    setLoadExtAction(ISD::SEXTLOAD, VT, MVT::i8, Expand);
    setLoadExtAction(ISD::SEXTLOAD, VT, MVT::i16, Expand);
    setLoadExtAction(ISD::SEXTLOAD, VT, MVT::i32, Expand);
    // setLoadExtAction(ISD::SEXTLOAD, VT, MVT::i32, Legal);
  }

  setBooleanContents(ZeroOrOneBooleanContent);

  // Function alignments (log2)
  setMinFunctionAlignment(Align(8));
  setPrefFunctionAlignment(Align(8));

  // inline memcpy() for kernel to see explicit copy
  MaxStoresPerMemset = MaxStoresPerMemsetOptSize = CONNEX_VECTOR_LENGTH;
  MaxStoresPerMemcpy = MaxStoresPerMemcpyOptSize = CONNEX_VECTOR_LENGTH;
  MaxStoresPerMemmove = MaxStoresPerMemmoveOptSize = CONNEX_VECTOR_LENGTH;

  // Inspired from ARMISelLowering.cpp:
  for (unsigned im = (unsigned)ISD::PRE_INC;
       im != (unsigned)ISD::LAST_INDEXED_MODE; ++im) {
    setIndexedLoadAction(im, MVT::i64, Legal);
    setIndexedLoadAction(im, MVT::i16, Promote);
    setIndexedStoreAction(im, MVT::i64, Legal);
    setIndexedStoreAction(im, MVT::i16, Promote);
  }

  /*
  // LLVM_DEBUG(dbgs() << "addVectorIntType(): calling setTypeAction()\n");
  // ValueTypeActionImpl::setTypeAction(MVT::i16, TypePromoteInteger);
  // ValueTypeActionImpl::setTypeAction(MVT::i32, TypePromoteInteger);
  //
  // LLVM_DEBUG(dbgs() << "addVectorIntType(): calling setTypeAction()\n");
  // ValueTypeActionImpl::setTypeAction(MVT::i16, TypeLegal);
  // ValueTypeActionImpl::setTypeAction(MVT::i32, TypeLegal);

  // IMPORTANT: the whole reason I am using below setTypeAction() is that we get
  //    error:
  // <<void llvm::TargetLoweringBase::setOperationAction(unsigned int,
  //             llvm::MVT, llvm::TargetLoweringBase::LegalizeAction):
  //    Assertion `Op < array_lengthof(OpActions[0]) &&
  //                                   "Table isn't big enough!"' failed.>>
  //  when using setOperationAction(Intrinsic::connex_repeat_x_times...):
  //    //setOperationAction(Intrinsic::connex_repeat_x_times, MVT::i16,
  //                         Expand); //Legal);
  //    //setOperationAction(Intrinsic::connex_repeat_x_times, MVT::i32, Legal);
  //    //setOperationAction(Intrinsic::connex_repeat_x_times, MVT::i32,
  //                         Promote);


  // IMPORTANT: This piece of code HAS to be put at the end of this method
  //  because otherwise one or more of the above calls are rendering this
  //  setTypeAction() below useless.
  //  But then it gives error like:
  //    <<LLVM ERROR: Cannot select: t39: ch = br_cc t26, setlt:ch, t17,
  //             Constant:i16<1>, BasicBlock:ch<for.cond.cleanup 0x20e28b8> >>
  //   because I made i16 a legal type and the instruction scheduler does NOT
  //      have to promote it to i64, although br_cc requires it (see TableGen
  //      definition).
  //   To fix this we should make sure we put setTypeAction(MVT::i16, TypeLegal)
  //     before all setOperation...() that intefere with it.

  // See http://llvm.org/docs/doxygen/html/TargetLowering_8h_source.html#l00096
  //  for enum LegalizeTypeAction.
  #ifdef DO_F16_EMULATION_IN_ISEL_LOWERING
  setOperationAction(Intrinsic::connex_reduce_f16, MVT::f16, Custom);
  #endif

  LLVM_DEBUG(dbgs() << "ConnexTargetLowering(): calling "
                       "setTypeAction(MVT::i16, ...)\n");
  LegalizeTypeAction ta = ValueTypeActions.getTypeAction(MVT::i16);
  LLVM_DEBUG(dbgs()
         << "  Before setTypeAction(MVT::i16, ...), i16 has action "
         << ta << "\n");

  // Inspired from lib/Target/X86/X86ISelLowering.cpp:
  // Gives error: "Do not know how to promote this operator's operand!"
  ValueTypeActions.setTypeAction(MVT::i16, TypeLegal);
  // Gives error: "Do not know how to promote this operator's operand!"
  //ValueTypeActions.setTypeAction(MVT::i16, TypeExpandInteger);
                                             //TypePromoteInteger);
  ta = ValueTypeActions.getTypeAction(// DAG.getContext(),
                                      MVT::i16);
                                      //TypeLegal);
  LLVM_DEBUG(dbgs()
         << "  setTypeAction(MVT::i16, ...) has set for i16 action to "
         << ta << "\n");
  */

  // NEW_FP16: it seems these are very useful
  // See http://llvm.org/doxygen/TargetLowering_8h_source.html#l00122
  // Quite GOOD:
  // 2021_01_27: ValueTypeActions.setTypeAction(MVT::f16, TypeLegal); //
  // 2021_02_02_TODO check if OK since const does NOT allow this call anymore
  //               getValueTypeActions().setTypeAction(MVT::f16, TypeLegal);
  // 2021_02_27: Important: This gives a serious error:
  //            <<PromoteFloatOperand Op #0: t23:
  //               v8f16 = BUILD_VECTOR ConstantFP:f16<APFloat(0)>,
  //                  Do not know how to promote this operator's operand!>>
  // Does NOT seem to be useful since this function is a simple function
  //   declared in X86 back end: addLegalFPImmediate(APFloat::getZero(
  //                                                 APFloat::IEEEhalf()));
  // inspired from X86ISelLowering

  /* "// Convert this float to a same size integer type,
   if an operation is not supported in target HW." */
  // ValueTypeActions.setTypeAction(MVT::f16, TypeSoftenFloat);
  //                                       // TypePromoteInteger);

  // setOperationAction(ISD::MSCATTER, TYPE_VECTOR_F16, Legal);
  // setOperationAction(ISD::MSCATTER, TYPE_VECTOR_F16, Custom);

  /*
  // It seems it does not help:

  setOperationAction(ISD::LOAD, MVT::f16, Promote);

  // Gives <<UNREACHABLE executed at CodeGen/SelectionDAG/LegalizeDAG.cpp:465!>>
  //setOperationAction(ISD::STORE, MVT::f16, LibCall);

  setOperationAction(ISD::STORE, MVT::f16, Promote);
  setOperationAction(ISD::FADD, MVT::f16, LibCall);
  */

  AddPromotedToType(ISD::LOAD, MVT::f16, MVT::i16);
  AddPromotedToType(ISD::STORE, MVT::f16, MVT::i16);
  /*
  // It seems it doesn't help:
  AddPromotedToType(ISD::FADD, MVT::f16, MVT::i16);
  */
  // End NEW_FP16

  LLVM_DEBUG(dbgs() << "Exiting ConnexTargetLowering()\n");
} // End ConnexTargetLowering::ConnexTargetLowering()

// Inspired from lib/Target/AMDGPU/AMDGPUISelLowering.cpp
SDValue ConnexTargetLowering::LowerDYNAMIC_STACKALLOC(SDValue Op,
                                                      SelectionDAG &DAG) const {
  const Function &Fn = DAG.getMachineFunction().getFunction();

  DiagnosticInfoUnsupported NoDynamicAlloca(Fn, "unsupported dynamic alloca",
                                            SDLoc(Op).getDebugLoc());
  DAG.getContext()->diagnose(NoDynamicAlloca);
  auto Ops = {DAG.getConstant(0, SDLoc(), Op.getValueType()), Op.getOperand(0)};
  return DAG.getMergeValues(Ops, SDLoc());
}

// Inspired from lib/Target/X86/X86ISelLowering.cpp.
// Widen vector InOp to vector type NVT.
static SDValue ChangeVectorType(SDValue InOp, MVT NVT, SelectionDAG &DAG,
                                bool FillWithZeroes = false,
                                // This is meant for the index operand of
                                //   MGATHER and MSCATTER
                                bool allowUnsafeChanges = false) {
  LLVM_DEBUG(dbgs() << "  ChangeVectorType(): InOp = "; InOp.dump();
             dbgs() << "\n");

  LLVM_DEBUG(dbgs() << "  ChangeVectorType(): NVT = "
                    << ((EVT)NVT).getEVTString() << "\n");

  // Check if InOp already has the right width.
  MVT InVT = InOp.getSimpleValueType();
  if (InVT == NVT)
    return InOp;

  if (InOp.isUndef())
    return DAG.getUNDEF(NVT);

  /*
  assert(InVT.getVectorElementType() == NVT.getVectorElementType() &&
         "input and widen element type must match");
  */

  unsigned InNumElts = InVT.getVectorNumElements();
  unsigned WidenNumElts = NVT.getVectorNumElements();
  LLVM_DEBUG(dbgs() << "  ChangeVectorType(): InNumElts = " << InNumElts
                    << "\n   WidenNumElts = " << WidenNumElts << "\n");
  /*
  assert(WidenNumElts > InNumElts && WidenNumElts % InNumElts == 0 &&
         "Unexpected request for vector widening");
  */
  if (allowUnsafeChanges == false)
    assert(WidenNumElts == InNumElts && "WidenNumElts == InNumElts failed");

  EVT EltVT = NVT.getVectorElementType();

  SDLoc dl(InOp);
  if (InOp.getOpcode() == ISD::CONCAT_VECTORS && InOp.getNumOperands() == 2) {
    SDValue N1 = InOp.getOperand(1);
    if ((ISD::isBuildVectorAllZeros(N1.getNode()) && FillWithZeroes) ||
        N1.isUndef()) {
      InOp = InOp.getOperand(0);
      InVT = InOp.getSimpleValueType();
      InNumElts = InVT.getVectorNumElements();
    }
  }

  if (ISD::isBuildVectorOfConstantSDNodes(InOp.getNode()) ||
      ISD::isBuildVectorOfConstantFPSDNodes(InOp.getNode())) {
    SmallVector<SDValue, CONNEX_VECTOR_LENGTH> Ops;
    for (unsigned i = 0; i < InNumElts; ++i)
      Ops.push_back(InOp.getOperand(i));

    /*
    SDValue FillVal = FillWithZeroes ? DAG.getConstant(0, dl, EltVT) :
      DAG.getUNDEF(EltVT);
    for (unsigned i = 0; i < WidenNumElts - InNumElts; ++i)
      Ops.push_back(FillVal);
    */
    return DAG.getBuildVector(NVT, dl, Ops);
  }

  LLVM_DEBUG(dbgs() << "  ChangeVectorType(): InOp = ";
             //     << InOp.getNode() << "\n");
             InOp.dump(); dbgs() << "\n");

  if (allowUnsafeChanges == false) {
    assert(0 && "ChangeVectorType(): I guess this case should not be reached");
  } else {
    for (int idxOpnd = 0; idxOpnd < InOp.getNumOperands(); idxOpnd++) {
      LLVM_DEBUG(dbgs() << "  ChangeVectorType(): N->getOperand(" << idxOpnd
                        << ") = ";
                 InOp.getOperand(idxOpnd).dump(););
    }

    /*
    SDValue Ops[] = { InOp.getOperand(0), InOp.getOperand(1),
                InOp.getOperand(2), InOp.getOperand(3), InOp.getOperand(4),
                InOp.getOperand(5), InOp.getOperand(6), InOp.getOperand(7) };
    */
    SmallVector<SDValue, 8> Ops;
    for (int idxOpnd = 0; idxOpnd < InOp.getNumOperands(); idxOpnd++) {
      Ops.push_back(InOp.getOperand(idxOpnd));
    }

    SDValue res = DAG.getNode(InOp->getOpcode(), dl, NVT, Ops);

    LLVM_DEBUG(dbgs() << "ChangeVectorType(): res = "; res.dump();
               dbgs() << "\n");

    return res;
  }

  SDValue FillVal =
      FillWithZeroes ? DAG.getConstant(0, dl, NVT) : DAG.getUNDEF(NVT);

  return DAG.getNode(ISD::INSERT_SUBVECTOR, dl, NVT, FillVal, InOp,
                     DAG.getIntPtrConstant(0, dl));
} // End ChangeVectorType()

void ConnexTargetLowering::replaceAddI32UseWithADDVH(MVT &aType, SDValue &Index,
                                                     SelectionDAG &DAG) const {
  SDLoc dl(Index);

  LLVM_DEBUG(dbgs() << "Entered ReplaceAddI32UseWithADDVH()\n");

  //  We make an unsafe assumption that if the Index of the
  // MSCATTER/MGATHER instruction is used in an ADD, then this Index is an
  // induction variable and we can change it to i16 type
  //    (we also assume this ind.var is NOT overflowing the i16 type).
  //  MEGA-TODO: Check if initializing this Index is safely done on i32 type or
  //  on i16.

  // Very Important:
  //   The Connex processor we target allows only
  //     indirect Loads (and Stores) that work on lanes of ONLY 16-bits.
  //   Therefore we need to make sure that the index/address register is not
  //     used in i32 operations and if it is we change them to MachineNodes
  //     here, in the ISelLowering phase (before ISelDAGToDAG), that have
  //     actually type v8i16.
  // Important-TODO: make t least a check that the BUILD_VECTOR with initial
  //  index/address value is a short (i16) value AND LOWER the
  //    TYPE_VECTOR_I32 to TYPE_VECTOR_I16
  //  by doing a splat with the lower 16-bits value of element 0

  // Inspired from LegalizeTypes.cpp
  SDNode *nodeIndex = Index.getNode();
  for (SDNode::use_iterator UI = nodeIndex->use_begin(),
                            UE = nodeIndex->use_end();
       UI != UE; ++UI) {
    // SDNode *nUser = UI.getUse().getUser();
    SDNode *nUser = UI->getUser();

    /*
    if (UI.getUse().getResNo() == i)
      assert(UI->getNodeId() == NewNode &&
             "Remapped value has non-trivial use!");
    */
    LLVM_DEBUG(dbgs() << "replaceAddI32UseWithADDVH(): nUser = "; nUser->dump();
               // dbgs() << "\n"
    );

    if (nUser->getOpcode() == ISD::ADD) {
      LLVM_DEBUG(dbgs() << "replaceAddI32UseWithADDVH(): Converting nUser "
                           "ISD::ADD to MachineSDNode Connex::ADDV_H\n");

      // Important: We do here an unsafe type hack: we use ADDV_H which actually
      //   has TYPE_VECTOR_I16 and declare the type returned is TYPE_VECTOR_I32.
      //   It is a type mismatch at the level of semantics of the defined
      //     MachineSDNodes of Connex - I've actually done this before and
      //        it seems SelectionDAG doesn't complain.
      //       (Note that llc actually does TypeLegalization).
      SDNode *nUserNew =
          DAG.getMachineNode(Connex::ADDV_H, dl,
                             // TYPE_VECTOR_I16,
                             aType,
                             // Ops
                             nUser->getOperand(0), nUser->getOperand(1));

      // From http://llvm.org/docs/doxygen/html/classllvm_1_1SelectionDAG.html
      DAG.ReplaceAllUsesWith(nUser, nUserNew);
    }
  }
} // End replaceAddI32UseWithADDVH()

// Inspired from lib/Target/X86/X86ISelLowering.cpp
SDValue ConnexTargetLowering::LowerMGATHER(SDValue &Op,
                                           SelectionDAG &DAG) const {
  LLVM_DEBUG(dbgs() << "Entered ConnexTargetLowering::LowerMGATHER()\n");

  MaskedGatherSDNode *N = cast<MaskedGatherSDNode>(Op.getNode());

  SDLoc dl(Op);
  EVT resVT = Op.getSimpleValueType();

  LLVM_DEBUG(dbgs() << "LowerMGATHER(): "
                    << "resVT = " << resVT.getEVTString() << "\n");

  SDValue Index = N->getIndex();
  SDValue Mask = N->getMask();
  SDValue Src = N->getPassThru(); // this is actually passthru
  MVT IndexVT = Index.getSimpleValueType();
  MVT MaskVT = Mask.getSimpleValueType();

  // unsigned NumElts = VT.getVectorNumElements();
  // assert(VT.getScalarSizeInBits() >= 32 && "Unsupported gather op");

  /*
  if (!Subtarget.hasVLX() && !VT.is512BitVector() &&
      !Index.getSimpleValueType().is512BitVector()) {
    // AVX512F supports only 512-bit vectors. Or data or index should
    // be 512 bit wide. If now the both index and data are 256-bit, but
    // the vector contains 8 elements, we just sign-extend the index
    if (NumElts == 8) {
      Index = DAG.getNode(ISD::SIGN_EXTEND, dl, MVT::v8i64, Index);
      SDValue Ops[] = { N->getOperand(0), N->getOperand(1),  N->getOperand(2),
                        N->getOperand(3), Index };
      DAG.UpdateNodeOperands(N, Ops);
      return Op;
    }

    // Minimal number of elements in Gather
    NumElts = 8;
    // Index
    MVT NewIndexVT = MVT::getVectorVT(IndexVT.getScalarType(), NumElts);
    Index = ExtendToType(Index, NewIndexVT, DAG);
    if (IndexVT.getScalarType() == MVT::i32)
      Index = DAG.getNode(ISD::SIGN_EXTEND, dl, MVT::v8i64, Index);

    // Mask
    MVT MaskBitVT = MVT::getVectorVT(MVT::i1, NumElts);
    // At this point we have promoted mask operand
    assert(MaskVT.getScalarSizeInBits() >= 32 && "unexpected mask type");
    MVT ExtMaskVT = MVT::getVectorVT(MaskVT.getScalarType(), NumElts);
    Mask = ExtendToType(Mask, ExtMaskVT, DAG, true);
    Mask = DAG.getNode(ISD::TRUNCATE, dl, MaskBitVT, Mask);

    // The pass-thru value
    MVT NewVT = MVT::getVectorVT(VT.getScalarType(), NumElts);
    Src = ExtendToType(Src, NewVT, DAG);

    SDValue Ops[] = { N->getChain(), Src, Mask, N->getBasePtr(), Index };
    SDValue NewGather = DAG.getMaskedGather(DAG.getVTList(NewVT, MVT::Other),
                                            N->getMemoryVT(), dl, Ops,
                                            N->getMemOperand());
    SDValue Exract = DAG.getNode(ISD::EXTRACT_SUBVECTOR, dl, VT,
                                 NewGather.getValue(0),
                                 DAG.getIntPtrConstant(0, dl));
    SDValue RetOps[] = {Exract, NewGather.getValue(1)};
    return DAG.getMergeValues(RetOps, dl);
  }
  */

  LLVM_DEBUG(dbgs() << "  LowerMGATHER(): Op.getNode() = " << Op.getNode()
                    << "\n   Op = ";
             Op->dump(&DAG));

  // MVT NewVT = TYPE_VECTOR_I16;
  // SDValue Index2 = ExtendToType(Index, NewVT, DAG);
  // The index value - is normally i32, and we have to lower it to i16
  MVT aType = (resVT == TYPE_VECTOR_I16 ? TYPE_VECTOR_I16 : TYPE_VECTOR_I32);
  /*
  MVT aType = (resVT == TYPE_VECTOR_I16 ? TYPE_VECTOR_I16 : TYPE_VECTOR_I16);
  */
  // NEW_FP16
  if (resVT == TYPE_VECTOR_F16)
    // I guess this case NEVER happens
    aType = TYPE_VECTOR_I16;

  // SDValue Index2 = ChangeVectorType(Index, aType, DAG);
  /*
  // We prevent errors in ChangeVectorType()
  SDValue Index2 = ChangeVectorType(Index, aType, DAG,
                                    false, // FillWithZeroes
                                    true); // allowUnsafeChanges
  */
  SDValue Index2;

  if (Index.getOpcode() == ISD::SIGN_EXTEND) {
    // This happens if we have e.g. vector type i32 (or i64)
    Index2 = Index.getOperand(0);
    // Index = Index2;
  } else {
    Index2 = Index;
  }

  /*
  SDValue Index2 = ChangeVectorType(Index, aType, DAG, false, true);

  // We get the following error:
  // <<void llvm::SelectionDAG::ReplaceAllUsesWith(llvm::SDNode*,
  llvm::SDNode*):
  //   Assertion `(!From->hasAnyUseOfValue(i) ||
  //           From->getValueType(i) == To->getValueType(i)) &&
  //           "Cannot use this version of ReplaceAllUsesWith!"' failed.>>
  // The reason is that we change index to have type TYPE_VECTOR_I16, while
  //   masked_gather has type TYPE_VECTOR_I32, and this type difference gives
  the
  //   assertion error.
  */

  // Important: We do NOT use Index2 created above

  LLVM_DEBUG(dbgs() << "  LowerMGATHER(): Index = "; Index->dump(););
  LLVM_DEBUG(dbgs() << "  LowerMGATHER(): Index2 = "; Index2->dump(););
  LLVM_DEBUG(dbgs() << "  LowerMGATHER(): N->getNumOperands() = "
                    << N->getNumOperands() << "\n");
  LLVM_DEBUG(dbgs() << "  LowerMGATHER(): N = "; N->dump();); // << "\n");
  for (int idxOpnd = 0; idxOpnd < N->getNumOperands(); idxOpnd++) {
    LLVM_DEBUG(dbgs() << "  LowerMGATHER(): N->getOperand(" << idxOpnd
                      << ") = ";
               N->getOperand(idxOpnd).dump(););
  }

  if (aType == TYPE_VECTOR_I32) {
    // replaceAddI32UseWithADDVH(aType, Index, DAG);
    replaceAddI32UseWithADDVH(aType, Index2, DAG);
  }

  assert(N->getNumOperands() == 6);
  // The definition of the MaskedGatherSDNode class can be found at
  //   http://llvm.org/doxygen/SelectionDAGNodes_8h_source.html#l02324
  // Important NOTE: we are treating here the machine-independent
  //    masked_gather, which has different parameters than the
  //    machine-SDNode masked_gather node defined in TableGen
  //    (with params specified by constraints defined in SDTMaskedGather).
  //  machine-independent masked_gather looks like:
  //  t21: v8i16,ch = masked_gather<LD256[%B](align=4)> t0, t29, t35,
  //                                                Constant:i64<51>, t32
  //    where:
  //     - 1st param (in this case t0) is chain (this case, EntryToken)
  //     - 2nd param (in this case t29) is passthru (vector)
  //     - 3rd param (in this case t35) is mask (vector)
  //     - 4th param (in this case Constant) is the base pointer (scalar) of the
  //        loads (the origin/reference for the index of the gather)
  //        (the base of GEP, also repeated in LD16[...] symbolically)
  //          NOTE: if it has value TargetConstant:i64<0> then we have
  //              LD16[<unknown>] - this seems to always make llc crash.
  //     - 5th param (in this case t32) is index (vector).
  //     - 6th param is scale.
  //
  //#if 0
  // Important: Here we avoid materializing the passthru operand
  SDValue ct = DAG.getConstant(1, dl, MVT::i64);
  SDValue Ops[] = {
      N->getOperand(0),

      // passthru
      N->getOperand(1),
      // Cycles forever in I-selection: DAG.getUNDEF(TYPE_VECTOR_I16),
      // See http://llvm.org/docs/doxygen/html/classllvm_1_1SelectionDAG.html

      // mask
      N->getOperand(2),

      // base pointer
      // ct,
      N->getOperand(3),

      Index2,
      // Index

      N->getOperand(5)};

  DAG.UpdateNodeOperands(N, Ops);
  //#endif

  LLVM_DEBUG(dbgs() << "  LowerMGATHER(), after update: Op.getNode() = "
                    << Op.getNode() //->dump(CurDAG);
                    << "\n   Op = ";
             Op->dump(&DAG); dbgs() << "\n   N = " << N; dbgs() << "\n   N = ";
             N->dump(&DAG);
             // dbgs() << "\n   Scale = "; Scale->dump(CurDAG);
             dbgs() << "\n   Index.getNode() = " << Index.getNode();
             //<< ", Base.getNode() = " << Base.getNode();
             dbgs() << "\n     Index = "; Index->dump(&DAG);
             //
             dbgs() << "\n     N->getBasePtr() = "; N->getBasePtr()->dump(&DAG);
             //
             dbgs() << "\n   Index2.getNode() = " << Index2.getNode();
             //<< ", Base.getNode() = " << Base.getNode();
             dbgs() << "\n     Index2 = "; Index2->dump(&DAG);
             //
             dbgs() << "\n   Mask.getNode() = " << Mask.getNode();
             //<< ", Base.getNode() = " << Base.getNode();
             dbgs() << "\n     Mask = "; Mask->dump(&DAG);
             //
             dbgs() << "\n   Src.getNode() = " << Src.getNode();
             //<< ", Base.getNode() = " << Base.getNode();
             dbgs() << "\n     Src = "; Src->dump(&DAG);
             //
             /*
             // Not working
             dbgs() << "\n   resVT.SimpleTy = " << ((MVT)resVT).SimpleTy;
             dbgs() << "\n   IndexVT.SimpleTy = " << IndexVT.SimpleTy;
             dbgs() << "\n   MaskVT.SimpleTy = " << MaskVT.SimpleTy;
             */
             //
             dbgs() << "\n");

  /*
  // NOT working
  if (N->getNumValues() > 1) {
      LLVM_DEBUG(dbgs() << "  LowerMGATHER(): calling getMergeValues()\n");
      SDValue RetOps[] = {Op.getValue(0), Op.getValue(1)};
      // NOT working: still gives assertion error after this:
      //   <<Assertion `Results.size() == N->getNumValues() && "Custom lowering
      //               returned the wrong number of results!"' failed.>>
      //   (and modifying LowerOperationWrapper() also does NOT help).
      return DAG.getMergeValues(RetOps, dl);
  }
  */

  LLVM_DEBUG(dbgs() << "Exiting ConnexTargetLowering::LowerMGATHER()\n");

  return Op;
} // End ConnexTargetLowering::LowerMGATHER()

// We only basically implement in LowerMSCATTER() a call to
//   replaceAddI32UseWithADDVH(aType, Index)
SDValue ConnexTargetLowering::LowerMSCATTER(SDValue &Op,
                                            // const ConnexSubtarget &Subtarget,
                                            SelectionDAG &DAG) const {
  LLVM_DEBUG(dbgs() << "Entered ConnexTargetLowering::LowerMSCATTER()\n");

  MaskedScatterSDNode *N = cast<MaskedScatterSDNode>(Op.getNode());

  LLVM_DEBUG(dbgs() << "LowerMSCATTER(): N->getNumOperands() = "
                    << N->getNumOperands() << "\n");
  for (int idxOpnd = 0; idxOpnd < N->getNumOperands(); idxOpnd++) {
    LLVM_DEBUG(dbgs() << "  LowerMSCATTER(): N->getOperand(" << idxOpnd
                      << ") = ";
               N->getOperand(idxOpnd).dump(););
  }
  assert(N->getNumOperands() == 6);

  SDLoc dl(Op);

  // It returns ch for the MSCATTER SDNode: EVT resVT = Op.getSimpleValueType();
  /*
  EVT resVT = Index.getOperand(1);
  LLVM_DEBUG(dbgs() << "LowerMSCATTER(): "
                    << "resVT = " << resVT.getEVTString()
                    << "\n");
  */

  SDValue Index = N->getIndex();
  SDValue Index2;

  LLVM_DEBUG(dbgs() << "  LowerMSCATTER(): Index = "; Index->dump(););

  if (Index.getOpcode() == ISD::SIGN_EXTEND) {
    // This happens if we have e.g. vector type i32 (or i64)
    Index2 = Index.getOperand(0);
    // DAG.ReplaceAllUsesWith(Index.getNode(), Index2.getNode());
  } else {
    Index2 = Index;
  }

  LLVM_DEBUG(dbgs() << "  LowerMSCATTER(): Index3 = "; Index2->dump(););

  SDValue Mask = N->getMask();
  SDValue Src = N->getValue(); // this is actually passthru
  MVT IndexVT = Index.getSimpleValueType();
  MVT MaskVT = Mask.getSimpleValueType();
  EVT SrcVT = Src.getSimpleValueType();
  LLVM_DEBUG(dbgs() << "  LowerMSCATTER(): "
                    << "SrcVT = " << SrcVT.getEVTString() << "\n");

  LLVM_DEBUG(dbgs() << "  LowerMSCATTER(): Op.getNode() = " << Op.getNode();
             dbgs() << "\n   Op = "; Op->dump(&DAG));

  // The index value - is normally i32, and we have to lower it to i16
  MVT aType = (SrcVT == TYPE_VECTOR_I16 ? TYPE_VECTOR_I16 : TYPE_VECTOR_I32);

  // NEW_FP16
  if (SrcVT == TYPE_VECTOR_F16)
    // I guess this case NEVER happens
    aType = TYPE_VECTOR_I16;
  LLVM_DEBUG(dbgs() << "LowerMSCATTER(): "
                    << "aType = " << ((EVT)aType).getEVTString() << "\n");

  if (aType == TYPE_VECTOR_I32) {
    // replaceAddI32UseWithADDVH(aType, Index, DAG);
    replaceAddI32UseWithADDVH(aType, Index2, DAG);
  }

  // The definition of the MaskedScatterSDNode class can be found at ...
  // It's parameters are:
  //   - 1st param is chain
  //   - 2nd param is value (vector) to be written
  //   - 3rd param is mask (vector)
  //   - 4th param is the base pointer (scalar) of the loads
  //      (the origin/reference for the index of the gather)
  //       (the base of GEP, also repeated in LD16[...] symbolically)
  //   - 5th param (in this case t32) is index (vector).
  //   - 6th param is scale.
  SDValue Ops[] = {N->getOperand(0), N->getOperand(1), N->getOperand(2),
                   N->getOperand(3), Index2,           N->getOperand(5)};
  DAG.UpdateNodeOperands(N, Ops);
  LLVM_DEBUG(dbgs() << "  LowerMSCATTER(), after update:";
             dbgs() << "\n   N = " << N; dbgs() << "\n   N = "; N->dump(&DAG););

  LLVM_DEBUG(dbgs() << "Exiting ConnexTargetLowering::LowerMSCATTER()\n");

  return Op;
} // End ConnexTargetLowering::LowerMSCATTER()

#ifdef DO_F16_EMULATION_IN_ISEL_LOWERING

#define MARKER_FOR_EMULATION

extern SDNode *CreateInlineAsmNode(SelectionDAG *CurDAG, std::string asmString,
                                   SDNode *nodeSYM_IMM, SDLoc &DL,
                                   bool specialCase = false);

SDValue ConnexTargetLowering::LowerMUL_F16(SDValue &Op,
                                           SelectionDAG *CurDAG) const {
  SDNode *Node = Op.getNode();

  LLVM_DEBUG(dbgs() << "Entered LowerMUL_F16(): [LATEST] Selecting Node = ";
             Node->dump(CurDAG); dbgs() << "\n");

  SDLoc DL(Node);

  EVT ViaVecTy;
  EVT typeVecNode;

  // EVT ResVecTy = Node->getValueType(1); // 0 is ch (chain)

  LLVM_DEBUG(dbgs() << "LowerMUL_F16(): We are in the case TYPE_VECTOR_F16\n");
  typeVecNode = TYPE_VECTOR_F16;

  SDValue nodeOpSrc1 = Node->getOperand(0);
  SDValue nodeOpSrc2 = Node->getOperand(1);

  LLVM_DEBUG(dbgs() << "LowerMUL_F16(): nodeOpSrc1.getValueType() = "
                    << nodeOpSrc1.getValueType().getEVTString() << "\n");
  LLVM_DEBUG(dbgs() << "LowerMUL_F16(): nodeOpSrc1 = ";
             (nodeOpSrc1.getNode())->dump(); dbgs() << "\n");
  LLVM_DEBUG(dbgs() << "LowerMUL_F16(): nodeOpSrc2.getValueType() = "
                    << nodeOpSrc2.getValueType().getEVTString() << "\n");
  LLVM_DEBUG(dbgs() << "LowerMUL_F16(): nodeOpSrc2 = ";
             (nodeOpSrc2.getNode())->dump(); dbgs() << "\n");
  // assert(nodeOpSrc.getValueType() == TYPE_VECTOR_F16);

  SDNode *nodeOpSrcCast1 = CurDAG->getMachineNode(Connex::NOP_BITCONVERT_WH, DL,
                                                  // The output type of the node
                                                  TYPE_VECTOR_I16,
#ifdef MARKER_FOR_EMULATION
                                                  MVT::Other,
                                                  // It gives a serious error:
                                                  // MVT::Glue,
#else
                                                  MVT::Glue,
#endif
                                                  nodeOpSrc1);

#ifdef MARKER_FOR_EMULATION
  std::string exprStrBegin = "// Starting MUL.f16 emulation ;)";
  SDNode *inlineAsmNodeBegin =
      CreateInlineAsmNode(CurDAG, exprStrBegin, nodeOpSrcCast1, DL);
  LLVM_DEBUG(dbgs() << "LowerMUL_F16(): inlineAsmNodeBegin = ";
             inlineAsmNodeBegin->dump(); dbgs() << "\n");
#endif

  SDNode *nodeOpSrcCast2 = CurDAG->getMachineNode(Connex::NOP_BITCONVERT_WH, DL,
                                                  // The output type of the node
                                                  TYPE_VECTOR_I16, MVT::Other,
                                                  // Important: it gives error:
                                                  // <<Assertion
                                                  //   `N->getNodeId() == -1 &&
                                                  //  "Node already inserted!">>
                                                  // MVT::Glue,
                                                  nodeOpSrc2,
  // chain
#ifdef MARKER_FOR_EMULATION
                                                  SDValue(inlineAsmNodeBegin, 0)
#else
                                                  SDValue(nodeOpSrcCast1, 1)
#endif
  );

/*
// Tested - works well, but a bit complicated and inefficient.
//   BUT a GOOD test for the various issues that can appear in llc
//   (COPY generated by TwoAddressInctruction in WHERE blocks and handled by me
//    in ConnexTargetMachine.cpp, etc)
*/
#include "Select_MULf16_OpincaaCodeGen.h"

#ifdef MARKER_FOR_EMULATION
  std::string exprStrEnd = "// Finishing MUL.f16 emulation ;)";
  SDNode *inlineAsmNodeEnd =
      CreateInlineAsmNode(CurDAG, exprStrEnd, resF16, DL);
  LLVM_DEBUG(dbgs() << "LowerMUL_F16(): inlineAsmNodeEnd = ";
             inlineAsmNodeEnd->dump(); dbgs() << "\n");
#endif

  // End of method - we convert resH (vector of i16) to resW (vector of i32)
  SDNode *resW = CurDAG->getMachineNode(Connex::NOP_BITCONVERT_HW, DL,
                                        typeVecNode, SDValue(resF16, 0),
  // chain edge
#ifdef MARKER_FOR_EMULATION
                                        SDValue(inlineAsmNodeEnd, 0)
#else
                                        SDValue(resF16, 1)
#endif
  );

  LLVM_DEBUG(dbgs() << "LowerMUL_F16(): resW = "; resW->dump(CurDAG);
             dbgs() << "\n");

  return SDValue(resW, 0);
} // End LowerMUL_F16()

SDValue ConnexTargetLowering::LowerADD_F16(SDValue &Op,
                                           SelectionDAG *CurDAG) const {
  SDNode *Node = Op.getNode();

  LLVM_DEBUG(dbgs() << "Entered LowerADD_F16(): [LATEST] Selecting Node = ";
             Node->dump(CurDAG); dbgs() << "\n");

  SDLoc DL(Node);

  EVT ViaVecTy;
  EVT typeVecNode;

  // EVT ResVecTy = Node->getValueType(1); // 0 is ch (chain)

  LLVM_DEBUG(dbgs() << "LowerADD_F16(): We are in the case TYPE_VECTOR_F16\n");
  typeVecNode = TYPE_VECTOR_F16;

  SDValue nodeOpSrc1 = Node->getOperand(0);
  SDValue nodeOpSrc2 = Node->getOperand(1);

  LLVM_DEBUG(dbgs() << "LowerADD_F16(): nodeOpSrc1.getValueType() = "
                    << nodeOpSrc1.getValueType().getEVTString() << "\n");
  LLVM_DEBUG(dbgs() << "LowerADD_F16(): nodeOpSrc1 = ";
             (nodeOpSrc1.getNode())->dump(); dbgs() << "\n");
  LLVM_DEBUG(dbgs() << "LowerADD_F16(): nodeOpSrc2.getValueType() = "
                    << nodeOpSrc2.getValueType().getEVTString() << "\n");
  LLVM_DEBUG(dbgs() << "LowerADD_F16(): nodeOpSrc2 = ";
             (nodeOpSrc2.getNode())->dump(); dbgs() << "\n");
  // assert(nodeOpSrc.getValueType() == TYPE_VECTOR_F16);

  SDNode *nodeOpSrcCast1 = CurDAG->getMachineNode(Connex::NOP_BITCONVERT_WH, DL,
                                                  // The output type of the node
                                                  TYPE_VECTOR_I16,
#ifdef MARKER_FOR_EMULATION
                                                  MVT::Other,
                                                  // It gives error: MVT::Glue,
#else
                                                  MVT::Glue,
#endif
                                                  nodeOpSrc1);

#ifdef MARKER_FOR_EMULATION
  std::string exprStrBegin = "// Starting ADD.f16 emulation ;)";
  SDNode *inlineAsmNodeBegin =
      CreateInlineAsmNode(CurDAG, exprStrBegin, nodeOpSrcCast1, DL);
  LLVM_DEBUG(dbgs() << "LowerADD_F16(): inlineAsmNodeBegin = ";
             inlineAsmNodeBegin->dump(); dbgs() << "\n");
#endif

  SDNode *nodeOpSrcCast2 = CurDAG->getMachineNode(Connex::NOP_BITCONVERT_WH, DL,
                                                  // The output type of the node
                                                  TYPE_VECTOR_I16, MVT::Other,
                                                  // Important: It gives error:
                                                  //  <<Assertion
                                                  //   `N->getNodeId() == -1 &&
                                                  //  "Node already inserted!">>
                                                  // MVT::Glue,
                                                  nodeOpSrc2,
                                                  // chain
#ifdef MARKER_FOR_EMULATION
                                                  SDValue(inlineAsmNodeBegin, 0)
#else
                                                  SDValue(nodeOpSrcCast1, 1)
#endif
  );

// Tested - works well, but a bit complicated and inefficient.
//   BUT a GOOD test for the various issues that can appear in llc
//   (COPY generated by TwoAddressInctruction in WHERE blocks and handled by me
//    in ConnexTargetMachine.cpp, etc)
#include "Select_ADDf16_OpincaaCodeGen.h"

#ifdef MARKER_FOR_EMULATION
  std::string exprStrEnd = "// Finishing ADD.f16 emulation ;)";
  SDNode *inlineAsmNodeEnd =
      CreateInlineAsmNode(CurDAG, exprStrEnd, resF16, DL);
  LLVM_DEBUG(dbgs() << "LowerADD_F16(): inlineAsmNodeEnd = ";
             inlineAsmNodeEnd->dump(); dbgs() << "\n");
#endif

  // End of method - we convert resH (vector of i16) to resW (vector of i32)
  SDNode *resW = CurDAG->getMachineNode(Connex::NOP_BITCONVERT_HW, DL,
                                        typeVecNode, SDValue(resF16, 0),
                                        // chain edge
#ifdef MARKER_FOR_EMULATION
                                        SDValue(inlineAsmNodeEnd, 0)
#else
                                        SDValue(resF16, 1)
#endif
  );
  LLVM_DEBUG(dbgs() << "LowerADD_F16(): resW = "; resW->dump(CurDAG);
             dbgs() << "\n");

  return SDValue(resW, 0);
} // End LowerADD_F16()

SDValue ConnexTargetLowering::LowerREDUCE_F16(SDValue &Op,
                                              SelectionDAG *CurDAG) const {
  SDNode *Node = Op.getNode();

  LLVM_DEBUG(dbgs() << "Entered SelectReduceF16(): Selecting Node = ";
             Node->dump(CurDAG); dbgs() << "\n");

  SDLoc DL(Node);

  EVT ViaVecTy;
  EVT typeVecNode;

  // EVT ResVecTy = Node->getValueType(1); // 0 is ch (chain)

  LLVM_DEBUG(
      dbgs() << "SelectReduceF16(): We are in the case TYPE_VECTOR_F16\n");
  typeVecNode = TYPE_VECTOR_F16;

  // NOTE: Opnd 1 is a ct
  SDValue nodeOpSrc = Node->getOperand(2);

  // We need to preserve the node chained with Node to avoid it is removed
  SDValue nodeOpChain = Node->getOperand(0); // Opnd 0 is ch (chain)

  LLVM_DEBUG(dbgs() << "SelectReduceF16(): nodeOpSrc.getValueType() = "
                    << nodeOpSrc.getValueType().getEVTString() << "\n");
  LLVM_DEBUG(dbgs() << "SelectReduceF16(): nodeOpSrc = ";
             (nodeOpSrc.getNode())->dump(); dbgs() << "\n");
  // assert(nodeOpSrc.getValueType() == TYPE_VECTOR_F16);

#ifdef MARKER_FOR_EMULATION
  SDNode *nodeOpSrcCastBogus1 = CurDAG->getMachineNode(
      Connex::NOP_BITCONVERT_HH, DL, TYPE_VECTOR_I16, MVT::Other,
      // It gives error: MVT::Glue,
      nodeOpSrc,
      // chain edge
      nodeOpChain);

  std::string exprStrBegin = "// Starting RED.f16 emulation ;)";
  SDNode *inlineAsmNodeBegin =
      CreateInlineAsmNode(CurDAG, exprStrBegin, nodeOpSrcCastBogus1, DL);
  LLVM_DEBUG(dbgs() << "SelectReduceF16: inlineAsmNodeBegin = ";
             inlineAsmNodeBegin->dump(); dbgs() << "\n");

  // This node is also bogus, only for the sake of "sandwhiching" the INLINE
  //   assembly with 2 NOPs.
  SDNode *nodeOpSrcCast =
      CurDAG->getMachineNode(Connex::NOP_BITCONVERT_HH,
                             // Important: this is a BOGUS
                             //   NOP_BITCONVERT - we just
                             //   put it since it has a Glue
                             //   result, while
                             //   nodeOpSrcCast2 does NOT
                             DL, TYPE_VECTOR_I16, MVT::Other,
                             // Important: it gives error:
                             //  <<Assertion
                             //     `N->getNodeId() == -1 &&
                             //   "Node already inserted!">>
                             // MVT::Glue,
                             SDValue(nodeOpSrcCastBogus1, 0),
                             // chain
                             SDValue(inlineAsmNodeBegin, 0));
#else
  SDNode *nodeOpSrcCast = CurDAG->getMachineNode(
      Connex::NOP_BITCONVERT_HH, DL, TYPE_VECTOR_I16, MVT::Glue, nodeOpSrc,
      // chain edge
      nodeOpChain);

#endif

  return SDValue();
} // End LowerREDUCE_F16()

#else  // ! DO_F16_EMULATION_IN_ISEL_LOWERING
SDValue ConnexTargetLowering::LowerMUL_F16(SDValue &Op,
                                           SelectionDAG *CurDAG) const {
  return SDValue();
} // End LowerMUL_F16()

SDValue ConnexTargetLowering::LowerADD_F16(SDValue &Op,
                                           SelectionDAG *CurDAG) const {
  return SDValue();
} // End LowerADD_F16()

SDValue ConnexTargetLowering::LowerREDUCE_F16(SDValue &Op,
                                              SelectionDAG *CurDAG) const {
  return SDValue();
} // End LowerREDUCE_F16()
#endif // #ifdef DO_F16_EMULATION_IN_ISEL_LOWERING

/* static */
SDValue ConnexTargetLowering::LowerVSELECT(SDValue &Op,
                                           // const ConnexSubtarget &Subtarget,
                                           SelectionDAG &DAG) const {
  assert(0 && "This code is no longer executed since VSELECT is handled in "
              "ConnexDAGToDAGISel::selectVSELECT().");
} // End LowerVSELECT()

/*
From http://llvm.org/docs/doxygen/html/classllvm_1_1TargetLowering.html:
 virtual SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const
   <<This callback is invoked for operations that are unsupported by the target,
     which are registered to use 'custom' lowering, and whose defined values
     are all legal.>>
 */
// From llvm/include/llvm/CodeGen/ISDOpcodes.h
SDValue ConnexTargetLowering::LowerOperation(SDValue Op,
                                             SelectionDAG &DAG) const {
  // This will print the numeric (decimal) value of the Opcode.
  LLVM_DEBUG(dbgs() << "Entered ConnexTargetLowering::LowerOperation(): "
                    << "Op.getOpcode() = " << Op.getOpcode()
                    << ", getTargetNodeName() = "
                    << getTargetNodeName(Op.getOpcode()) << "\n");

  /*
  LLVM_DEBUG(dbgs() << "ConnexTargetLowering::LowerOperation(): ISD::VSELECT = "
                    << ISD::VSELECT << "\n");
  if (Op.getOpcode() == ISD::VSELECT)
    LLVM_DEBUG(dbgs()
      << "ConnexTargetLowering::LowerOperation() - ISD::VSELECT\n");
  */

  switch (Op.getOpcode()) {
    /* // NEW_BIGGER_OPS
      // TODO: check for the type to be i32/u32
      (Op.getOperand(0).getValueType() == MVT::i32) &&
          (Op.getOperand(1).getValueType() == MVT::i32) {

         // % NOTE: reg alloc is NOT performed yet - but this is
         //       dataflow mostly...
         //
         // Let's do an incorrect, but simpler version:
         // Dest_v32i16 = Src1_v32i16 ADD Src2_v32i16
         // //Reg_dest_low16 = Reg_src1_low16 ADD Reg_src2_low16
         // //Reg_dest_high16 = Reg_src1_high16 ADD Reg_src2_high16
         // Reg_tmp = 1
         // //WHERE INDEX & 1 == 0
         // LDIX (load index of the Processing Element) %to Reg_tmp2
         //  AND 1
         //    == 0
         //      WHERE true
         //        WHERE CARRY
         //           Reg_dest_high16 = Reg_src1_high16 ADD Reg_tmp
         //        END_WHERE
         //      END_WHERE
        return DAG.getNode(ConnexISD::ADD,
                           DL,
                           Op.getValueType(),
                           Chain,
                           LHS,
                           RHS,
                           // TODO_CHANGE_BACKEND:
                           //DAG.getConstant(CC, DL, MVT::i64), Dest);
                           DAG.getConstant(CC, DL, TYPE_SCALAR_ELEMENT), Dest);
      }

      // The Op.getOperand(0).getValueType() == MVT::u32
      // in this
      // return DAG.getNode(ConnexISD::ADD, DL, Op.getValueType(), Chain, LHS,
      //                    RHS,
      //    // TODO_CHANGE_BACKEND:
      //             //DAG.getConstant(CC, DL, MVT::i64), Dest);
      //             DAG.getConstant(CC, DL, TYPE_SCALAR_ELEMENT), Dest);
      return Lower(Op, DAG);
    */

#ifdef DO_F16_EMULATION_IN_ISEL_LOWERING
  // NEW_FP16
  /*
    case ISD::Intrinsic::connex_reduce_f16: {
      LLVM_DEBUG(dbgs()
        << "LowerOperation() for Intrinsic::connex_reduce_f16\n");

      SDLoc DL(Op);
      SDNode *Node = Op.getNode();
      EVT ResVecTy = Node->getValueType(0);
  // MEGA-TODO: input opnd has to have type TYPE_VECTOR_F16
      if (ResVecTy == MVT::f16) {
        LLVM_DEBUG(dbgs()
          << "LowerOperation() for Intrinsic::connex_reduce_f16 for f16\n");
        return LowerREDUCE_F16(Op, &DAG);
      }

      break;
    }
  */

  // HANDLING_F16_IN_ISEL_LOWERING(2018_08_17)
  case ISD::FMUL: {
    LLVM_DEBUG(dbgs() << "LowerOperation() for FMUL\n");

    SDLoc DL(Op);
    SDNode *Node = Op.getNode();
    EVT ResVecTy = Node->getValueType(0);

#ifdef DO_MUL_F16_EMULATION_IN_ISEL_LOWERING
    // if (ResVecTy == MVT::f16)
    if (ResVecTy == TYPE_VECTOR_F16) {
      LLVM_DEBUG(dbgs() << "LowerOperation() for FMUL for f16\n");
      return LowerMUL_F16(Op, &DAG);
    }
#endif

    break;
  }
  case ISD::FADD: {
    LLVM_DEBUG(dbgs() << "LowerOperation() for FADD\n");

    SDLoc DL(Op);
    SDNode *Node = Op.getNode();
    EVT ResVecTy = Node->getValueType(0);

#ifdef DO_ADD_F16_EMULATION_IN_ISEL_LOWERING
    // if (ResVecTy == MVT::f16)
    if (ResVecTy == TYPE_VECTOR_F16) {
      LLVM_DEBUG(dbgs() << "LowerOperation() for FADD for f16\n");
      return LowerADD_F16(Op, &DAG);
      // return DAG.getNode(Connex::ADD_rr,
      //                    DL,
      //                    Op.getValueType(),
      //                    Op.getOperand(1),
      //                    Op.getOperand(2));
    }
#endif

    break;
  }
#endif // #ifdef DO_F16_EMULATION_IN_ISEL_LOWERING

  case ISD::BR_CC:
    return LowerBR_CC(Op, DAG);
  case ISD::GlobalAddress:
    return LowerGlobalAddress(Op, DAG);
  case ISD::SELECT_CC:
    return LowerSELECT_CC(Op, DAG);
  case ISD::INSERT_VECTOR_ELT:
    // Inspired from [LLVM]/llvm/lib/Target/ARM/ARMISelLowering.cpp
    return LowerINSERT_VECTOR_ELT(Op, DAG);
  case ISD::EXTRACT_VECTOR_ELT:
    // From [LLVM]/llvm/lib/Target/Mips/MipsSEISelLowering.cpp
    return LowerEXTRACT_VECTOR_ELT(Op, DAG);
    //  return EXTRACT_VECTOR_ELT;
  case ISD::BUILD_VECTOR:
    // From [LLVM]/llvm/lib/Target/Mips/MipsSEISelLowering.cpp
    return LowerBUILD_VECTOR(Op, DAG);
  case ISD::VECTOR_SHUFFLE:
    // From [LLVM]/llvm/lib/Target/Mips/MipsSEISelLowering.cpp
    return LowerVECTOR_SHUFFLE(Op, DAG);

  // Inspired from lib/Target/AMDGPU/AMDGPUISelLowering.cpp
  case ISD::DYNAMIC_STACKALLOC:
    return LowerDYNAMIC_STACKALLOC(Op, DAG);

  // From [LLVM]/llvm/lib/Target/Mips/MipsISelLowering.cpp
  case ISD::ConstantPool:
    return LowerConstantPool(Op, DAG);

  case ISD::MGATHER:
    // From [LLVM]/llvm/lib/Target/X86/X86ISelLowering.cpp
    return LowerMGATHER(Op, DAG);

  case ISD::MSCATTER:
    // From [LLVM]/llvm/lib/Target/X86/X86ISelLowering.cpp
    return LowerMSCATTER(Op, DAG);

    /* // TREAT_SETCC_VSELECT
    // Inspired From lib/Target/Mips/MipsSEISelLowering.cpp
    static bool isLegalDSPCondCode(EVT Ty, ISD::CondCode CC) {
      bool IsV216 = (Ty == MVT::v2i16);

      switch (CC) {
      case ISD::SETEQ:
      case ISD::SETNE:  return true;
      case ISD::SETLT:
      case ISD::SETLE:
      case ISD::SETGT:
      case ISD::SETGE:  return IsV216;
      case ISD::SETULT:
      case ISD::SETULE:
      case ISD::SETUGT:
      case ISD::SETUGE: return !IsV216;
      default:          return false;
      }
    }

    case ISD::SETCC:
    //static SDValue performSETCCCombine(SDNode *N, SelectionDAG &DAG) {
      SDNode *N = Op.getNode();

      EVT Ty = N->getValueType(0);

      if ((Ty != MVT::v2i16) && (Ty != MVT::v4i8))
        return SDValue();

      if (!isLegalDSPCondCode(Ty,
                              cast<CondCodeSDNode>(N->getOperand(2))->get()))
        return SDValue();

      return DAG.getNode(MipsISD::SETCC_DSP,
                         SDLoc(N),
                         Ty,
                         N->getOperand(0),
                         N->getOperand(1),
                         N->getOperand(2));
    //
    */

  case ISD::VSELECT: {
    // return LowerVSELECT(Op, DAG);
  } // End ISD::VSELECT

  default:
    llvm_unreachable("unimplemented operand");
  }
} // End ConnexTargetLowering::LowerOperation

// Calling Convention Implementation
#include "ConnexGenCallingConv.inc"

// Taken from lib/Target/Mips/MipsISelLowering.cpp
static SDValue UnpackFromArgumentSlot(SDValue Val, const CCValAssign &VA,
                                      EVT ArgVT, const SDLoc &DL,
                                      SelectionDAG &DAG) {
  LLVM_DEBUG(
      dbgs() << "Entered ConnexISelLowering::UnpackFromArgumentSlot()\n");
  MVT LocVT = VA.getLocVT();
  EVT ValVT = VA.getValVT();

  // Shift into the upper bits if necessary.
  switch (VA.getLocInfo()) {
  default:
    break;
  case CCValAssign::AExtUpper:
  case CCValAssign::SExtUpper:
  case CCValAssign::ZExtUpper: {
    unsigned ValSizeInBits = ArgVT.getSizeInBits();
    unsigned LocSizeInBits = VA.getLocVT().getSizeInBits();
    unsigned Opcode =
        VA.getLocInfo() == CCValAssign::ZExtUpper ? ISD::SRL : ISD::SRA;
    Val = DAG.getNode(
        Opcode, DL, VA.getLocVT(), Val,
        DAG.getConstant(LocSizeInBits - ValSizeInBits, DL, VA.getLocVT()));
    break;
  }
  }

  // If this is an value smaller than the argument slot size (32-bit for O32,
  // 64-bit for N32/N64), it has been promoted in some way to the argument slot
  // size. Extract the value and insert any appropriate assertions regarding
  // sign/zero extension.
  switch (VA.getLocInfo()) {
  default:
    llvm_unreachable("Unknown loc info!");
  case CCValAssign::Full:
    break;
  case CCValAssign::AExtUpper:
  case CCValAssign::AExt:
    Val = DAG.getNode(ISD::TRUNCATE, DL, ValVT, Val);
    break;
  case CCValAssign::SExtUpper:
  case CCValAssign::SExt:
    Val = DAG.getNode(ISD::AssertSext, DL, LocVT, Val, DAG.getValueType(ValVT));
    Val = DAG.getNode(ISD::TRUNCATE, DL, ValVT, Val);
    break;
  case CCValAssign::ZExtUpper:
  case CCValAssign::ZExt:
    Val = DAG.getNode(ISD::AssertZext, DL, LocVT, Val, DAG.getValueType(ValVT));
    Val = DAG.getNode(ISD::TRUNCATE, DL, ValVT, Val);
    break;
  case CCValAssign::BCvt:
    Val = DAG.getNode(ISD::BITCAST, DL, ValVT, Val);
    break;
  }

  return Val;
}

/*
void ConnexTargetLowering::writeVarArgRegs(std::vector<SDValue> &OutChains,
                                           SDValue Chain, const SDLoc &DL,
                                           SelectionDAG &DAG,
                                           CCState &State) const {
  ArrayRef<MCPhysReg> ArgRegs = ABI.GetVarArgRegs();
  unsigned Idx = State.getFirstUnallocated(ArgRegs);
  unsigned RegSizeInBytes = Subtarget.getGPRSizeInBytes();
  MVT RegTy = MVT::getIntegerVT(RegSizeInBytes * 8);
  const TargetRegisterClass *RC = getRegClassFor(RegTy);
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MipsFunctionInfo *MipsFI = MF.getInfo<MipsFunctionInfo>();

  // Offset of the first variable argument from stack pointer.
  int VaArgOffset;

  if (ArgRegs.size() == Idx)
    VaArgOffset = alignTo(State.getNextStackOffset(), RegSizeInBytes);
  else {
    VaArgOffset =
        (int)ABI.GetCalleeAllocdArgSizeInBytes(State.getCallingConv()) -
        (int)(RegSizeInBytes * (ArgRegs.size() - Idx));
  }

  // Record the frame index of the first variable argument
  // which is a value necessary to VASTART.
  int FI = MFI->CreateFixedObject(RegSizeInBytes, VaArgOffset, true);
  MipsFI->setVarArgsFrameIndex(FI);

  // Copy the integer registers that have not been used for argument passing
  // to the argument register save area. For O32, the save area is allocated
  // in the caller's stack frame, while for N32/64, it is allocated in the
  // callee's stack frame.
  for (unsigned I = Idx; I < ArgRegs.size();
       ++I, VaArgOffset += RegSizeInBytes) {
    unsigned Reg = addLiveIn(MF, ArgRegs[I], RC);
    SDValue ArgValue = DAG.getCopyFromReg(Chain, DL, Reg, RegTy);
    FI = MFI->CreateFixedObject(RegSizeInBytes, VaArgOffset, true);
    SDValue PtrOff = DAG.getFrameIndex(FI, getPointerTy(DAG.getDataLayout()));
    SDValue Store = DAG.getStore(Chain, DL, ArgValue, PtrOff,
                                 MachinePointerInfo(), false, false, 0);
    cast<StoreSDNode>(Store.getNode())->getMemOperand()->setValue(
        (Value *)nullptr);
    OutChains.push_back(Store);
  }
}
*/

SDValue ConnexTargetLowering::LowerFormalArguments(
    SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &DL,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const {
  LLVM_DEBUG(
      dbgs() << "Entered ConnexTargetLowering::LowerFormalArguments()\n");
  LLVM_DEBUG(dbgs() << "  LowerFormalArguments(): CallConv = " << CallConv
                    << "\n");

  switch (CallConv) {
  default:
    llvm_unreachable("Unsupported calling convention");
  case CallingConv::SPIR_FUNC:
    // We add this since in clang we use target SPIR to support f16
  case CallingConv::C:
  case CallingConv::Fast:
    break;
  }

  // Inspired from lib/Target/Mips/MipsISelLowering.cpp,
  //               MipsTargetLowering::LowerFormalArguments():
  // Used with vargs to acumulate store chains.
  std::vector<SDValue> OutChains;

  MachineFunction &MF = DAG.getMachineFunction();
  MachineRegisterInfo &RegInfo = MF.getRegInfo();

  // Assign locations to all of the incoming arguments.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, IsVarArg, MF, ArgLocs, *DAG.getContext());

  // TODO_CHANGE_BACKEND:
  // CCInfo.AnalyzeFormalArguments(Ins, CC_Connex64);
  CCInfo.AnalyzeFormalArguments(Ins, CC_Connex64);

  unsigned i = 0;
  for (auto &VA : ArgLocs) {
    if (VA.isRegLoc()) {
      LLVM_DEBUG(dbgs() << "LowerFormalArguments(): case VA.isRegLoc()\n");
      // Arguments passed in registers
      EVT RegVT = VA.getLocVT();
      switch (RegVT.getSimpleVT().SimpleTy) {
      default: {
        errs() << "LowerFormalArguments Unhandled argument type: "
               << RegVT.getEVTString() << '\n';
        llvm_unreachable(0);
      }
        // TODO_CHANGE_BACKEND:
      case TYPE_SCALAR_ELEMENT:
        unsigned VReg = RegInfo.createVirtualRegister(&Connex::GPRRegClass);
        RegInfo.addLiveIn(VA.getLocReg(), VReg);
        SDValue ArgValue = DAG.getCopyFromReg(Chain, DL, VReg, RegVT);

        // If this is an 8/16/32-bit value, it is really passed promoted to 64
        // bits. Insert an assert[sz]ext to capture this, then truncate to the
        // right size.
        if (VA.getLocInfo() == CCValAssign::SExt)
          ArgValue = DAG.getNode(ISD::AssertSext, DL, RegVT, ArgValue,
                                 DAG.getValueType(VA.getValVT()));
        else if (VA.getLocInfo() == CCValAssign::ZExt)
          ArgValue = DAG.getNode(ISD::AssertZext, DL, RegVT, ArgValue,
                                 DAG.getValueType(VA.getValVT()));

        if (VA.getLocInfo() != CCValAssign::Full)
          ArgValue = DAG.getNode(ISD::TRUNCATE, DL, VA.getValVT(), ArgValue);

        InVals.push_back(ArgValue);
      }
    } // End VA.isRegLoc()
    else {
      /*
      fail(DL, DAG, "defined with too many args");
      InVals.push_back(DAG.getConstant(0, DL, VA.getLocVT()));
      */

      LLVM_DEBUG(dbgs() << "LowerFormalArguments(): case NOT VA.isRegLoc()\n");

      // Inspired from lib/Target/Mips/MipsISelLowering.cpp,
      //               MipsTargetLowering::LowerFormalArguments():
      MachineFrameInfo &MFI = MF.getFrameInfo();

      MVT LocVT = VA.getLocVT();

      /*
      if (ABI.IsO32()) {
        // We ought to be able to use LocVT directly but O32 sets it to i32
        // when allocating floating point values to integer registers.
        // This shouldn't influence how we load the value into registers unless
        // we are targeting softfloat.
        if (VA.getValVT().isFloatingPoint() && !Subtarget.useSoftFloat())
          LocVT = VA.getValVT();
      }
      */
      // sanity check
      assert(VA.isMemLoc());

      // The stack pointer offset is relative to the caller stack frame.
      int FI = MFI.CreateFixedObject(LocVT.getSizeInBits() / 8,
                                     VA.getLocMemOffset(), true);

      // Create load nodes to retrieve arguments from the stack
      SDValue FIN = DAG.getFrameIndex(FI, getPointerTy(DAG.getDataLayout()));
      SDValue ArgValue = DAG.getLoad(
          LocVT, DL, Chain, FIN,
          MachinePointerInfo::getFixedStack(DAG.getMachineFunction(), FI));
      // false, false, false, 0);
      OutChains.push_back(ArgValue.getValue(1));

      ArgValue = UnpackFromArgumentSlot(ArgValue, VA, Ins[i].ArgVT, DL, DAG);

      InVals.push_back(ArgValue);
    }

    i++;
  }

  /*
  if (IsVarArg || MF.getFunction()->hasStructRetAttr()) {
    fail(DL, DAG, "functions with VarArgs or StructRet are not supported");
  }
  */

  // Inspired from lib/Target/Mips/MipsISelLowering.cpp,
  //               MipsTargetLowering::LowerFormalArguments():
  /*
  ConnexFunctionInfo *MipsFI = MF.getInfo<ConnexFunctionInfo>();
  unsigned e = ArgLocs.size();
  for (i = 0 ; i != e; ++i) {
    // The mips ABIs for returning structs by value requires that we copy
    // the sret argument into $v0 for the return. Save the argument into
    // a virtual register so that we can access it from the return points.
    if (Ins[i].Flags.isSRet()) {
      unsigned Reg = MipsFI->getSRetReturnReg();
      if (!Reg) {
        Reg = MF.getRegInfo().createVirtualRegister(
  // TODO_CHANGE_BACKEND:
            //getRegClassFor(ABI.IsN64() ? MVT::i64 : MVT::i32));
            getRegClassFor(ABI.IsN64() ? MVT::i64 : MVT::i32));
        MipsFI->setSRetReturnReg(Reg);
      }
      SDValue Copy = DAG.getCopyToReg(DAG.getEntryNode(), DL, Reg, InVals[i]);
      Chain = DAG.getNode(ISD::TokenFactor, DL, MVT::Other, Copy, Chain);
      break;
    }
  }

  if (IsVarArg)
    writeVarArgRegs(OutChains, Chain, DL, DAG, CCInfo);
  */

  // All stores are grouped in one node to allow the matching between
  // the size of Ins and InVals. This only happens when on varg functions
  if (!OutChains.empty()) {
    OutChains.push_back(Chain);
    Chain = DAG.getNode(ISD::TokenFactor, DL, MVT::Other, OutChains);
  }

  return Chain;
}

const unsigned ConnexTargetLowering::MaxArgs = 5;

SDValue
ConnexTargetLowering::LowerCall(TargetLowering::CallLoweringInfo &CLI,
                                SmallVectorImpl<SDValue> &InVals) const {
  LLVM_DEBUG(dbgs() << "Entered ConnexISelLowering::LowerCall()\n");
  SelectionDAG &DAG = CLI.DAG;
  auto &Outs = CLI.Outs;
  auto &OutVals = CLI.OutVals;
  auto &Ins = CLI.Ins;
  SDValue Chain = CLI.Chain;
  SDValue Callee = CLI.Callee;
  bool &IsTailCall = CLI.IsTailCall;
  CallingConv::ID CallConv = CLI.CallConv;
  bool IsVarArg = CLI.IsVarArg;
  MachineFunction &MF = DAG.getMachineFunction();

  // Connex target does not support tail call optimization.
  IsTailCall = false;

  switch (CallConv) {
  default:
    report_fatal_error("Unsupported calling convention");
  case CallingConv::Fast:
  case CallingConv::C:
    break;
  }

  // Analyze operands of the call, assigning locations to each operand.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, IsVarArg, MF, ArgLocs, *DAG.getContext());

  CCInfo.AnalyzeCallOperands(Outs, CC_Connex64);

  unsigned NumBytes = CCInfo.getStackSize();

  if (Outs.size() > MaxArgs)
    fail(CLI.DL, DAG, "too many args to ", Callee);

  for (auto &Arg : Outs) {
    ISD::ArgFlagsTy Flags = Arg.Flags;
    if (!Flags.isByVal())
      continue;

    fail(CLI.DL, DAG, "pass by value not supported ", Callee);
  }

  auto PtrVT = getPointerTy(MF.getDataLayout());
  Chain = DAG.getCALLSEQ_START(Chain,
                               // DAG.getConstant(NumBytes, CLI.DL,
                               //                            PtrVT, true),
                               NumBytes, 0, CLI.DL);

  SmallVector<std::pair<unsigned, SDValue>, MaxArgs> RegsToPass;

  // LLVM_DEBUG(dbgs() << "DAG. = "; DAG.dump(); /* << "\n" */);
  LLVM_DEBUG(dbgs() << "DAG = "; DAG.dump(); /* << "\n" */);
  // LLVM_DEBUG(dbgs() << "CLI = " << CLI << "\n");
  LLVM_DEBUG(dbgs() << "InVals.size() = " << InVals.size() << "\n");

  for (unsigned j = 0; j < InVals.size(); ++j) {
    // LLVM_DEBUG(dbgs() << "InVals[j] = " << InVals[j] << "\n");
    LLVM_DEBUG(dbgs() << "InVals[" << j << "] = "; InVals[j]->dump();
               /* << "\n" */);
  }
  LLVM_DEBUG(dbgs() << "ArgLocs.size() = " << ArgLocs.size() << "\n");

  // Walk arg assignments
  for (unsigned i = 0,
                e = std::min(static_cast<unsigned>(ArgLocs.size()), MaxArgs);
       i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];
    SDValue Arg = OutVals[i];

    // LLVM_DEBUG(dbgs() << "ArgLocs[i] = " << ArgLocs[i] << "\n");
    LLVM_DEBUG(dbgs() << "Arg = "; Arg->dump(); /* << "\n" */);

    // Promote the value if needed.
    switch (VA.getLocInfo()) {
    default:
      llvm_unreachable("Unknown loc info");
    case CCValAssign::Full:
      break;
    case CCValAssign::SExt:
      Arg = DAG.getNode(ISD::SIGN_EXTEND, CLI.DL, VA.getLocVT(), Arg);
      break;
    case CCValAssign::ZExt:
      Arg = DAG.getNode(ISD::ZERO_EXTEND, CLI.DL, VA.getLocVT(), Arg);
      break;
    case CCValAssign::AExt:
      Arg = DAG.getNode(ISD::ANY_EXTEND, CLI.DL, VA.getLocVT(), Arg);
      break;
    }

    // Push arguments into RegsToPass vector
    if (VA.isRegLoc())
      RegsToPass.push_back(std::make_pair(VA.getLocReg(), Arg));
    else {
      if (VA.isMemLoc())
        LLVM_DEBUG(dbgs() << "VA.isMemLoc() == true\n");
      llvm_unreachable("call arg pass bug");
    }
  }

  SDValue InFlag;

  // Build a sequence of copy-to-reg nodes chained together with token chain and
  // flag operands which copy the outgoing args into registers.  The InFlag in
  // necessary since all emitted instructions must be stuck together.
  for (auto &Reg : RegsToPass) {
    Chain = DAG.getCopyToReg(Chain, CLI.DL, Reg.first, Reg.second, InFlag);
    InFlag = Chain.getValue(1);
  }

  // If the callee is a GlobalAddress node (quite common, every direct call is)
  // turn it into a TargetGlobalAddress node so that legalize doesn't hack it.
  // Likewise ExternalSymbol -> TargetExternalSymbol.
  if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee))
    Callee = DAG.getTargetGlobalAddress(G->getGlobal(), CLI.DL, PtrVT,
                                        G->getOffset(), 0);
  else if (ExternalSymbolSDNode *E = dyn_cast<ExternalSymbolSDNode>(Callee))
    Callee = DAG.getTargetExternalSymbol(E->getSymbol(), PtrVT, 0);

  // Returns a chain & a flag for retval copy to use.
  SDVTList NodeTys = DAG.getVTList(MVT::Other, MVT::Glue);
  SmallVector<SDValue, 8> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Callee);

  // Add argument registers to the end of the list so that they are
  // known live into the call.
  for (auto &Reg : RegsToPass)
    Ops.push_back(DAG.getRegister(Reg.first, Reg.second.getValueType()));

  if (InFlag.getNode())
    Ops.push_back(InFlag);

  Chain = DAG.getNode(ConnexISD::CALL, CLI.DL, NodeTys, Ops);
  InFlag = Chain.getValue(1);

  // Create the CALLSEQ_END node.
  Chain = DAG.getCALLSEQ_END(
      Chain, DAG.getConstant(NumBytes, CLI.DL, PtrVT, true),
      DAG.getConstant(0, CLI.DL, PtrVT, true), InFlag, CLI.DL);
  InFlag = Chain.getValue(1);

  // Handle result values, copying them out of physregs into vregs that we
  // return.
  return LowerCallResult(Chain, InFlag, CallConv, IsVarArg, Ins, CLI.DL, DAG,
                         InVals);
}

SDValue
ConnexTargetLowering::LowerReturn(SDValue Chain, CallingConv::ID CallConv,
                                  bool IsVarArg,
                                  const SmallVectorImpl<ISD::OutputArg> &Outs,
                                  const SmallVectorImpl<SDValue> &OutVals,
                                  const SDLoc &DL, SelectionDAG &DAG) const {
  unsigned Opc = ConnexISD::RET_FLAG;

  // CCValAssign - represent the assignment of the return value to a location
  SmallVector<CCValAssign, 16> RVLocs;
  MachineFunction &MF = DAG.getMachineFunction();

  // CCState - Info about the registers and stack slot.
  CCState CCInfo(CallConv, IsVarArg, MF, RVLocs, *DAG.getContext());

  if (MF.getFunction().getReturnType()->isAggregateType()) {
    fail(DL, DAG, "only integer returns supported");
    return DAG.getNode(Opc, DL, MVT::Other, Chain);
  }

  // Analize return values.
  CCInfo.AnalyzeReturn(Outs, RetCC_Connex64);

  SDValue Flag;
  SmallVector<SDValue, 4> RetOps(1, Chain);

  // Copy the result values into the output registers.
  for (unsigned i = 0; i != RVLocs.size(); ++i) {
    CCValAssign &VA = RVLocs[i];
    assert(VA.isRegLoc() && "Can only return in registers!");

    Chain = DAG.getCopyToReg(Chain, DL, VA.getLocReg(), OutVals[i], Flag);

    // Guarantee that all emitted copies are stuck together,
    // avoiding something bad.
    Flag = Chain.getValue(1);
    RetOps.push_back(DAG.getRegister(VA.getLocReg(), VA.getLocVT()));
  }

  RetOps[0] = Chain; // Update chain.

  // Add the flag if we have it.
  if (Flag.getNode())
    RetOps.push_back(Flag);

  return DAG.getNode(Opc, DL, MVT::Other, RetOps);
}

SDValue ConnexTargetLowering::LowerCallResult(
    SDValue Chain, SDValue InFlag, CallingConv::ID CallConv, bool IsVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &DL,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const {
  MachineFunction &MF = DAG.getMachineFunction();
  // Assign locations to each value returned by this call.
  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, IsVarArg, MF, RVLocs, *DAG.getContext());

  if (Ins.size() >= 2) {
    fail(DL, DAG, "only small returns supported");
    for (unsigned i = 0, e = Ins.size(); i != e; ++i)
      InVals.push_back(DAG.getConstant(0, DL, Ins[i].VT));
    return DAG.getCopyFromReg(Chain, DL, 1, Ins[0].VT, InFlag).getValue(1);
  }

  CCInfo.AnalyzeCallResult(Ins, RetCC_Connex64);

  // Copy all of the result registers out of their specified physreg.
  for (auto &Val : RVLocs) {
    Chain =
        DAG.getCopyFromReg(Chain, DL, Val.getLocReg(), Val.getValVT(), InFlag)
            .getValue(1);
    InFlag = Chain.getValue(2);
    InVals.push_back(Chain.getValue(0));
  }

  return Chain;
}

static void NegateCC(SDValue &LHS, SDValue &RHS, ISD::CondCode &CC) {
  switch (CC) {
  default:
    break;
  case ISD::SETULT:
  case ISD::SETULE:
  case ISD::SETLT:
  case ISD::SETLE:
    CC = ISD::getSetCCSwappedOperands(CC);
    std::swap(LHS, RHS);
    break;
  }
}

SDValue ConnexTargetLowering::LowerBR_CC(SDValue Op, SelectionDAG &DAG) const {
  SDValue Chain = Op.getOperand(0);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(1))->get();
  SDValue LHS = Op.getOperand(2);
  SDValue RHS = Op.getOperand(3);
  SDValue Dest = Op.getOperand(4);
  SDLoc DL(Op);

  NegateCC(LHS, RHS, CC);

  return DAG.getNode(ConnexISD::BR_CC, DL, Op.getValueType(), Chain, LHS, RHS,
                     // TODO_CHANGE_BACKEND:
                     // DAG.getConstant(CC, DL, MVT::i64), Dest);
                     DAG.getConstant(CC, DL, TYPE_SCALAR_ELEMENT), Dest);
}

SDValue ConnexTargetLowering::LowerSELECT_CC(SDValue Op,
                                             SelectionDAG &DAG) const {
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  SDValue TrueV = Op.getOperand(2);
  SDValue FalseV = Op.getOperand(3);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(4))->get();
  SDLoc DL(Op);

  NegateCC(LHS, RHS, CC);

  // TODO_CHANGE_BACKEND:
  // SDValue TargetCC = DAG.getConstant(CC, DL, MVT::i64);
  SDValue TargetCC = DAG.getConstant(CC, DL, TYPE_SCALAR_ELEMENT);

  SDVTList VTs = DAG.getVTList(Op.getValueType(), MVT::Glue);
  SDValue Ops[] = {LHS, RHS, TargetCC, TrueV, FalseV};

  return DAG.getNode(ConnexISD::SELECT_CC, DL, VTs, Ops);
}

const char *ConnexTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch ((ConnexISD::NodeType)Opcode) {
  case ConnexISD::FIRST_NUMBER:
    break;
  case ConnexISD::RET_FLAG:
    return "ConnexISD::RET_FLAG";
  case ConnexISD::CALL:
    return "ConnexISD::CALL";
  case ConnexISD::SELECT_CC:
    return "ConnexISD::SELECT_CC";
  case ConnexISD::BR_CC:
    return "ConnexISD::BR_CC";
  case ConnexISD::Wrapper:
    return "ConnexISD::Wrapper";
  // Inspired from lib/Target/Mips/MipsISelLowering.cpp
  case ConnexISD::VSHF:
    return "ConnexISD::VSHF";
  /* We should IGNORE gcc -Wswitch when it gives:
      <<warning: case value '...' not in enumerated type
        'llvm::ConnexISD::NodeType'>>
    See definition of NodeType in ConnexISelLowering.h.
  */
  case ISD::MGATHER:
    return "ISD::MGATHER";
    /*
     // Probably not good
     //case ConnexISD::VSELECT:
     // We should IGNORE gcc -Wswitch when it gives:
     //    <<warning: case value '...' not in enumerated type
     //      'llvm::ConnexISD::NodeType'>>
     //  See definition of NodeType in ConnexISelLowering.h.
    */
  case ISD::VSELECT:
    return "ISD::VSELECT";
  /*
  case ConnexISD::ConstantPool:
    return "ConnexISD::ConstantPool";
  */
  default:
    // return TargetLowering::NodeType;
    // See
    //   http://llvm.org/docs/doxygen/html/TargetLowering_8cpp_source.html
    //   - returns nullptr: return TargetLowering::getTargetNodeName(Opcode);
    return "NONAME (getTargetNodeName NOT supporting this Opcode)";
  }

  return nullptr;
}

SDValue ConnexTargetLowering::LowerGlobalAddress(SDValue Op,
                                                 SelectionDAG &DAG) const {
  SDLoc DL(Op);
  const GlobalValue *GV = cast<GlobalAddressSDNode>(Op)->getGlobal();

  // TODO_CHANGE_BACKEND:
  SDValue GA = DAG.getTargetGlobalAddress(GV, DL, TYPE_SCALAR_ELEMENT);

  // TODO_CHANGE_BACKEND:
  return DAG.getNode(ConnexISD::Wrapper, DL, TYPE_SCALAR_ELEMENT, GA);
}

MachineBasicBlock *
ConnexTargetLowering::EmitInstrWithCustomInserter(MachineInstr &MI,
                                                  MachineBasicBlock *BB) const {
  const TargetInstrInfo &TII = *BB->getParent()->getSubtarget().getInstrInfo();
  DebugLoc DL = MI.getDebugLoc();

  assert(MI.getOpcode() == Connex::Select && "Unexpected instr type to insert");

  // To "insert" a SELECT instruction, we actually have to insert the diamond
  // control-flow pattern.  The incoming instruction knows the destination vreg
  // to set, the condition code register to branch on, the true/false values to
  // select between, and a branch opcode to use.
  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  MachineFunction::iterator I = ++BB->getIterator();

  // ThisMBB:
  // ...
  //  TrueVal = ...
  //  jmp_XX r1, r2 goto Copy1MBB
  //  fallthrough --> Copy0MBB
  MachineBasicBlock *ThisMBB = BB;
  MachineFunction *F = BB->getParent();
  MachineBasicBlock *Copy0MBB = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *Copy1MBB = F->CreateMachineBasicBlock(LLVM_BB);

  F->insert(I, Copy0MBB);
  F->insert(I, Copy1MBB);
  // Update machine-CFG edges by transferring all successors of the current
  // block to the new block which will contain the Phi node for the select.
  Copy1MBB->splice(Copy1MBB->begin(), BB,
                   std::next(MachineBasicBlock::iterator(MI)), BB->end());
  Copy1MBB->transferSuccessorsAndUpdatePHIs(BB);
  // Next, add the true and fallthrough blocks as its successors.
  BB->addSuccessor(Copy0MBB);
  BB->addSuccessor(Copy1MBB);

  // Insert Branch if Flag
  unsigned LHS = MI.getOperand(1).getReg();
  unsigned RHS = MI.getOperand(2).getReg();
  int CC = MI.getOperand(3).getImm();
  switch (CC) {
  case ISD::SETGT:
    BuildMI(BB, DL, TII.get(Connex::JSGT_rr))
        .addReg(LHS)
        .addReg(RHS)
        .addMBB(Copy1MBB);
    break;
  case ISD::SETUGT:
    BuildMI(BB, DL, TII.get(Connex::JUGT_rr))
        .addReg(LHS)
        .addReg(RHS)
        .addMBB(Copy1MBB);
    break;
  case ISD::SETGE:
    BuildMI(BB, DL, TII.get(Connex::JSGE_rr))
        .addReg(LHS)
        .addReg(RHS)
        .addMBB(Copy1MBB);
    break;
  case ISD::SETUGE:
    BuildMI(BB, DL, TII.get(Connex::JUGE_rr))
        .addReg(LHS)
        .addReg(RHS)
        .addMBB(Copy1MBB);
    break;
  case ISD::SETEQ:
    BuildMI(BB, DL, TII.get(Connex::JEQ_rr))
        .addReg(LHS)
        .addReg(RHS)
        .addMBB(Copy1MBB);
    break;
  case ISD::SETNE:
    BuildMI(BB, DL, TII.get(Connex::JNE_rr))
        .addReg(LHS)
        .addReg(RHS)
        .addMBB(Copy1MBB);
    break;
  default:
    report_fatal_error("unimplemented select CondCode " + Twine(CC));
  }

  // Copy0MBB:
  //  %FalseValue = ...
  //  # fallthrough to Copy1MBB
  BB = Copy0MBB;

  // Update machine-CFG edges
  BB->addSuccessor(Copy1MBB);

  // Copy1MBB:
  //  %Result = phi [ %FalseValue, Copy0MBB ], [ %TrueValue, ThisMBB ]
  // ...
  BB = Copy1MBB;
  BuildMI(*BB, BB->begin(), DL, TII.get(Connex::PHI), MI.getOperand(0).getReg())
      .addReg(MI.getOperand(5).getReg())
      .addMBB(Copy0MBB)
      .addReg(MI.getOperand(4).getReg())
      .addMBB(ThisMBB);

  MI.eraseFromParent(); // The pseudo instruction is gone now.
  return BB;
}

static bool isIndexVectorVal(const BuildVectorSDNode *N) {
  unsigned int nOps = N->getNumOperands();

  // SDValue Operand0 = N->getOperand(0);

  for (unsigned int i = 0; i < nOps; ++i) {
    // if (N->getOperand(i) != Operand0)
    // See
    //  llvm.org/docs/ProgrammersManual.html#the-isa-cast-and-dyn-cast-templates
    ConstantSDNode *ctNode = dyn_cast<ConstantSDNode>(N->getOperand(i));
    if (ctNode == NULL)
      return false;

    LLVM_DEBUG(dbgs() << "    ctNode = "; ctNode->dump());

    if (N->getConstantOperandVal(i) != i)
      return false;
  }
  /*
  if (Op->getOpcode() == ISD::UNDEF)
    return true;
    if (isConstantOrUndef(Op->getOperand(i)))
      return true;
  */

  return true;
}

// From llvm/lib/Target/Mips/MipsSEISelLowering.cpp
static bool isConstantOrUndef(const SDValue Op) {
  if (Op->getOpcode() == ISD::UNDEF)
    return true;
  if (isa<ConstantSDNode>(Op))
    return true;
  if (isa<ConstantFPSDNode>(Op))
    return true;
  return false;
}

// From llvm/lib/Target/Mips/MipsSEISelLowering.cpp
static bool isConstantOrUndefBUILD_VECTOR(const BuildVectorSDNode *Op) {
  for (unsigned i = 0; i < Op->getNumOperands(); ++i)
    if (isConstantOrUndef(Op->getOperand(i)))
      return true;
  return false;
}

// Getting inspired from lib/Target/X86/X86ISelLowering.cpp
SDValue ConnexTargetLowering::LowerBITCAST(SDValue Op,
                                           SelectionDAG &DAG) const {
  EVT SrcVT = Op.getOperand(0).getSimpleValueType();
  EVT DstVT = Op.getSimpleValueType();

  LLVM_DEBUG(dbgs() << "Entered ConnexTargetLowering::LowerBITCAST(): "
                    << "SrcVT = " << SrcVT.getEVTString() << ", DstVT = "
                    << DstVT.getEVTString() << ". Returning SrcVT... \n");

  // return SDValue();
  // return Op;
  return Op.getOperand(0);
}

SDValue ConnexTargetLowering::LowerADD_I32(SDValue Op,
                                           SelectionDAG &DAG) const {
  // TODO: build opnd0&1 that takes the same operands, but have type
  //       TYPE_VECTOR_I16
  LLVM_DEBUG(dbgs() << "Entered ConnexTargetLowering::LowerADD_I32()\n");

  assert(Op.getOperand(0).getValueType() == TYPE_VECTOR_I32);

  SDValue opnd0 = Op.getOperand(0);
  SDValue opnd1 = Op.getOperand(1);

  // I need to convert the TYPE_VECTOR_I16 vector operand to TYPE_VECTOR_I32.

  SDValue opnd1Native =
      DAG.getNode(ISD::BITCAST, SDLoc(Op), TYPE_VECTOR_I16, opnd0);
  SDValue opnd2Native =
      DAG.getNode(ISD::BITCAST, SDLoc(Op), TYPE_VECTOR_I16, opnd1);

  SDValue Result =
      DAG.getNode(ISD::ADD,
                  // ConnexISD::ADDV_H,
                  SDLoc(Op), TYPE_VECTOR_I16, opnd1Native, opnd2Native);

  LLVM_DEBUG(dbgs() << "LowerADD_I32: UNSPECIFIED case\n");
  return Result; // SDValue();
}

// From [LLVM]/llvm/lib/Target/Mips/MipsSEISelLowering.cpp
// Lowers ISD::BUILD_VECTOR into appropriate SelectionDAG nodes for the
// backend.
//
// Lowers according to the following rules:
// - Constant splats are legal as-is as long as the SplatBitSize is a power of
//   2 less than or equal to 64 and the value fits into a signed 10-bit
//   immediate
// - Constant splats are lowered to bitconverted BUILD_VECTORs if SplatBitSize
//   is a power of 2 less than or equal to 64 and the value does not fit into a
//   signed 10-bit immediate
// - Non-constant splats are legal as-is.
// - Non-constant non-splats are lowered to sequences of INSERT_VECTOR_ELT.
// - All others are illegal and must be expanded.
SDValue ConnexTargetLowering::LowerBUILD_VECTOR(SDValue Op,
                                                SelectionDAG &DAG) const {
  LLVM_DEBUG(dbgs() << "Entered ConnexTargetLowering::LowerBUILD_VECTOR()\n");

  BuildVectorSDNode *BVN = cast<BuildVectorSDNode>(Op);
  EVT ResTy = Op->getValueType(0);
  SDLoc DL(Op);
  APInt SplatValue, SplatUndef;
  unsigned SplatBitSize;
  bool HasAnyUndefs;

  /*
  if (!Subtarget.hasMSA() || !ResTy.is128BitVector())
    return SDValue();
  */

  // From http://llvm.org/docs/doxygen/html/classllvm_1_1BuildVectorSDNode.html:
  //  bool isConstantSplat(APInt &SplatValue, APInt &SplatUndef,
  //                       unsigned &SplatBitSize, bool &HasAnyUndefs,
  //                       unsigned MinSplatBits=0,
  //                       bool isBigEndian=false) const
  //    Check if this is a constant splat, and if so, find the smallest element
  //          size that splats the vector.
  //   By constant splat we understand a vector filled with the same
  //       constant value in all elements.
  if (BVN->isConstantSplat(SplatValue, SplatUndef, SplatBitSize, HasAnyUndefs,
                           8, false) //, true)
      //! Subtarget.isLittle())
      && SplatBitSize <= 64) {
    LLVM_DEBUG(dbgs() << "LowerBUILD_VECTOR(): Case isConstantSplat(): "
                      << "SplatValue = " << SplatValue
                      << ", SplatUndef = " << SplatUndef
                      << ", SplatBitSize = " << SplatBitSize << "\n");
    /*
    LLVM_DEBUG(dbgs() << "LowerBUILD_VECTOR: SplatValue = "
                      << SplatValue.toString(10, 1) << "\n");
    LLVM_DEBUG(dbgs() << "LowerBUILD_VECTOR: SplatUndef = "
                      << SplatUndef.toString(10, 1) << "\n");
    LLVM_DEBUG(dbgs() << "LowerBUILD_VECTOR: SplatBitSize = "
                      << SplatBitSize << "\n");
    */

    // We can only cope with 8 or 16 or 32 (NOT 64, etc) bit elements
    if (SplatBitSize != 8 && SplatBitSize != 16 && SplatBitSize != 32) {
      /* MEGA-TODO: NOT sure this is correct for case vector register is
        TYPE_VECTOR_I32 or TYPE_VECTOR_I16 */
      LLVM_DEBUG(dbgs() << "LowerBUILD_VECTOR: canNOT cope with "
                        << SplatBitSize << " bits.\n");
      return SDValue();
    }

    // If the value fits into a simm10 then we can use ldi.[bhwd]
    // However, if it isn't an integer type we will have to bitcast from an
    // integer type first. Also, if there are any undefs, we must lower them
    // to defined values first.
    if (ResTy.isInteger() && !HasAnyUndefs && SplatValue.isSignedIntN(10)) {
      // See http://llvm.org/docs/doxygen/html/classllvm_1_1SDValue.html
      // LLVM_DEBUG(dbgs() << "LowerBUILD_VECTOR: return Op (Op = "
      //                  << Op << ")\n");
      LLVM_DEBUG(dbgs() << "    LowerBUILD_VECTOR(): Case SIMM10 taken. "
                        << "(Op = ";
                 Op->dump(); dbgs() << ")\n");

      LLVM_DEBUG(dbgs() << "LowerBUILD_VECTOR: return Op\n");
      return Op;

      // TODO We should return as selected instruction VLOAD
    }

    EVT ViaVecTy;

    LLVM_DEBUG(dbgs() << "LowerBUILD_VECTOR: SplatBitSize = " << SplatBitSize
                      << "\n");

    switch (SplatBitSize) {
    default:
      return SDValue();

    // TODO_CHANGE_BACKEND:
    case 8:
      // ViaVecTy = MVT::v16i8;
      ViaVecTy = TYPE_VECTOR_I16;
      break;
    case 16:
      ViaVecTy = TYPE_VECTOR_I16;
      break;
    case 32:
      ViaVecTy = TYPE_VECTOR_I32;
      break;
    case 64:
      ViaVecTy = TYPE_VECTOR_I64;
      /* TODO: NOT sure this is correct for case vector register is
      TYPE_VECTOR_I32 or TYPE_VECTOR_I16 */
      break;
      /*
      // There's no fill.d to fall back on for 64-bit values
      LLVM_DEBUG(dbgs() << "LowerBUILD_VECTOR: case 64 return SDValue.\n");
      return SDValue();
      */
    }

    LLVM_DEBUG(dbgs() << "LowerBUILD_VECTOR: Before DAG.getConstant()\n");
    // SelectionDAG::getConstant will promote SplatValue appropriately.
    SDValue Result = DAG.getConstant(SplatValue, DL, ViaVecTy);
    LLVM_DEBUG(dbgs() << "LowerBUILD_VECTOR: After DAG.getConstant()\n");

    LLVM_DEBUG(dbgs() << "LowerBUILD_VECTOR(): "
                      << "(Result = ";
               Result->dump(); dbgs() << ")\n");

    // See http://llvm.org/docs/doxygen/html/structllvm_1_1EVT.html
    LLVM_DEBUG(dbgs() << "LowerBUILD_VECTOR(): "
                      << "(ResTy = " << ResTy.getEVTString() << ")\n");

    /*
    // Bitcast to the type we originally wanted
    if (ViaVecTy != ResTy)
      Result = DAG.getNode(ISD::BITCAST, SDLoc(BVN), ResTy, Result);
    */

    LLVM_DEBUG(dbgs() << "LowerBUILD_VECTOR: return Result\n");
    return Result;
  } else if (isSplatVector(BVN)) {
    // This is used for splat vectors filled with the same variable
    LLVM_DEBUG(dbgs() << "LowerBUILD_VECTOR: case isSplatVector(BVN)\n");
    return Op;
  } else if (isIndexVectorVal(BVN)) {
    LLVM_DEBUG(dbgs() << "LowerBUILD_VECTOR: case isIndexVectorVal(BVN)\n");

    SDNode *Res = DAG.getMachineNode(Connex::LDIX_H, DL, TYPE_VECTOR_I16
                                     // We add a chain edge
                                     // CurDAG->getEntryNode()
                                     // opChain
    );
    return SDValue(Res, 0);

    // LDIX_H
    // return Op;
  } else
      // This case seems to not have been taken for BUILD_VECTOR from
      // reduction pattern -
      // see
      // Tests/201_LoopVectorize/27_reduce_bugs/isConstantOrUndefBUILD_VECTOR
      if (!isConstantOrUndefBUILD_VECTOR(BVN)) {
    LLVM_DEBUG(
        dbgs()
        << "LowerBUILD_VECTOR: case !isConstantOrUndefBUILD_VECTOR(BVN)\n");

    // Use INSERT_VECTOR_ELT operations rather than expand to stores.
    // The resulting code is the same length as the expansion, but it doesn't
    // use memory operations
    EVT ResTy = BVN->getValueType(0);

    assert(ResTy.isVector());

    return Op; // Not 100% sure it covers all cases
  }

  LLVM_DEBUG(dbgs() << "LowerBUILD_VECTOR: UNSPECIFIED case\n");
  return SDValue();
}

// Inspired from [LLVM]/llvm/lib/Target/ARM/ARMISelLowering.cpp
SDValue ConnexTargetLowering::LowerINSERT_VECTOR_ELT(SDValue Op,
                                                     SelectionDAG &DAG) const {
  LLVM_DEBUG(
      dbgs() << "Entered ConnexTargetLowering::LowerINSERT_VECTOR_ELT().\n");

  /*
  TODO:
  We need to implement INSERT_VECTOR_ELT with:
  WHERE INDEX == lane(op2)
      VLOAD Rdst, ct (op3)
  END_WHERE
  */

  // INSERT_VECTOR_ELT is legal only for immediate indexes.
  SDValue Lane = Op.getOperand(2);
  if (!isa<ConstantSDNode>(Lane))
    return SDValue();

  LLVM_DEBUG(dbgs() << "ConnexTargetLowering::LowerINSERT_VECTOR_ELT(): "
                       "2nd opnd (lane) is ct.\n");

  return Op;
}

/*
ALEX_TO_PROCESS
From /lib/Target/AMDGPU/AMDGPUISelLowering.h
/// This node is for VLIW targets and it is used to represent a vector
  /// that is stored in consecutive registers with the same channel.
  /// For example:
  ///   |X  |Y|Z|W|
  /// T0|v.x| | | |
  /// T1|v.y| | | |
  /// T2|v.z| | | |
  /// T3|v.w| | | |
  BUILD_VERTICAL_VECTOR,


From llvm/lib/Target/AMDGPU/R600ISelLowering.cpp
SDValue R600TargetLowering::LowerINSERT_VECTOR_ELT(SDValue Op,
                                                   SelectionDAG &DAG) const {
  SDLoc DL(Op);
  SDValue Vector = Op.getOperand(0);
  SDValue Value = Op.getOperand(1);
  SDValue Index = Op.getOperand(2);

  if (isa<ConstantSDNode>(Index) ||
      Vector.getOpcode() == AMDGPUISD::BUILD_VERTICAL_VECTOR)
    return Op;

  Vector = vectorToVerticalVector(DAG, Vector);
  SDValue Insert = DAG.getNode(ISD::INSERT_VECTOR_ELT, DL, Op.getValueType(),
                               Vector, Value, Index);
  return vectorToVerticalVector(DAG, Insert);
}
*/

// From [LLVM]/llvm/lib/Target/Mips/MipsSEISelLowering.cpp
// Lower ISD::EXTRACT_VECTOR_ELT into MipsISD::VEXTRACT_SEXT_ELT.
//
// The non-value bits resulting from ISD::EXTRACT_VECTOR_ELT are undefined. We
// choose to sign-extend but we could have equally chosen zero-extend. The
// DAGCombiner will fold any sign/zero extension of the ISD::EXTRACT_VECTOR_ELT
// result into this node later (possibly changing it to a zero-extend in the
// process).
SDValue ConnexTargetLowering::LowerEXTRACT_VECTOR_ELT(SDValue Op,
                                                      SelectionDAG &DAG) const {

  SDLoc DL(Op);
  EVT ResTy = Op->getValueType(0);
  SDValue Op0 = Op->getOperand(0);
  EVT VecTy = Op0->getValueType(0);

  /* TODO : See http://llvm.org/docs/doxygen/html/classllvm_1_1SDValue.html -
             requires to print each components: Type, operation, etc. */
  LLVM_DEBUG(
      dbgs()
      << "Entered ConnexTargetLowering::LowerEXTRACT_VECTOR_ELT(): Op = ");

  return SDValue();
}

// Inspired from llvm/lib/Target/X86/X86ISelLowering.cpp:
//
// ConstantPool, JumpTable, GlobalAddress, and ExternalSymbol are lowered as
// their target countpart wrapped in the X86ISD::Wrapper node. Suppose N is
// one of the above mentioned nodes. It has to be wrapped because otherwise
// Select(N) returns N. So the raw TargetGlobalAddress nodes, etc. can only
// be used to form addressing mode. These wrapped nodes will be selected
// into MOV32ri.
SDValue ConnexTargetLowering::LowerConstantPool(SDValue Op,
                                                SelectionDAG &DAG) const {
  LLVM_DEBUG(dbgs() << "Entered ConnexTargetLowering::LowerConstantPool().\n");

  ConstantPoolSDNode *CP = cast<ConstantPoolSDNode>(Op);

  // In PIC mode (unless we're in RIPRel PIC mode) we add an offset to the
  //   global base reg.

  // unsigned char OpFlag = Subtarget.classifyLocalReference(nullptr);

  /* If we avoid using WrapperKind in DAG.getNode() below then
   *   we end up with an instruction selection error like
    <<Combining: t55: i64 = <<Deleted Node!>> TargetConstantPool:i64<<8 x i64>
             <i64 0, i64 -1, i64 -2, i64 -3, i64 -4, i64 -5, i64 -6, i64 -7>> 0
    llc: llvm/lib/CodeGen/SelectionDAG/DAGCombiner.cpp:1466:
      llvm::SDValue {anonymous}::DAGCombiner::combine(llvm::SDNode*):
      Assertion `N->getOpcode() != ISD::DELETED_NODE &&
      "Node was deleted but visit returned NULL!"' failed.>>
    (see ~/LLVM/Tests/201_LoopVectorize/25_GOOD_map/NEW/6/UF_1/NEW/STDerr31 )
  */
  unsigned WrapperKind = ConnexISD::Wrapper;

  // CodeModel::Model M = DAG.getTarget().getCodeModel();

  auto PtrVT = getPointerTy(DAG.getDataLayout());
  // See http://llvm.org/docs/doxygen/html/classllvm_1_1SelectionDAG.html
  SDValue Result = DAG.getTargetConstantPool(CP->getConstVal(), PtrVT,
                                             CP->getAlign(), CP->getOffset(),
                                             // OpFlag);
                                             0);
  SDLoc DL(CP);
  Result = DAG.getNode(WrapperKind,
                       // 0,
                       DL, PtrVT, Result);

  return Result;
}

// Inspired from [LLVM]/llvm/lib/Target/Mips/MipsSEISelLowering.h
// Lower VECTOR_SHUFFLE into one of a number of instructions depending on the
//     indices in the shuffle.
//
//  Very Important: Here, in ISelLowering the DAG Combiner changes
//    (I think in all cases) the vector_shuffle SDNode into a BUILD_VECTOR.
//    So we have to identify it here, before the DAG Combiner changes it and
//      replace it with the equivalent Connex instructions.
//   In fact, the DAG Combiner combines, if possible, a few vector_shuffles
//   into only one - I personally find it annoying, without any real benefit...
SDValue ConnexTargetLowering::LowerVECTOR_SHUFFLE(SDValue Op,
                                                  SelectionDAG &DAG) const {
  LLVM_DEBUG(dbgs() << "Entered ConnexTargetLowering::LowerVECTOR_SHUFFLE()\n");
  LLVM_DEBUG(dbgs() << "  Op = "; (Op.getNode())->dump());

  // return SDValue();

  EVT ResTy = Op->getValueType(0);

  // See http://llvm.org/doxygen/SelectionDAGNodes_8h_source.html#l01432
  ShuffleVectorSDNode *SVN = dyn_cast<ShuffleVectorSDNode>(Op);
  assert(SVN != NULL);

  unsigned int numElemsMask = SVN->getValueType(0).getVectorNumElements();
  assert(numElemsMask == CONNEX_VECTOR_LENGTH);

  int mask[CONNEX_VECTOR_LENGTH];
  for (unsigned int i = 0; i < numElemsMask; ++i) {
    mask[i] = SVN->getMaskElt(i);
    LLVM_DEBUG(dbgs() << "    mask[" << i << "] = " << mask[i] << "\n");
  }

  if (mask[0] == 0) {
    // It seems we have no shifting
  } else {
    // It seems we have shifting by constant delta
    int delta = mask[0];

    bool shiftByDelta = true;
    // Checking if we really have shifting by delta
    int i;
    for (i = 0; i < numElemsMask - delta; ++i) {
      // MEGA-TODO: we should also check that we have delta-shift w.r.t. the
      //            2nd data vector operand: if (mask[i] != CVL + i + delta)
      if (mask[i] != i + delta) {
        shiftByDelta = false;
        break;
      }
    }
    LLVM_DEBUG(dbgs() << "    shiftByDelta = " << shiftByDelta << "\n");

    bool circularShiftByDelta = false;
    if (shiftByDelta == true) {
      circularShiftByDelta = true;
      for (i = numElemsMask - delta; i < numElemsMask; ++i) {
        // MEGA-TODO: we should also check that we have circular-delta-shift
        //            w.r.t. the 2nd data vector operand
        if (mask[i] != i + delta) {
          circularShiftByDelta = false;
          break;
        }
      }
    }
    LLVM_DEBUG(dbgs() << "    circularShiftByDelta = " << circularShiftByDelta
                      << "\n");

    bool assignPartOf2ndOpnd = true;
    if (assignPartOf2ndOpnd == false) {
      for (i = numElemsMask - delta; i < numElemsMask; ++i) {
        if (mask[i] == CONNEX_VECTOR_LENGTH + i + delta) {
          assignPartOf2ndOpnd = false;
          break;
        }
      }
    }
    LLVM_DEBUG(dbgs() << "    assignPartOf2ndOpnd = " << assignPartOf2ndOpnd
                      << "\n");

    MachineFunction &MF = DAG.getMachineFunction();
    MachineRegisterInfo &RegInfo = MF.getRegInfo();

    SDLoc DL(Op);
    SDValue svnOp0 = SVN->getOperand(0);
    SDValue svnOp1 = SVN->getOperand(1);
    SDNode *ldSh; // Def required here
    //

    if (circularShiftByDelta || shiftByDelta) {
      SDValue chain = DAG.getEntryNode();

      SDValue ctDelta = DAG.getConstant(delta, DL, MVT::i16, true, false);
      SDNode *vloadDelta = DAG.getMachineNode(
          Connex::VLOAD_H, DL, TYPE_VECTOR_I16, MVT::Glue, ctDelta
          // Glue (or chain) input edge
          // TODO maybe: SDValue(ldIx, 1)
      );

      SDNode *cellShl =
          DAG.getMachineNode(Connex::CELLSHL_H, DL,
                             // NO return type
                             MVT::Glue, svnOp0, SDValue(vloadDelta, 0),
                             // The glue input edge
                             SDValue(vloadDelta, 1));

      // MEGA-TODO: put delta NOPs
      SDValue ct1 = DAG.getConstant(1 /* Num of cycles to NOP */, DL, MVT::i16,
                                    true, false);
      SDNode *nop = DAG.getMachineNode(Connex::NOP_BPF, DL, MVT::Glue, ct1,
                                       // Glue/chain edge
                                       SDValue(cellShl, 0));

      ldSh = DAG.getMachineNode(Connex::LDSH_H, DL,
                                // Return type
                                TYPE_VECTOR_I16, MVT::Glue,
                                // The glue output port of predecessor
                                SDValue(nop, 0));
    } // End if (circularShiftByDelta || shiftByDelta)

    /* // BUGGY_DUE_TO_DAG_COMBINER
    unsigned virtReg = RegInfo.createVirtualRegister(&Connex::VectorHRegClass);
    // Very Important:
    //  From http://llvm.org/docs/doxygen/html/classllvm_1_1SelectionDAG.html:
    //   SDValue getCopyToReg(SDValue Chain, SDLoc dl,
    //                        unsigned Reg,
    //                        SDValue N,
    //                        SDValue Glue)
    SDValue copyToReg = DAG.getCopyToReg(
                                     // Very Important: Chain input edge
                                     (circularShiftByDelta || shiftByDelta) ?
                                        SDValue(ldSh, 1) :
                                        DAG.getEntryNode(),

                                     DL,
                                     virtReg,

                                     // Value copied to register
                                     (circularShiftByDelta || shiftByDelta) ?
                                          SDValue(ldSh, 0) : svnOp1,
                                     // Very Important: Glue input edge
                                     (circularShiftByDelta || shiftByDelta) ?
                                        SDValue(ldSh, 1) :
                                        DAG.getEntryNode()
                                          // Hope this passes as a glue
                                    );
    LLVM_DEBUG(dbgs() << "  copyToReg = ";
               (copyToReg.getNode())->dump());
    */

    SDNode *endWhere; // Definition required

    if (assignPartOf2ndOpnd) {
      SDNode *ldIx = DAG.getMachineNode(
          Connex::LDIX_H, DL, TYPE_VECTOR_I16, MVT::Glue,
          // We add a chain edge
          (circularShiftByDelta || shiftByDelta) ? SDValue(ldSh, 1)
                                                 : DAG.getEntryNode());

      SDValue ctCVLDelta = DAG.getConstant(CONNEX_VECTOR_LENGTH - delta, DL,
                                           MVT::i16, true, false);
      SDNode *vloadCVLDelta = DAG.getMachineNode(
          Connex::VLOAD_H, DL, TYPE_VECTOR_I16, MVT::Glue, ctCVLDelta,
          // Glue (or chain) input edge
          SDValue(ldIx, 1));

      SDNode *lt =
          DAG.getMachineNode(Connex::LT_H, DL, TYPE_VECTOR_I16, MVT::Glue,
                             SDValue(ldIx, 0), SDValue(vloadCVLDelta, 0),
                             // Glue (or chain) input edge
                             SDValue(vloadCVLDelta, 1));

      SDValue ct1 = DAG.getConstant(1 /* Num of cycles to NOP */, DL, MVT::i16,
                                    true, false);
      SDNode *nop = DAG.getMachineNode(Connex::NOP_BPF, DL, MVT::Glue, ct1,
                                       // Glue/chain edge
                                       SDValue(lt, 1));

      SDNode *whereLt =
          DAG.getMachineNode(Connex::WHERELT, //_BUNDLE_H,
                             DL,
                             // Return type
                             TYPE_VECTOR_I16, MVT::Glue, SDValue(lt, 0),
                             // svnOp1,
                             // The glue output port of CopyToReg.
                             SDValue(nop, 0));

      SDValue ct0 = DAG.getConstant(0, DL, MVT::i16, true, false);
      SDNode *ishl =
          DAG.getMachineNode(Connex::ISHLV_SPECIAL_H, DL,
                             DAG.getVTList(TYPE_VECTOR_I16, MVT::Glue),
                             {svnOp1, ct0,
#ifdef BUGGY_DUE_TO_DAG_COMBINER
                              DAG.getRegister(virtReg, TYPE_VECTOR_I16),
#else
                              (circularShiftByDelta || shiftByDelta)
                                  ? SDValue(ldSh, 0)
                                  : svnOp1,
#endif
                              // Glue (or chain) input edge
                              SDValue(whereLt, 1)});
      endWhere = DAG.getMachineNode(Connex::END_WHERE, DL, TYPE_VECTOR_I16,
                                    MVT::Glue, SDValue(ishl, 0),
                                    // Glue (or chain) input edge
                                    SDValue(ishl, 1));
    } // End if (assignPartOf2ndOpnd)

    if (assignPartOf2ndOpnd)
      DAG.ReplaceAllUsesWith(SVN, endWhere);
    else if (circularShiftByDelta || shiftByDelta)
      DAG.ReplaceAllUsesWith(SVN, ldSh);
  }

  return SDValue();

  /*
  ShuffleVectorSDNode *N = SVN;
  unsigned int nOps = N->getNumOperands();
  for (unsigned int i = 0; i < nOps; ++i) {
    // See
    //  llvm.org/docs/ProgrammersManual.html#the-isa-cast-and-dyn-cast-templates
    ConstantSDNode *ctNode = dyn_cast<ConstantSDNode>(N->getOperand(i));
    LLVM_DEBUG(dbgs() << "    ctNode = " << ctNode << "\n");
    if (ctNode == NULL)
      continue; //return false;

    LLVM_DEBUG(dbgs() << "    *ctNode = "; ctNode->dump());

    //if (N->getConstantOperandVal(i) != i)
    //  return false;
  }
  */

  // MEGA-TODO: check for delta..CVL-delta, 2CVL-delta.. 2CVL-1
  // MEGA-TODO: else if BVN is 0..x x + CVL + 1 .. 2CVL-1

  /* TODO: here it was cycling forever in reduction
     loop code - see ~/LLVM/Tests/201_LoopVectorize/27_reduce/OLD/STDerr_old15
     for exact details. */

  // Note: HexagonISelLowering.cpp has also method LowerVECTOR_SHIFT()

  /* // NOT_ORIGINAL_CODE
  // From MipsISelLowering.cpp
  ShuffleVectorSDNode *Node = cast<ShuffleVectorSDNode>(Op);

  int ResTyNumElts = ResTy.getVectorNumElements();
  SmallVector<int, 16> Indices;

  for (int i = 0; i < ResTyNumElts; ++i)
    Indices.push_back(Node->getMaskElt(i));

  // splati.[bhwd] is preferable to the others but is matched from
  // MipsISD::VSHF.
  if (isVECTOR_SHUFFLE_SPLATI(Op, ResTy, Indices, DAG))
    return lowerVECTOR_SHUFFLE_VSHF(Op, ResTy, Indices, DAG);
  SDValue Result = lowerVECTOR_SHUFFLE_ILVEV(Op, ResTy, Indices, DAG);
  if (Result.getNode())
    return Result;
  Result = lowerVECTOR_SHUFFLE_ILVOD(Op, ResTy, Indices, DAG);
  if (Result.getNode())
    return Result;
  Result = lowerVECTOR_SHUFFLE_ILVL(Op, ResTy, Indices, DAG);
  if (Result.getNode())
    return Result;
  Result = lowerVECTOR_SHUFFLE_ILVR(Op, ResTy, Indices, DAG);
  if (Result.getNode())
    return Result;
  Result = lowerVECTOR_SHUFFLE_PCKEV(Op, ResTy, Indices, DAG);
  if (Result.getNode())
    return Result;
  Result = lowerVECTOR_SHUFFLE_PCKOD(Op, ResTy, Indices, DAG);
  if (Result.getNode())
    return Result;
  Result = lowerVECTOR_SHUFFLE_SHF(Op, ResTy, Indices, DAG);
  if (Result.getNode())
    return Result;
  return lowerVECTOR_SHUFFLE_VSHF(Op, ResTy, Indices, DAG);
  */
} // end ConnexTargetLowering::LowerVECTOR_SHUFFLE()

// From http://llvm.org/docs/doxygen/html/classllvm_1_1TargetLoweringBase.html:
// virtual EVT getSetCCResultType(const DataLayout &DL, LLVMContext &Context,
//                                EVT VT) const
//    Return the ValueType of the result of SETCC operations.
// See also http://llvm.org/doxygen/TargetLoweringBase_8cpp_source.html
//  This code fixes the issue with type legalization of vector type:
//   Reported in llvm-dev thread:
//     http://lists.llvm.org/pipermail/llvm-dev/2016-June/100719.html
EVT ConnexTargetLowering::getSetCCResultType(const DataLayout &DL,
                                             LLVMContext &Ctx, EVT VT) const {
  EVT res;

  LLVM_DEBUG(
      dbgs() << "Entered ConnexTargetLowering::getSetCCResultType().\n"
             << "  VT = "
             // See http://llvm.org/docs/doxygen/html/structllvm_1_1EVT.html
             << VT.getEVTString() << "  [END]\n");

  if (VT.isVector()) {
    LLVM_DEBUG(dbgs() << "getSetCCResultType(): "
                      << "VT.getVectorNumElements() = "
                      << VT.getVectorNumElements() << "\n");
    // From llvm/lib/Target/NVPTX/NVPTXISelLowering.h
    // res = EVT::getVectorVT(Ctx, MVT::i1, VT.getVectorNumElements());

    // From llvm/lib/Target/NVPTX/MipsISelLowering.h
    res = VT.changeVectorElementTypeToInteger();

    LLVM_DEBUG(
        dbgs() << "getSetCCResultType(), case VT.isVector(): res = "
               // See http://llvm.org/docs/doxygen/html/structllvm_1_1EVT.html
               << res.getEVTString() << "  [END]\n");

    return res;
  }

  res = getPointerTy(DL).SimpleTy;

  LLVM_DEBUG(dbgs() << "getSetCCResultType(): res = "
                    // See llvm.org/docs/doxygen/html/structllvm_1_1EVT.html
                    << res.getEVTString() << "  [END]\n");

  // Using the code from lib/CodeGen/TargetLoweringBase.cpp
  return res;

  /*
  // This was the original code from llvm/lib/Target/NVPTX/NVPTXISelLowering.h
  Cycles forever - see !!!!
  return MVT::i1;
  */

  /* Messes up 25_Map (for types i16 or i32), etc:
     llc gives assertion error:
      llc: lib/CodeGen/SelectionDAG/SelectionDAG.cpp:3116:
        llvm::SDValue llvm::SelectionDAG::getNode(unsigned int,
                                                  const llvm::SDLoc&,
                                                  llvm::EVT, llvm::SDValue):
          Assertion `VT.isInteger() && Operand.getValueType().isInteger() &&
                    "Invalid ZERO_EXTEND!"' failed.
  //return VT;
  */
}
/*
lib/Target/PowerPC/PPCISelLowering.cpp
EVT PPCTargetLowering::getSetCCResultType(const DataLayout &DL, LLVMContext &C,
                                          EVT VT) const {
  if (!VT.isVector())
    return Subtarget.useCRBits() ? MVT::i1 : MVT::i32;

  if (Subtarget.hasQPX())
    return EVT::getVectorVT(C, MVT::i1, VT.getVectorNumElements());

  return VT.changeVectorElementTypeToInteger();
}
*/
