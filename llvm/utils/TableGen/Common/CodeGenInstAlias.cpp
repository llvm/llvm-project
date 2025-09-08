//===- CodeGenInstAlias.cpp - CodeGen InstAlias Class Wrapper -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the CodeGenInstAlias class.
//
//===----------------------------------------------------------------------===//

#include "CodeGenInstAlias.h"
#include "CodeGenInstruction.h"
#include "CodeGenRegisters.h"
#include "CodeGenTarget.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

using namespace llvm;

unsigned CodeGenInstAlias::ResultOperand::getMINumOperands() const {
  if (!isRecord())
    return 1;

  const Record *Rec = getRecord();
  if (!Rec->isSubClassOf("Operand"))
    return 1;

  const DagInit *MIOpInfo = Rec->getValueAsDag("MIOperandInfo");
  if (MIOpInfo->getNumArgs() == 0) {
    // Unspecified, so it defaults to 1
    return 1;
  }

  return MIOpInfo->getNumArgs();
}

static const Record *getInitValueAsRegClass(const Init *V) {
  if (const auto *VDefInit = dyn_cast<DefInit>(V)) {
    const Record *R = VDefInit->getDef();
    if (R->isSubClassOf("RegisterClass"))
      return R;
    if (R->isSubClassOf("RegisterOperand"))
      return R->getValueAsDef("RegClass");
  }
  return nullptr;
}

using ResultOperand = CodeGenInstAlias::ResultOperand;

static Expected<ResultOperand> matchSimpleOperand(const Init *Arg,
                                                  const StringInit *ArgName,
                                                  const Record *Op,
                                                  const CodeGenTarget &T) {
  if (Op->isSubClassOf("RegisterClass") ||
      Op->isSubClassOf("RegisterOperand")) {
    const Record *OpRC =
        Op->isSubClassOf("RegisterClass") ? Op : Op->getValueAsDef("RegClass");

    if (const auto *ArgDef = dyn_cast<DefInit>(Arg)) {
      const Record *ArgRec = ArgDef->getDef();

      // Match 'RegClass:$name' or 'RegOp:$name'.
      if (const Record *ArgRC = getInitValueAsRegClass(Arg)) {
        if (!T.getRegisterClass(OpRC).hasSubClass(&T.getRegisterClass(ArgRC)))
          return createStringError(
              "argument register class" + ArgRC->getName() +
              " is not a subclass of operand register class " +
              OpRC->getName());
        if (!ArgName)
          return createStringError("register class argument must have a name");
        return ResultOperand::createRecord(ArgName->getAsUnquotedString(),
                                           ArgRec);
      }

      // Match 'Reg'.
      if (ArgRec->isSubClassOf("Register")) {
        if (!T.getRegisterClass(OpRC).contains(T.getRegBank().getReg(ArgRec)))
          return createStringError(
              "register argument " + ArgRec->getName() +
              " is not a member of operand register class " + OpRC->getName());
        if (ArgName)
          return createStringError("register argument must not have a name");
        return ResultOperand::createRegister(ArgRec);
      }

      // Match 'zero_reg'.
      if (ArgRec->getName() == "zero_reg") {
        if (ArgName)
          return createStringError("register argument must not have a name");
        return ResultOperand::createRegister(nullptr);
      }
    }

    return createStringError("argument must be a subclass of RegisterClass, "
                             "RegisterOperand, or zero_reg");
  }

  if (Op->isSubClassOf("Operand")) {
    // Match integer or bits.
    if (const auto *ArgInt = dyn_cast_or_null<IntInit>(
            Arg->convertInitializerTo(IntRecTy::get(Arg->getRecordKeeper())))) {
      if (ArgName)
        return createStringError("integer argument must not have a name");
      return ResultOperand::createImmediate(ArgInt->getValue());
    }

    // Match a subclass of Operand.
    if (const auto *ArgDef = dyn_cast<DefInit>(Arg);
        ArgDef && ArgDef->getDef()->isSubClassOf("Operand")) {
      if (!ArgName)
        return createStringError("argument must have a name");
      return ResultOperand::createRecord(ArgName->getAsUnquotedString(),
                                         ArgDef->getDef());
    }

    return createStringError("argument must be a subclass of Operand");
  }

  llvm_unreachable("Unknown operand kind");
}

static Expected<ResultOperand> matchComplexOperand(const Init *Arg,
                                                   const StringInit *ArgName,
                                                   const Record *Op) {
  assert(Op->isSubClassOf("Operand"));
  const auto *ArgDef = dyn_cast<DefInit>(Arg);
  if (!ArgDef || !ArgDef->getDef()->isSubClassOf("Operand"))
    return createStringError("argument must be a subclass of Operand");
  if (!ArgName)
    return createStringError("argument must have a name");
  return ResultOperand::createRecord(ArgName->getAsUnquotedString(),
                                     ArgDef->getDef());
}

CodeGenInstAlias::CodeGenInstAlias(const Record *R, const CodeGenTarget &T)
    : TheDef(R) {
  Result = R->getValueAsDag("ResultInst");
  AsmString = R->getValueAsString("AsmString");

  // Verify that the root of the result is an instruction.
  const DefInit *DI = dyn_cast<DefInit>(Result->getOperator());
  if (!DI || !DI->getDef()->isSubClassOf("Instruction"))
    PrintFatalError(R->getLoc(),
                    "result of inst alias should be an instruction");

  ResultInst = &T.getInstruction(DI->getDef());

  // NameClass - If argument names are repeated, we need to verify they have
  // the same class.
  StringMap<const Record *> NameClass;
  for (unsigned i = 0, e = Result->getNumArgs(); i != e; ++i) {
    const DefInit *ADI = dyn_cast<DefInit>(Result->getArg(i));
    if (!ADI || !Result->getArgName(i))
      continue;
    // Verify we don't have something like: (someinst GR16:$foo, GR32:$foo)
    // $foo can exist multiple times in the result list, but it must have the
    // same type.
    const Record *&Entry = NameClass[Result->getArgNameStr(i)];
    if (Entry && Entry != ADI->getDef())
      PrintFatalError(R->getLoc(), "result value $" + Result->getArgNameStr(i) +
                                       " is both " + Entry->getName() +
                                       " and " + ADI->getDef()->getName() +
                                       "!");
    Entry = ADI->getDef();
  }

  // Decode and validate the arguments of the result.
  unsigned ArgIdx = 0;
  for (auto [OpIdx, OpInfo] : enumerate(ResultInst->Operands)) {
    // Tied registers don't have an entry in the result dag unless they're part
    // of a complex operand, in which case we include them anyways, as we
    // don't have any other way to specify the whole operand.
    if (OpInfo.MINumOperands == 1 && OpInfo.getTiedRegister() != -1) {
      // Tied operands of different RegisterClass should be explicit within an
      // instruction's syntax and so cannot be skipped.
      int TiedOpNum = OpInfo.getTiedRegister();
      if (OpInfo.Rec->getName() ==
          ResultInst->Operands[TiedOpNum].Rec->getName())
        continue;
    }

    if (ArgIdx >= Result->getNumArgs())
      PrintFatalError(R->getLoc(), "not enough arguments for instruction!");

    const Record *Op = OpInfo.Rec;
    if (Op->isSubClassOf("Operand") && !OpInfo.MIOperandInfo->arg_empty()) {
      // Complex operand (a subclass of Operand with non-empty MIOperandInfo).
      // The argument can be a DAG or a subclass of Operand.
      if (auto *ArgDag = dyn_cast<DagInit>(Result->getArg(ArgIdx))) {
        // The argument is a DAG. The operator must be the name of the operand.
        if (auto *Operator = dyn_cast<DefInit>(ArgDag->getOperator());
            !Operator || Operator->getDef()->getName() != Op->getName())
          PrintFatalError(R, "argument #" + Twine(ArgIdx) +
                                 " operator must be " + Op->getName());
        // The number of sub-arguments and the number of sub-operands
        // must match exactly.
        unsigned NumSubOps = OpInfo.MIOperandInfo->getNumArgs();
        unsigned NumSubArgs = ArgDag->getNumArgs();
        if (NumSubArgs != NumSubOps)
          PrintFatalError(R, "argument #" + Twine(ArgIdx) +
                                 " must have exactly " + Twine(NumSubOps) +
                                 " sub-arguments");
        // Match sub-operands individually.
        for (unsigned SubOpIdx = 0; SubOpIdx != NumSubOps; ++SubOpIdx) {
          const Record *SubOp =
              cast<DefInit>(OpInfo.MIOperandInfo->getArg(SubOpIdx))->getDef();
          Expected<ResultOperand> ResOpOrErr = matchSimpleOperand(
              ArgDag->getArg(SubOpIdx), ArgDag->getArgName(SubOpIdx), SubOp, T);
          if (!ResOpOrErr)
            PrintFatalError(R, "in argument #" + Twine(ArgIdx) + "." +
                                   Twine(SubOpIdx) + ": " +
                                   toString(ResOpOrErr.takeError()));
          ResultOperands.push_back(*ResOpOrErr);
          ResultInstOperandIndex.emplace_back(OpIdx, SubOpIdx);
        }
      } else {
        // Match complex operand as a whole.
        Expected<ResultOperand> ResOpOrErr = matchComplexOperand(
            Result->getArg(ArgIdx), Result->getArgName(ArgIdx), Op);
        if (!ResOpOrErr)
          PrintFatalError(R, "in argument #" + Twine(ArgIdx) + ": " +
                                 toString(ResOpOrErr.takeError()));
        ResultOperands.push_back(*ResOpOrErr);
        ResultInstOperandIndex.emplace_back(OpIdx, -1);
      }
    } else {
      // Simple operand (RegisterClass, RegisterOperand or Operand with empty
      // MIOperandInfo).
      Expected<ResultOperand> ResOpOrErr = matchSimpleOperand(
          Result->getArg(ArgIdx), Result->getArgName(ArgIdx), Op, T);
      if (!ResOpOrErr)
        PrintFatalError(R, "in argument #" + Twine(ArgIdx) + ": " +
                               toString(ResOpOrErr.takeError()));
      ResultOperands.push_back(*ResOpOrErr);
      ResultInstOperandIndex.emplace_back(OpIdx, -1);
    }
    ArgIdx++;
  }

  if (ArgIdx != Result->getNumArgs())
    PrintFatalError(R->getLoc(), "too many operands for instruction!");
}
