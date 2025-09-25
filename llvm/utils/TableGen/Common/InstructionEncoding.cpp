//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InstructionEncoding.h"
#include "CodeGenInstruction.h"
#include "VarLenCodeEmitterGen.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Error.h"

using namespace llvm;

std::pair<std::string, bool>
InstructionEncoding::findOperandDecoderMethod(const Record *Record) {
  std::string Decoder;

  const RecordVal *DecoderString = Record->getValue("DecoderMethod");
  const StringInit *String =
      DecoderString ? dyn_cast<StringInit>(DecoderString->getValue()) : nullptr;
  if (String) {
    Decoder = String->getValue().str();
    if (!Decoder.empty())
      return {Decoder, false};
  }

  if (Record->isSubClassOf("RegisterOperand"))
    // Allows use of a DecoderMethod in referenced RegisterClass if set.
    return findOperandDecoderMethod(Record->getValueAsDef("RegClass"));

  if (Record->isSubClassOf("RegisterClass")) {
    Decoder = "Decode" + Record->getName().str() + "RegisterClass";
  } else if (Record->isSubClassOf("RegClassByHwMode")) {
    Decoder = "Decode" + Record->getName().str() + "RegClassByHwMode";
  } else if (Record->isSubClassOf("PointerLikeRegClass")) {
    Decoder = "DecodePointerLikeRegClass" +
              utostr(Record->getValueAsInt("RegClassKind"));
  }

  return {Decoder, true};
}

OperandInfo InstructionEncoding::getOpInfo(const Record *TypeRecord) {
  const RecordVal *HasCompleteDecoderVal =
      TypeRecord->getValue("hasCompleteDecoder");
  const BitInit *HasCompleteDecoderBit =
      HasCompleteDecoderVal
          ? dyn_cast<BitInit>(HasCompleteDecoderVal->getValue())
          : nullptr;
  bool HasCompleteDecoder =
      HasCompleteDecoderBit ? HasCompleteDecoderBit->getValue() : true;

  return OperandInfo(findOperandDecoderMethod(TypeRecord).first,
                     HasCompleteDecoder);
}

void InstructionEncoding::parseVarLenEncoding(const VarLenInst &VLI) {
  InstBits = KnownBits(VLI.size());
  SoftFailMask = APInt(VLI.size(), 0);

  // Parse Inst field.
  unsigned I = 0;
  for (const EncodingSegment &S : VLI) {
    if (const auto *SegmentBits = dyn_cast<BitsInit>(S.Value)) {
      for (const Init *V : SegmentBits->getBits()) {
        if (const auto *B = dyn_cast<BitInit>(V)) {
          if (B->getValue())
            InstBits.One.setBit(I);
          else
            InstBits.Zero.setBit(I);
        }
        ++I;
      }
    } else if (const auto *B = dyn_cast<BitInit>(S.Value)) {
      if (B->getValue())
        InstBits.One.setBit(I);
      else
        InstBits.Zero.setBit(I);
      ++I;
    } else {
      I += S.BitWidth;
    }
  }
  assert(I == VLI.size());
}

void InstructionEncoding::parseFixedLenEncoding(
    const BitsInit &RecordInstBits) {
  // For fixed length instructions, sometimes the `Inst` field specifies more
  // bits than the actual size of the instruction, which is specified in `Size`.
  // In such cases, we do some basic validation and drop the upper bits.
  unsigned BitWidth = EncodingDef->getValueAsInt("Size") * 8;
  unsigned InstNumBits = RecordInstBits.getNumBits();

  // Returns true if all bits in `Bits` are zero or unset.
  auto CheckAllZeroOrUnset = [&](ArrayRef<const Init *> Bits,
                                 const RecordVal *Field) {
    bool AllZeroOrUnset = llvm::all_of(Bits, [](const Init *Bit) {
      if (const auto *BI = dyn_cast<BitInit>(Bit))
        return !BI->getValue();
      return isa<UnsetInit>(Bit);
    });
    if (AllZeroOrUnset)
      return;
    PrintNote([Field](raw_ostream &OS) { Field->print(OS); });
    PrintFatalError(EncodingDef, Twine(Name) + ": Size is " + Twine(BitWidth) +
                                     " bits, but " + Field->getName() +
                                     " bits beyond that are    not zero/unset");
  };

  if (InstNumBits < BitWidth)
    PrintFatalError(EncodingDef, Twine(Name) + ": Size is " + Twine(BitWidth) +
                                     " bits, but Inst specifies only " +
                                     Twine(InstNumBits) + " bits");

  if (InstNumBits > BitWidth) {
    // Ensure that all the bits beyond 'Size' are 0 or unset (i.e., carry no
    // actual encoding).
    ArrayRef<const Init *> UpperBits =
        RecordInstBits.getBits().drop_front(BitWidth);
    const RecordVal *InstField = EncodingDef->getValue("Inst");
    CheckAllZeroOrUnset(UpperBits, InstField);
  }

  ArrayRef<const Init *> ActiveInstBits =
      RecordInstBits.getBits().take_front(BitWidth);
  InstBits = KnownBits(BitWidth);
  SoftFailMask = APInt(BitWidth, 0);

  // Parse Inst field.
  for (auto [I, V] : enumerate(ActiveInstBits)) {
    if (const auto *B = dyn_cast<BitInit>(V)) {
      if (B->getValue())
        InstBits.One.setBit(I);
      else
        InstBits.Zero.setBit(I);
    }
  }

  // Parse SoftFail field.
  const RecordVal *SoftFailField = EncodingDef->getValue("SoftFail");
  if (!SoftFailField)
    return;

  const auto *SFBits = dyn_cast<BitsInit>(SoftFailField->getValue());
  if (!SFBits || SFBits->getNumBits() != InstNumBits) {
    PrintNote(EncodingDef->getLoc(), "in record");
    PrintFatalError(SoftFailField,
                    formatv("SoftFail field, if defined, must be "
                            "of the same type as Inst, which is bits<{}>",
                            InstNumBits));
  }

  if (InstNumBits > BitWidth) {
    // Ensure that all upper bits of `SoftFail` are 0 or unset.
    ArrayRef<const Init *> UpperBits = SFBits->getBits().drop_front(BitWidth);
    CheckAllZeroOrUnset(UpperBits, SoftFailField);
  }

  ArrayRef<const Init *> ActiveSFBits = SFBits->getBits().take_front(BitWidth);
  for (auto [I, V] : enumerate(ActiveSFBits)) {
    if (const auto *B = dyn_cast<BitInit>(V); B && B->getValue()) {
      if (!InstBits.Zero[I] && !InstBits.One[I]) {
        PrintNote(EncodingDef->getLoc(), "in record");
        PrintError(SoftFailField,
                   formatv("SoftFail{{{0}} = 1 requires Inst{{{0}} "
                           "to be fully defined (0 or 1, not '?')",
                           I));
      }
      SoftFailMask.setBit(I);
    }
  }
}

void InstructionEncoding::parseVarLenOperands(const VarLenInst &VLI) {
  SmallVector<int> TiedTo;

  for (const auto &[Idx, Op] : enumerate(Inst->Operands)) {
    if (Op.MIOperandInfo && Op.MIOperandInfo->getNumArgs() > 0)
      for (auto *Arg : Op.MIOperandInfo->getArgs())
        Operands.push_back(getOpInfo(cast<DefInit>(Arg)->getDef()));
    else
      Operands.push_back(getOpInfo(Op.Rec));

    int TiedReg = Op.getTiedRegister();
    TiedTo.push_back(-1);
    if (TiedReg != -1) {
      TiedTo[Idx] = TiedReg;
      TiedTo[TiedReg] = Idx;
    }
  }

  unsigned CurrBitPos = 0;
  for (const auto &EncodingSegment : VLI) {
    unsigned Offset = 0;
    StringRef OpName;

    if (const StringInit *SI = dyn_cast<StringInit>(EncodingSegment.Value)) {
      OpName = SI->getValue();
    } else if (const DagInit *DI = dyn_cast<DagInit>(EncodingSegment.Value)) {
      OpName = cast<StringInit>(DI->getArg(0))->getValue();
      Offset = cast<IntInit>(DI->getArg(2))->getValue();
    }

    if (!OpName.empty()) {
      auto OpSubOpPair = Inst->Operands.parseOperandName(OpName);
      unsigned OpIdx = Inst->Operands.getFlattenedOperandNumber(OpSubOpPair);
      Operands[OpIdx].addField(CurrBitPos, EncodingSegment.BitWidth, Offset);
      if (!EncodingSegment.CustomDecoder.empty())
        Operands[OpIdx].Decoder = EncodingSegment.CustomDecoder.str();

      int TiedReg = TiedTo[OpSubOpPair.first];
      if (TiedReg != -1) {
        unsigned OpIdx = Inst->Operands.getFlattenedOperandNumber(
            {TiedReg, OpSubOpPair.second});
        Operands[OpIdx].addField(CurrBitPos, EncodingSegment.BitWidth, Offset);
      }
    }

    CurrBitPos += EncodingSegment.BitWidth;
  }
}

static void debugDumpRecord(const Record &Rec) {
  // Dump the record, so we can see what's going on.
  PrintNote([&Rec](raw_ostream &OS) {
    OS << "Dumping record for previous error:\n";
    OS << Rec;
  });
}

/// For an operand field named OpName: populate OpInfo.InitValue with the
/// constant-valued bit values, and OpInfo.Fields with the ranges of bits to
/// insert from the decoded instruction.
static void addOneOperandFields(const Record *EncodingDef,
                                const BitsInit &InstBits,
                                std::map<StringRef, StringRef> &TiedNames,
                                const Record *OpRec, StringRef OpName,
                                OperandInfo &OpInfo) {
  OpInfo.Name = OpName;

  // Find a field with the operand's name.
  const RecordVal *OpEncodingField = EncodingDef->getValue(OpName);

  // If there is no such field, try tied operand's name.
  if (!OpEncodingField) {
    if (auto I = TiedNames.find(OpName); I != TiedNames.end())
      OpEncodingField = EncodingDef->getValue(I->second);

    // If still no luck, we're done with this operand.
    if (!OpEncodingField) {
      OpInfo.HasNoEncoding = true;
      return;
    }
  }

  // Some or all bits of the operand may be required to be 0 or 1 depending
  // on the instruction's encoding. Collect those bits.
  if (const auto *OpBit = dyn_cast<BitInit>(OpEncodingField->getValue())) {
    OpInfo.InitValue = OpBit->getValue();
    return;
  }
  if (const auto *OpBits = dyn_cast<BitsInit>(OpEncodingField->getValue())) {
    if (OpBits->getNumBits() == 0) {
      if (OpInfo.Decoder.empty()) {
        PrintError(EncodingDef->getLoc(), "operand '" + OpName + "' of type '" +
                                              OpRec->getName() +
                                              "' must have a decoder method");
      }
      return;
    }
    for (unsigned I = 0; I < OpBits->getNumBits(); ++I) {
      if (const auto *OpBit = dyn_cast<BitInit>(OpBits->getBit(I)))
        OpInfo.InitValue = OpInfo.InitValue.value_or(0) |
                           static_cast<uint64_t>(OpBit->getValue()) << I;
    }
  }

  // Find out where the variable bits of the operand are encoded. The bits don't
  // have to be consecutive or in ascending order. For example, an operand could
  // be encoded as follows:
  //
  //  7    6      5      4    3    2      1    0
  // {1, op{5}, op{2}, op{1}, 0, op{4}, op{3}, ?}
  //
  // In this example the operand is encoded in three segments:
  //
  //           Base Width Offset
  // op{2...1}   4    2     1
  // op{4...3}   1    2     3
  // op{5}       6    1     5
  //
  for (unsigned I = 0, J = 0; I != InstBits.getNumBits(); I = J) {
    const VarInit *Var;
    unsigned Offset = 0;
    for (; J != InstBits.getNumBits(); ++J) {
      const Init *BitJ = InstBits.getBit(J);
      if (const auto *VBI = dyn_cast<VarBitInit>(BitJ)) {
        Var = dyn_cast<VarInit>(VBI->getBitVar());
        if (I == J)
          Offset = VBI->getBitNum();
        else if (VBI->getBitNum() != Offset + J - I)
          break;
      } else {
        Var = dyn_cast<VarInit>(BitJ);
      }
      if (!Var ||
          (Var->getName() != OpName && Var->getName() != TiedNames[OpName]))
        break;
    }
    if (I == J)
      ++J;
    else
      OpInfo.addField(I, J - I, Offset);
  }

  if (!OpInfo.InitValue && OpInfo.fields().empty()) {
    // We found a field in InstructionEncoding record that corresponds to the
    // named operand, but that field has no constant bits and doesn't contribute
    // to the Inst field. For now, treat that field as if it didn't exist.
    // TODO: Remove along with IgnoreNonDecodableOperands.
    OpInfo.HasNoEncoding = true;
  }
}

void InstructionEncoding::parseFixedLenOperands(const BitsInit &Bits) {
  // Search for tied operands, so that we can correctly instantiate
  // operands that are not explicitly represented in the encoding.
  std::map<StringRef, StringRef> TiedNames;
  for (const auto &Op : Inst->Operands) {
    for (const auto &[J, CI] : enumerate(Op.Constraints)) {
      if (!CI.isTied())
        continue;
      std::pair<unsigned, unsigned> SO =
          Inst->Operands.getSubOperandNumber(CI.getTiedOperand());
      StringRef TiedName = Inst->Operands[SO.first].SubOpNames[SO.second];
      if (TiedName.empty())
        TiedName = Inst->Operands[SO.first].Name;
      StringRef MyName = Op.SubOpNames[J];
      if (MyName.empty())
        MyName = Op.Name;

      TiedNames[MyName] = TiedName;
      TiedNames[TiedName] = MyName;
    }
  }

  // For each operand, see if we can figure out where it is encoded.
  for (const CGIOperandList::OperandInfo &Op : Inst->Operands) {
    // Lookup the decoder method and construct a new OperandInfo to hold our
    // result.
    OperandInfo OpInfo = getOpInfo(Op.Rec);

    // If we have named sub-operands...
    if (Op.MIOperandInfo && !Op.SubOpNames[0].empty()) {
      // Then there should not be a custom decoder specified on the top-level
      // type.
      if (!OpInfo.Decoder.empty()) {
        PrintError(EncodingDef,
                   "DecoderEmitter: operand \"" + Op.Name + "\" has type \"" +
                       Op.Rec->getName() +
                       "\" with a custom DecoderMethod, but also named "
                       "sub-operands.");
        continue;
      }

      // Decode each of the sub-ops separately.
      for (auto [SubOpName, SubOp] :
           zip_equal(Op.SubOpNames, Op.MIOperandInfo->getArgs())) {
        const Record *SubOpRec = cast<DefInit>(SubOp)->getDef();
        OperandInfo SubOpInfo = getOpInfo(SubOpRec);
        addOneOperandFields(EncodingDef, Bits, TiedNames, SubOpRec, SubOpName,
                            SubOpInfo);
        Operands.push_back(std::move(SubOpInfo));
      }
      continue;
    }

    // Otherwise, if we have an operand with sub-operands, but they aren't
    // named...
    if (Op.MIOperandInfo && OpInfo.Decoder.empty()) {
      // If we have sub-ops, we'd better have a custom decoder.
      // (Otherwise we don't know how to populate them properly...)
      if (Op.MIOperandInfo->getNumArgs()) {
        PrintError(EncodingDef,
                   "DecoderEmitter: operand \"" + Op.Name +
                       "\" has non-empty MIOperandInfo, but doesn't "
                       "have a custom decoder!");
        debugDumpRecord(*EncodingDef);
        continue;
      }
    }

    addOneOperandFields(EncodingDef, Bits, TiedNames, Op.Rec, Op.Name, OpInfo);
    Operands.push_back(std::move(OpInfo));
  }
}

InstructionEncoding::InstructionEncoding(const Record *EncodingDef,
                                         const CodeGenInstruction *Inst)
    : EncodingDef(EncodingDef), Inst(Inst) {
  const Record *InstDef = Inst->TheDef;

  // Give this encoding a name.
  if (EncodingDef != InstDef)
    Name = (EncodingDef->getName() + Twine(':')).str();
  Name.append(InstDef->getName());

  DecoderNamespace = EncodingDef->getValueAsString("DecoderNamespace");
  DecoderMethod = EncodingDef->getValueAsString("DecoderMethod");
  if (!DecoderMethod.empty())
    HasCompleteDecoder = EncodingDef->getValueAsBit("hasCompleteDecoder");

  const RecordVal *InstField = EncodingDef->getValue("Inst");
  if (const auto *DI = dyn_cast<DagInit>(InstField->getValue())) {
    VarLenInst VLI(DI, InstField);
    parseVarLenEncoding(VLI);
    // If the encoding has a custom decoder, don't bother parsing the operands.
    if (DecoderMethod.empty())
      parseVarLenOperands(VLI);
  } else {
    const auto *BI = cast<BitsInit>(InstField->getValue());
    parseFixedLenEncoding(*BI);
    // If the encoding has a custom decoder, don't bother parsing the operands.
    if (DecoderMethod.empty())
      parseFixedLenOperands(*BI);
  }

  if (DecoderMethod.empty()) {
    // A generated decoder is always successful if none of the operand
    // decoders can fail (all are always successful).
    HasCompleteDecoder = all_of(Operands, [](const OperandInfo &Op) {
      // By default, a generated operand decoder is assumed to always succeed.
      // This can be overridden by the user.
      return Op.Decoder.empty() || Op.HasCompleteDecoder;
    });
  }
}
