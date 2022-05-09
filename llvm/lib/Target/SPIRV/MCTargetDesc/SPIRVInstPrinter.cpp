//===-- SPIRVInstPrinter.cpp - Output SPIR-V MCInsts as ASM -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This class prints a SPIR-V MCInst to a .s file.
//
//===----------------------------------------------------------------------===//

#include "SPIRVInstPrinter.h"
#include "SPIRV.h"
#include "SPIRVBaseInfo.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"

using namespace llvm;

#define DEBUG_TYPE "asm-printer"

// Include the auto-generated portion of the assembly writer.
#include "SPIRVGenAsmWriter.inc"

void SPIRVInstPrinter::printRemainingVariableOps(const MCInst *MI,
                                                 unsigned StartIndex,
                                                 raw_ostream &O,
                                                 bool SkipFirstSpace,
                                                 bool SkipImmediates) {
  const unsigned NumOps = MI->getNumOperands();
  for (unsigned i = StartIndex; i < NumOps; ++i) {
    if (!SkipImmediates || !MI->getOperand(i).isImm()) {
      if (!SkipFirstSpace || i != StartIndex)
        O << ' ';
      printOperand(MI, i, O);
    }
  }
}

void SPIRVInstPrinter::printOpConstantVarOps(const MCInst *MI,
                                             unsigned StartIndex,
                                             raw_ostream &O) {
  O << ' ';
  if (MI->getNumOperands() - StartIndex == 2) { // Handle 64 bit literals.
    uint64_t Imm = MI->getOperand(StartIndex).getImm();
    Imm |= (MI->getOperand(StartIndex + 1).getImm() << 32);
    O << Imm;
  } else {
    printRemainingVariableOps(MI, StartIndex, O, true, false);
  }
}

void SPIRVInstPrinter::recordOpExtInstImport(const MCInst *MI) {
  llvm_unreachable("Unimplemented recordOpExtInstImport");
}

void SPIRVInstPrinter::printInst(const MCInst *MI, uint64_t Address,
                                 StringRef Annot, const MCSubtargetInfo &STI,
                                 raw_ostream &OS) {
  const unsigned OpCode = MI->getOpcode();
  printInstruction(MI, Address, OS);

  if (OpCode == SPIRV::OpDecorate) {
    printOpDecorate(MI, OS);
  } else if (OpCode == SPIRV::OpExtInstImport) {
    recordOpExtInstImport(MI);
  } else if (OpCode == SPIRV::OpExtInst) {
    printOpExtInst(MI, OS);
  } else {
    // Print any extra operands for variadic instructions.
    MCInstrDesc MCDesc = MII.get(OpCode);
    if (MCDesc.isVariadic()) {
      const unsigned NumFixedOps = MCDesc.getNumOperands();
      const unsigned LastFixedIndex = NumFixedOps - 1;
      const int FirstVariableIndex = NumFixedOps;
      if (NumFixedOps > 0 &&
          MCDesc.OpInfo[LastFixedIndex].OperandType == MCOI::OPERAND_UNKNOWN) {
        // For instructions where a custom type (not reg or immediate) comes as
        // the last operand before the variable_ops. This is usually a StringImm
        // operand, but there are a few other cases.
        switch (OpCode) {
        case SPIRV::OpTypeImage:
          OS << ' ';
          printAccessQualifier(MI, FirstVariableIndex, OS);
          break;
        case SPIRV::OpVariable:
          OS << ' ';
          printOperand(MI, FirstVariableIndex, OS);
          break;
        case SPIRV::OpEntryPoint: {
          // Print the interface ID operands, skipping the name's string
          // literal.
          printRemainingVariableOps(MI, NumFixedOps, OS, false, true);
          break;
        }
        case SPIRV::OpExecutionMode:
        case SPIRV::OpExecutionModeId:
        case SPIRV::OpLoopMerge: {
          // Print any literals after the OPERAND_UNKNOWN argument normally.
          printRemainingVariableOps(MI, NumFixedOps, OS);
          break;
        }
        default:
          break; // printStringImm has already been handled
        }
      } else {
        // For instructions with no fixed ops or a reg/immediate as the final
        // fixed operand, we can usually print the rest with "printOperand", but
        // check for a few cases with custom types first.
        switch (OpCode) {
        case SPIRV::OpLoad:
        case SPIRV::OpStore:
          OS << ' ';
          printMemoryOperand(MI, FirstVariableIndex, OS);
          printRemainingVariableOps(MI, FirstVariableIndex + 1, OS);
          break;
        case SPIRV::OpImageSampleImplicitLod:
        case SPIRV::OpImageSampleDrefImplicitLod:
        case SPIRV::OpImageSampleProjImplicitLod:
        case SPIRV::OpImageSampleProjDrefImplicitLod:
        case SPIRV::OpImageFetch:
        case SPIRV::OpImageGather:
        case SPIRV::OpImageDrefGather:
        case SPIRV::OpImageRead:
        case SPIRV::OpImageWrite:
        case SPIRV::OpImageSparseSampleImplicitLod:
        case SPIRV::OpImageSparseSampleDrefImplicitLod:
        case SPIRV::OpImageSparseSampleProjImplicitLod:
        case SPIRV::OpImageSparseSampleProjDrefImplicitLod:
        case SPIRV::OpImageSparseFetch:
        case SPIRV::OpImageSparseGather:
        case SPIRV::OpImageSparseDrefGather:
        case SPIRV::OpImageSparseRead:
        case SPIRV::OpImageSampleFootprintNV:
          OS << ' ';
          printImageOperand(MI, FirstVariableIndex, OS);
          printRemainingVariableOps(MI, NumFixedOps + 1, OS);
          break;
        case SPIRV::OpCopyMemory:
        case SPIRV::OpCopyMemorySized: {
          const unsigned NumOps = MI->getNumOperands();
          for (unsigned i = NumFixedOps; i < NumOps; ++i) {
            OS << ' ';
            printMemoryOperand(MI, i, OS);
            if (MI->getOperand(i).getImm() &
                static_cast<unsigned>(SPIRV::MemoryOperand::Aligned)) {
              assert(i + 1 < NumOps && "Missing alignment operand");
              OS << ' ';
              printOperand(MI, i + 1, OS);
              i += 1;
            }
          }
          break;
        }
        case SPIRV::OpConstantI:
        case SPIRV::OpConstantF:
          printOpConstantVarOps(MI, NumFixedOps, OS);
          break;
        default:
          printRemainingVariableOps(MI, NumFixedOps, OS);
          break;
        }
      }
    }
  }

  printAnnotation(OS, Annot);
}

void SPIRVInstPrinter::printOpExtInst(const MCInst *MI, raw_ostream &O) {
  llvm_unreachable("Unimplemented printOpExtInst");
}

void SPIRVInstPrinter::printOpDecorate(const MCInst *MI, raw_ostream &O) {
  // The fixed operands have already been printed, so just need to decide what
  // type of decoration operands to print based on the Decoration type.
  MCInstrDesc MCDesc = MII.get(MI->getOpcode());
  unsigned NumFixedOps = MCDesc.getNumOperands();

  if (NumFixedOps != MI->getNumOperands()) {
    auto DecOp = MI->getOperand(NumFixedOps - 1);
    auto Dec = static_cast<SPIRV::Decoration>(DecOp.getImm());

    O << ' ';

    switch (Dec) {
    case SPIRV::Decoration::BuiltIn:
      printBuiltIn(MI, NumFixedOps, O);
      break;
    case SPIRV::Decoration::UniformId:
      printScope(MI, NumFixedOps, O);
      break;
    case SPIRV::Decoration::FuncParamAttr:
      printFunctionParameterAttribute(MI, NumFixedOps, O);
      break;
    case SPIRV::Decoration::FPRoundingMode:
      printFPRoundingMode(MI, NumFixedOps, O);
      break;
    case SPIRV::Decoration::FPFastMathMode:
      printFPFastMathMode(MI, NumFixedOps, O);
      break;
    case SPIRV::Decoration::LinkageAttributes:
    case SPIRV::Decoration::UserSemantic:
      printStringImm(MI, NumFixedOps, O);
      break;
    default:
      printRemainingVariableOps(MI, NumFixedOps, O, true);
      break;
    }
  }
}

static void printExpr(const MCExpr *Expr, raw_ostream &O) {
#ifndef NDEBUG
  const MCSymbolRefExpr *SRE;

  if (const MCBinaryExpr *BE = dyn_cast<MCBinaryExpr>(Expr))
    SRE = cast<MCSymbolRefExpr>(BE->getLHS());
  else
    SRE = cast<MCSymbolRefExpr>(Expr);

  MCSymbolRefExpr::VariantKind Kind = SRE->getKind();

  assert(Kind == MCSymbolRefExpr::VK_None);
#endif
  O << *Expr;
}

void SPIRVInstPrinter::printOperand(const MCInst *MI, unsigned OpNo,
                                    raw_ostream &O, const char *Modifier) {
  assert((Modifier == 0 || Modifier[0] == 0) && "No modifiers supported");
  if (OpNo < MI->getNumOperands()) {
    const MCOperand &Op = MI->getOperand(OpNo);
    if (Op.isReg())
      O << '%' << (Register::virtReg2Index(Op.getReg()) + 1);
    else if (Op.isImm())
      O << formatImm((int64_t)Op.getImm());
    else if (Op.isDFPImm())
      O << formatImm((double)Op.getDFPImm());
    else if (Op.isExpr())
      printExpr(Op.getExpr(), O);
    else
      llvm_unreachable("Unexpected operand type");
  }
}

void SPIRVInstPrinter::printStringImm(const MCInst *MI, unsigned OpNo,
                                      raw_ostream &O) {
  const unsigned NumOps = MI->getNumOperands();
  unsigned StrStartIndex = OpNo;
  while (StrStartIndex < NumOps) {
    if (MI->getOperand(StrStartIndex).isReg())
      break;

    std::string Str = getSPIRVStringOperand(*MI, OpNo);
    if (StrStartIndex != OpNo)
      O << ' '; // Add a space if we're starting a new string/argument.
    O << '"';
    for (char c : Str) {
      if (c == '"')
        O.write('\\'); // Escape " characters (might break for complex UTF-8).
      O.write(c);
    }
    O << '"';

    unsigned numOpsInString = (Str.size() / 4) + 1;
    StrStartIndex += numOpsInString;

    // Check for final Op of "OpDecorate %x %stringImm %linkageAttribute".
    if (MI->getOpcode() == SPIRV::OpDecorate &&
        MI->getOperand(1).getImm() ==
            static_cast<unsigned>(SPIRV::Decoration::LinkageAttributes)) {
      O << ' ';
      printLinkageType(MI, StrStartIndex, O);
      break;
    }
  }
}

void SPIRVInstPrinter::printExtInst(const MCInst *MI, unsigned OpNo,
                                    raw_ostream &O) {
  llvm_unreachable("Unimplemented printExtInst");
}

void SPIRVInstPrinter::printCapability(const MCInst *MI, unsigned OpNo,
                                       raw_ostream &O) {
  if (OpNo < MI->getNumOperands()) {
    SPIRV::Capability e =
        static_cast<SPIRV::Capability>(MI->getOperand(OpNo).getImm());
    O << SPIRV::getCapabilityName(e);
  }
}

void SPIRVInstPrinter::printSourceLanguage(const MCInst *MI, unsigned OpNo,
                                           raw_ostream &O) {
  if (OpNo < MI->getNumOperands()) {
    SPIRV::SourceLanguage e =
        static_cast<SPIRV::SourceLanguage>(MI->getOperand(OpNo).getImm());
    O << SPIRV::getSourceLanguageName(e);
  }
}

void SPIRVInstPrinter::printExecutionModel(const MCInst *MI, unsigned OpNo,
                                           raw_ostream &O) {
  if (OpNo < MI->getNumOperands()) {
    SPIRV::ExecutionModel e =
        static_cast<SPIRV::ExecutionModel>(MI->getOperand(OpNo).getImm());
    O << SPIRV::getExecutionModelName(e);
  }
}

void SPIRVInstPrinter::printAddressingModel(const MCInst *MI, unsigned OpNo,
                                            raw_ostream &O) {
  if (OpNo < MI->getNumOperands()) {
    SPIRV::AddressingModel e =
        static_cast<SPIRV::AddressingModel>(MI->getOperand(OpNo).getImm());
    O << SPIRV::getAddressingModelName(e);
  }
}

void SPIRVInstPrinter::printMemoryModel(const MCInst *MI, unsigned OpNo,
                                        raw_ostream &O) {
  if (OpNo < MI->getNumOperands()) {
    SPIRV::MemoryModel e =
        static_cast<SPIRV::MemoryModel>(MI->getOperand(OpNo).getImm());
    O << SPIRV::getMemoryModelName(e);
  }
}

void SPIRVInstPrinter::printExecutionMode(const MCInst *MI, unsigned OpNo,
                                          raw_ostream &O) {
  if (OpNo < MI->getNumOperands()) {
    SPIRV::ExecutionMode e =
        static_cast<SPIRV::ExecutionMode>(MI->getOperand(OpNo).getImm());
    O << SPIRV::getExecutionModeName(e);
  }
}

void SPIRVInstPrinter::printStorageClass(const MCInst *MI, unsigned OpNo,
                                         raw_ostream &O) {
  if (OpNo < MI->getNumOperands()) {
    SPIRV::StorageClass e =
        static_cast<SPIRV::StorageClass>(MI->getOperand(OpNo).getImm());
    O << SPIRV::getStorageClassName(e);
  }
}

void SPIRVInstPrinter::printDim(const MCInst *MI, unsigned OpNo,
                                raw_ostream &O) {
  if (OpNo < MI->getNumOperands()) {
    SPIRV::Dim e = static_cast<SPIRV::Dim>(MI->getOperand(OpNo).getImm());
    O << SPIRV::getDimName(e);
  }
}

void SPIRVInstPrinter::printSamplerAddressingMode(const MCInst *MI,
                                                  unsigned OpNo,
                                                  raw_ostream &O) {
  if (OpNo < MI->getNumOperands()) {
    SPIRV::SamplerAddressingMode e = static_cast<SPIRV::SamplerAddressingMode>(
        MI->getOperand(OpNo).getImm());
    O << SPIRV::getSamplerAddressingModeName(e);
  }
}

void SPIRVInstPrinter::printSamplerFilterMode(const MCInst *MI, unsigned OpNo,
                                              raw_ostream &O) {
  if (OpNo < MI->getNumOperands()) {
    SPIRV::SamplerFilterMode e =
        static_cast<SPIRV::SamplerFilterMode>(MI->getOperand(OpNo).getImm());
    O << SPIRV::getSamplerFilterModeName(e);
  }
}

void SPIRVInstPrinter::printImageFormat(const MCInst *MI, unsigned OpNo,
                                        raw_ostream &O) {
  if (OpNo < MI->getNumOperands()) {
    SPIRV::ImageFormat e =
        static_cast<SPIRV::ImageFormat>(MI->getOperand(OpNo).getImm());
    O << SPIRV::getImageFormatName(e);
  }
}

void SPIRVInstPrinter::printImageChannelOrder(const MCInst *MI, unsigned OpNo,
                                              raw_ostream &O) {
  if (OpNo < MI->getNumOperands()) {
    SPIRV::ImageChannelOrder e =
        static_cast<SPIRV::ImageChannelOrder>(MI->getOperand(OpNo).getImm());
    O << SPIRV::getImageChannelOrderName(e);
  }
}

void SPIRVInstPrinter::printImageChannelDataType(const MCInst *MI,
                                                 unsigned OpNo,
                                                 raw_ostream &O) {
  if (OpNo < MI->getNumOperands()) {
    SPIRV::ImageChannelDataType e =
        static_cast<SPIRV::ImageChannelDataType>(MI->getOperand(OpNo).getImm());
    O << SPIRV::getImageChannelDataTypeName(e);
  }
}

void SPIRVInstPrinter::printImageOperand(const MCInst *MI, unsigned OpNo,
                                         raw_ostream &O) {
  if (OpNo < MI->getNumOperands()) {
    unsigned e = static_cast<unsigned>(MI->getOperand(OpNo).getImm());
    O << SPIRV::getImageOperandName(e);
  }
}

void SPIRVInstPrinter::printFPFastMathMode(const MCInst *MI, unsigned OpNo,
                                           raw_ostream &O) {
  if (OpNo < MI->getNumOperands()) {
    unsigned e = static_cast<unsigned>(MI->getOperand(OpNo).getImm());
    O << SPIRV::getFPFastMathModeName(e);
  }
}

void SPIRVInstPrinter::printFPRoundingMode(const MCInst *MI, unsigned OpNo,
                                           raw_ostream &O) {
  if (OpNo < MI->getNumOperands()) {
    SPIRV::FPRoundingMode e =
        static_cast<SPIRV::FPRoundingMode>(MI->getOperand(OpNo).getImm());
    O << SPIRV::getFPRoundingModeName(e);
  }
}

void SPIRVInstPrinter::printLinkageType(const MCInst *MI, unsigned OpNo,
                                        raw_ostream &O) {
  if (OpNo < MI->getNumOperands()) {
    SPIRV::LinkageType e =
        static_cast<SPIRV::LinkageType>(MI->getOperand(OpNo).getImm());
    O << SPIRV::getLinkageTypeName(e);
  }
}

void SPIRVInstPrinter::printAccessQualifier(const MCInst *MI, unsigned OpNo,
                                            raw_ostream &O) {
  if (OpNo < MI->getNumOperands()) {
    SPIRV::AccessQualifier e =
        static_cast<SPIRV::AccessQualifier>(MI->getOperand(OpNo).getImm());
    O << SPIRV::getAccessQualifierName(e);
  }
}

void SPIRVInstPrinter::printFunctionParameterAttribute(const MCInst *MI,
                                                       unsigned OpNo,
                                                       raw_ostream &O) {
  if (OpNo < MI->getNumOperands()) {
    SPIRV::FunctionParameterAttribute e =
        static_cast<SPIRV::FunctionParameterAttribute>(
            MI->getOperand(OpNo).getImm());
    O << SPIRV::getFunctionParameterAttributeName(e);
  }
}

void SPIRVInstPrinter::printDecoration(const MCInst *MI, unsigned OpNo,
                                       raw_ostream &O) {
  if (OpNo < MI->getNumOperands()) {
    SPIRV::Decoration e =
        static_cast<SPIRV::Decoration>(MI->getOperand(OpNo).getImm());
    O << SPIRV::getDecorationName(e);
  }
}

void SPIRVInstPrinter::printBuiltIn(const MCInst *MI, unsigned OpNo,
                                    raw_ostream &O) {
  if (OpNo < MI->getNumOperands()) {
    SPIRV::BuiltIn e =
        static_cast<SPIRV::BuiltIn>(MI->getOperand(OpNo).getImm());
    O << SPIRV::getBuiltInName(e);
  }
}

void SPIRVInstPrinter::printSelectionControl(const MCInst *MI, unsigned OpNo,
                                             raw_ostream &O) {
  if (OpNo < MI->getNumOperands()) {
    unsigned e = static_cast<unsigned>(MI->getOperand(OpNo).getImm());
    O << SPIRV::getSelectionControlName(e);
  }
}

void SPIRVInstPrinter::printLoopControl(const MCInst *MI, unsigned OpNo,
                                        raw_ostream &O) {
  if (OpNo < MI->getNumOperands()) {
    unsigned e = static_cast<unsigned>(MI->getOperand(OpNo).getImm());
    O << SPIRV::getLoopControlName(e);
  }
}

void SPIRVInstPrinter::printFunctionControl(const MCInst *MI, unsigned OpNo,
                                            raw_ostream &O) {
  if (OpNo < MI->getNumOperands()) {
    unsigned e = static_cast<unsigned>(MI->getOperand(OpNo).getImm());
    O << SPIRV::getFunctionControlName(e);
  }
}

void SPIRVInstPrinter::printMemorySemantics(const MCInst *MI, unsigned OpNo,
                                            raw_ostream &O) {
  if (OpNo < MI->getNumOperands()) {
    unsigned e = static_cast<unsigned>(MI->getOperand(OpNo).getImm());
    O << SPIRV::getMemorySemanticsName(e);
  }
}

void SPIRVInstPrinter::printMemoryOperand(const MCInst *MI, unsigned OpNo,
                                          raw_ostream &O) {
  if (OpNo < MI->getNumOperands()) {
    unsigned e = static_cast<unsigned>(MI->getOperand(OpNo).getImm());
    O << SPIRV::getMemoryOperandName(e);
  }
}

void SPIRVInstPrinter::printScope(const MCInst *MI, unsigned OpNo,
                                  raw_ostream &O) {
  if (OpNo < MI->getNumOperands()) {
    SPIRV::Scope e = static_cast<SPIRV::Scope>(MI->getOperand(OpNo).getImm());
    O << SPIRV::getScopeName(e);
  }
}

void SPIRVInstPrinter::printGroupOperation(const MCInst *MI, unsigned OpNo,
                                           raw_ostream &O) {
  if (OpNo < MI->getNumOperands()) {
    SPIRV::GroupOperation e =
        static_cast<SPIRV::GroupOperation>(MI->getOperand(OpNo).getImm());
    O << SPIRV::getGroupOperationName(e);
  }
}

void SPIRVInstPrinter::printKernelEnqueueFlags(const MCInst *MI, unsigned OpNo,
                                               raw_ostream &O) {
  if (OpNo < MI->getNumOperands()) {
    SPIRV::KernelEnqueueFlags e =
        static_cast<SPIRV::KernelEnqueueFlags>(MI->getOperand(OpNo).getImm());
    O << SPIRV::getKernelEnqueueFlagsName(e);
  }
}

void SPIRVInstPrinter::printKernelProfilingInfo(const MCInst *MI, unsigned OpNo,
                                                raw_ostream &O) {
  if (OpNo < MI->getNumOperands()) {
    SPIRV::KernelProfilingInfo e =
        static_cast<SPIRV::KernelProfilingInfo>(MI->getOperand(OpNo).getImm());
    O << SPIRV::getKernelProfilingInfoName(e);
  }
}
