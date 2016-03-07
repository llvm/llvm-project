//===--------------------AMDKernelCodeTUtils.cpp --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//
//
/// \file - utility functions to parse/print amd_kernel_code_t structure
//
//===----------------------------------------------------------------------===//

#include "AMDKernelCodeTUtils.h"
#include "SIDefines.h"
#include <llvm/MC/MCParser/MCAsmLexer.h>
#include <llvm/Support/raw_ostream.h>

using namespace llvm;

static ArrayRef<StringRef> get_amd_kernel_code_t_FldNames() {
  static StringRef const Table[] = {
    "", // not found placeholder
#define RECORD(name, print, parse) #name
#include "AMDKernelCodeTInfo.h"
#undef RECORD
  };
  return makeArrayRef(Table);
}

static StringMap<int> createIndexMap(const ArrayRef<StringRef>& a) {
  StringMap<int> map;
  for (auto Name : a)
    map.insert(std::make_pair(Name, map.size()));
  return map;
}

static int get_amd_kernel_code_t_FieldIndex(StringRef name) {
  static const auto map = createIndexMap(get_amd_kernel_code_t_FldNames());
  return map.lookup(name) - 1; // returns -1 if not found
}

static StringRef get_amd_kernel_code_t_FieldName(int index) {
  return get_amd_kernel_code_t_FldNames()[index + 1];
}


// Field printing

raw_ostream& printName(raw_ostream& OS, StringRef Name) {
  return OS << Name << " = ";
}

template <typename T, T amd_kernel_code_t::*ptr>
void printField(StringRef Name,
                const amd_kernel_code_t& C,
                raw_ostream& OS) {
  printName(OS, Name) << (int)(C.*ptr);
}

template <typename T, T amd_kernel_code_t::*ptr, int shift, int width=1>
void printBitField(StringRef Name,
                   const amd_kernel_code_t& c,
                   raw_ostream& OS) {
  const auto Mask = (static_cast<T>(1) << width) - 1;
  printName(OS, Name) << (int)((c.*ptr >> shift) & Mask);
}

typedef void(*PrintFx)(StringRef,
                       const amd_kernel_code_t&,
                       raw_ostream&);

static ArrayRef<PrintFx> getPrinterTable() {
  static const PrintFx Table[] = {
#define RECORD(name, print, parse) print
#include "AMDKernelCodeTInfo.h"
#undef RECORD
  };
  return makeArrayRef(Table);
}

void llvm::printAmdKernelCodeField(const amd_kernel_code_t& C,
                                   int FldIndex,
                                   raw_ostream& OS) {
  auto Printer = getPrinterTable()[FldIndex];
  if (Printer)
    Printer(get_amd_kernel_code_t_FieldName(FldIndex), C, OS);
}

void llvm::dumpAmdKernelCode(const amd_kernel_code_t* C,
                             raw_ostream& OS,
                             const char* tab) {
  const int Size = getPrinterTable().size();
  for (int i = 0; i < Size; ++i) {
    OS << tab;
    printAmdKernelCodeField(*C, i, OS);
    OS << '\n';
  }
}


// Field parsing

static bool expectEqualInt(MCAsmLexer& Lexer, raw_ostream& Err) {
  if (Lexer.isNot(AsmToken::Equal)) {
    Err << "expected '='";
    return false;
  }
  Lexer.Lex();
  if (Lexer.isNot(AsmToken::Integer)) {
    Err << "integer literal expected";
    return false;
  }
  return true;
}

template <typename T, T amd_kernel_code_t::*ptr>
bool parseField(amd_kernel_code_t& C,
                MCAsmLexer& Lexer,
                raw_ostream& Err) {
  if (!expectEqualInt(Lexer, Err))
    return false;
  C.*ptr = (T)Lexer.getTok().getIntVal();
  return true;
}

template <typename T, T amd_kernel_code_t::*ptr, int shift, int width = 1>
bool parseBitField(amd_kernel_code_t& C,
                   MCAsmLexer& Lexer,
                   raw_ostream& Err) {
  if (!expectEqualInt(Lexer, Err))
    return false;
  const uint64_t Mask = ((UINT64_C(1)  << width) - 1) << shift;
  C.*ptr &= (T)~Mask;
  C.*ptr |= (T)((Lexer.getTok().getIntVal() << shift) & Mask);
  return true;
}

typedef bool(*ParseFx)(amd_kernel_code_t&,
                       MCAsmLexer& Lexer,
                       raw_ostream& Err);

static ArrayRef<ParseFx> getParserTable() {
  static const ParseFx Table[] = {
#define RECORD(name, print, parse) parse
#include "AMDKernelCodeTInfo.h"
#undef RECORD
  };
  return makeArrayRef(Table);
}

bool llvm::parseAmdKernelCodeField(StringRef ID,
                                   MCAsmLexer& Lexer,
                                   amd_kernel_code_t& C,
                                   raw_ostream& Err) {
  const int Idx = get_amd_kernel_code_t_FieldIndex(ID);
  if (Idx < 0) {
    Err << "unexpected amd_kernel_code_t field name " << ID;
    return false;
  }
  auto Parser = getParserTable()[Idx];
  return Parser ? Parser(C, Lexer, Err) : false;
}
