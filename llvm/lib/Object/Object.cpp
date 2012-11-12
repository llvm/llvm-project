//===- Object.cpp - C bindings to the object file library--------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the C bindings to the file-format-independent object
// library.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/ObjectFile.h"
#include "llvm-c/Object.h"

using namespace llvm;
using namespace object;

// ObjectFile creation
LLVMObjectFileRef LLVMCreateObjectFile(LLVMMemoryBufferRef MemBuf) {
  return wrap(ObjectFile::createObjectFile(unwrap(MemBuf)));
}

void LLVMDisposeObjectFile(LLVMObjectFileRef ObjectFile) {
  delete unwrap(ObjectFile);
}

// ObjectFile Section iterators
LLVMSectionIteratorRef LLVMGetSections(LLVMObjectFileRef ObjectFile) {
  section_iterator SI = unwrap(ObjectFile)->begin_sections();
  return wrap(new section_iterator(SI));
}

void LLVMDisposeSectionIterator(LLVMSectionIteratorRef SI) {
  delete unwrap(SI);
}

LLVMBool LLVMIsSectionIteratorAtEnd(LLVMObjectFileRef ObjectFile,
                                LLVMSectionIteratorRef SI) {
  return (*unwrap(SI) == unwrap(ObjectFile)->end_sections()) ? 1 : 0;
}

void LLVMMoveToNextSection(LLVMSectionIteratorRef SI) {
  error_code ec;
  unwrap(SI)->increment(ec);
  if (ec) report_fatal_error("LLVMMoveToNextSection failed: " + ec.message());
}

void LLVMMoveToContainingSection(LLVMSectionIteratorRef Sect,
                                 LLVMSymbolIteratorRef Sym) {
  if (error_code ec = (*unwrap(Sym))->getSection(*unwrap(Sect)))
    report_fatal_error(ec.message());
}

// ObjectFile Symbol iterators
LLVMSymbolIteratorRef LLVMGetSymbols(LLVMObjectFileRef ObjectFile) {
  symbol_iterator SI = unwrap(ObjectFile)->begin_symbols();
  return wrap(new symbol_iterator(SI));
}

void LLVMDisposeSymbolIterator(LLVMSymbolIteratorRef SI) {
  delete unwrap(SI);
}

LLVMBool LLVMIsSymbolIteratorAtEnd(LLVMObjectFileRef ObjectFile,
                                LLVMSymbolIteratorRef SI) {
  return (*unwrap(SI) == unwrap(ObjectFile)->end_symbols()) ? 1 : 0;
}

void LLVMMoveToNextSymbol(LLVMSymbolIteratorRef SI) {
  error_code ec;
  unwrap(SI)->increment(ec);
  if (ec) report_fatal_error("LLVMMoveToNextSymbol failed: " + ec.message());
}

// SectionRef accessors
const char *LLVMGetSectionName(LLVMSectionIteratorRef SI) {
  StringRef ret;
  if (error_code ec = (*unwrap(SI))->getName(ret))
   report_fatal_error(ec.message());
  return ret.data();
}

uint64_t LLVMGetSectionSize(LLVMSectionIteratorRef SI) {
  uint64_t ret;
  if (error_code ec = (*unwrap(SI))->getSize(ret))
    report_fatal_error(ec.message());
  return ret;
}

const char *LLVMGetSectionContents(LLVMSectionIteratorRef SI) {
  StringRef ret;
  if (error_code ec = (*unwrap(SI))->getContents(ret))
    report_fatal_error(ec.message());
  return ret.data();
}

uint64_t LLVMGetSectionAddress(LLVMSectionIteratorRef SI) {
  uint64_t ret;
  if (error_code ec = (*unwrap(SI))->getAddress(ret))
    report_fatal_error(ec.message());
  return ret;
}

LLVMBool LLVMGetSectionContainsSymbol(LLVMSectionIteratorRef SI,
                                 LLVMSymbolIteratorRef Sym) {
  bool ret;
  if (error_code ec = (*unwrap(SI))->containsSymbol(**unwrap(Sym), ret))
    report_fatal_error(ec.message());
  return ret;
}

// Section Relocation iterators
LLVMRelocationIteratorRef LLVMGetRelocations(LLVMSectionIteratorRef Section) {
  relocation_iterator SI = (*unwrap(Section))->begin_relocations();
  return wrap(new relocation_iterator(SI));
}

void LLVMDisposeRelocationIterator(LLVMRelocationIteratorRef SI) {
  delete unwrap(SI);
}

LLVMBool LLVMIsRelocationIteratorAtEnd(LLVMSectionIteratorRef Section,
                                       LLVMRelocationIteratorRef SI) {
  return (*unwrap(SI) == (*unwrap(Section))->end_relocations()) ? 1 : 0;
}

void LLVMMoveToNextRelocation(LLVMRelocationIteratorRef SI) {
  error_code ec;
  unwrap(SI)->increment(ec);
  if (ec) report_fatal_error("LLVMMoveToNextRelocation failed: " +
                             ec.message());
}


// SymbolRef accessors
const char *LLVMGetSymbolName(LLVMSymbolIteratorRef SI) {
  StringRef ret;
  if (error_code ec = (*unwrap(SI))->getName(ret))
    report_fatal_error(ec.message());
  return ret.data();
}

uint64_t LLVMGetSymbolAddress(LLVMSymbolIteratorRef SI) {
  uint64_t ret;
  if (error_code ec = (*unwrap(SI))->getAddress(ret))
    report_fatal_error(ec.message());
  return ret;
}

uint64_t LLVMGetSymbolFileOffset(LLVMSymbolIteratorRef SI) {
  uint64_t ret;
  if (error_code ec = (*unwrap(SI))->getFileOffset(ret))
    report_fatal_error(ec.message());
  return ret;
}

uint64_t LLVMGetSymbolSize(LLVMSymbolIteratorRef SI) {
  uint64_t ret;
  if (error_code ec = (*unwrap(SI))->getSize(ret))
    report_fatal_error(ec.message());
  return ret;
}

// RelocationRef accessors
uint64_t LLVMGetRelocationAddress(LLVMRelocationIteratorRef RI) {
  uint64_t ret;
  if (error_code ec = (*unwrap(RI))->getAddress(ret))
    report_fatal_error(ec.message());
  return ret;
}

uint64_t LLVMGetRelocationOffset(LLVMRelocationIteratorRef RI) {
  uint64_t ret;
  if (error_code ec = (*unwrap(RI))->getOffset(ret))
    report_fatal_error(ec.message());
  return ret;
}

LLVMSymbolIteratorRef LLVMGetRelocationSymbol(LLVMRelocationIteratorRef RI) {
  SymbolRef ret;
  if (error_code ec = (*unwrap(RI))->getSymbol(ret))
    report_fatal_error(ec.message());

  return wrap(new symbol_iterator(ret));
}

uint64_t LLVMGetRelocationType(LLVMRelocationIteratorRef RI) {
  uint64_t ret;
  if (error_code ec = (*unwrap(RI))->getType(ret))
    report_fatal_error(ec.message());
  return ret;
}

// NOTE: Caller takes ownership of returned string.
const char *LLVMGetRelocationTypeName(LLVMRelocationIteratorRef RI) {
  SmallVector<char, 0> ret;
  if (error_code ec = (*unwrap(RI))->getTypeName(ret))
    report_fatal_error(ec.message());

  char *str = static_cast<char*>(malloc(ret.size()));
  std::copy(ret.begin(), ret.end(), str);
  return str;
}

// NOTE: Caller takes ownership of returned string.
const char *LLVMGetRelocationValueString(LLVMRelocationIteratorRef RI) {
  SmallVector<char, 0> ret;
  if (error_code ec = (*unwrap(RI))->getValueString(ret))
    report_fatal_error(ec.message());

  char *str = static_cast<char*>(malloc(ret.size()));
  std::copy(ret.begin(), ret.end(), str);
  return str;
}

