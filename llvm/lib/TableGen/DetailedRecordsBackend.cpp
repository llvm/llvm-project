//===- DetailedRecordBackend.cpp - Detailed Records Report      -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This Tablegen backend prints a report that includes all the global 
// variables, classes, and records in complete detail. It includes more
// detail than the default TableGen printer backend.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include <string>
#include <utility>

using namespace llvm;

namespace {

class DetailedRecordsEmitter {
private:
  const RecordKeeper &Records;

public:
  explicit DetailedRecordsEmitter(const RecordKeeper &RK) : Records(RK) {}

  void run(raw_ostream &OS);
  void printReportHeading(raw_ostream &OS);
  void printSectionHeading(StringRef Title, int Count, raw_ostream &OS);
  void printVariables(raw_ostream &OS);
  void printClasses(raw_ostream &OS);
  void printRecords(raw_ostream &OS);
  void printAllocationStats(raw_ostream &OS);
  void printDefms(const Record &Rec, raw_ostream &OS);
  void printTemplateArgs(const Record &Rec, raw_ostream &OS);
  void printSuperclasses(const Record &Rec, raw_ostream &OS);
  void printFields(const Record &Rec, raw_ostream &OS);
}; // emitter class

} // anonymous namespace

// Print the report.
void DetailedRecordsEmitter::run(raw_ostream &OS) {
  printReportHeading(OS);
  printVariables(OS);
  printClasses(OS);
  printRecords(OS);
  printAllocationStats(OS);
}

// Print the report heading, including the source file name.
void DetailedRecordsEmitter::printReportHeading(raw_ostream &OS) {
  OS << formatv("DETAILED RECORDS for file {0}\n", Records.getInputFilename());
}

// Print a section heading with the name of the section and the item count.
void DetailedRecordsEmitter::printSectionHeading(StringRef Title, int Count,
                                                 raw_ostream &OS) {
  OS << formatv("\n{0} {1} ({2}) {0}\n", "--------------------", Title, Count);
}

// Print the global variables.
void DetailedRecordsEmitter::printVariables(raw_ostream &OS) {
  const auto GlobalList = Records.getGlobals();
  printSectionHeading("Global Variables", GlobalList.size(), OS);

  OS << '\n';
  for (const auto &Var : GlobalList)
    OS << Var.first << " = " << Var.second->getAsString() << '\n';
}

// Print classes, including the template arguments, superclasses, and fields.
void DetailedRecordsEmitter::printClasses(raw_ostream &OS) {
  const auto &ClassList = Records.getClasses();
  printSectionHeading("Classes", ClassList.size(), OS);

  for (const auto &[Name, Class] : ClassList) {
    OS << formatv("\n{0}  |{1}|\n", Class->getNameInitAsString(),
                  SrcMgr.getFormattedLocationNoOffset(Class->getLoc().front()));
    printTemplateArgs(*Class, OS);
    printSuperclasses(*Class, OS);
    printFields(*Class, OS);
  }
}

// Print the records, including the defm sequences, supercasses, and fields.
void DetailedRecordsEmitter::printRecords(raw_ostream &OS) {
  const auto &RecordList = Records.getDefs();
  printSectionHeading("Records", RecordList.size(), OS);

  for (const auto &[DefName, Rec] : RecordList) {
    std::string Name = Rec->getNameInitAsString();
    OS << formatv("\n{0}  |{1}|\n", Name.empty() ? "\"\"" : Name,
                  SrcMgr.getFormattedLocationNoOffset(Rec->getLoc().front()));
    printDefms(*Rec, OS);
    printSuperclasses(*Rec, OS);
    printFields(*Rec, OS);
  }
}

// Print memory allocation related stats.
void DetailedRecordsEmitter::printAllocationStats(raw_ostream &OS) {
  OS << formatv("\n{0} Memory Allocation Stats {0}\n", "--------------------");
  Records.dumpAllocationStats(OS);
}

// Print the record's defm source locations, if any. Note that they
// are stored in the reverse order of their invocation.
void DetailedRecordsEmitter::printDefms(const Record &Rec, raw_ostream &OS) {
  const auto &LocList = Rec.getLoc();
  if (LocList.size() < 2)
    return;

  OS << "  Defm sequence:";
  for (const SMLoc Loc : reverse(LocList))
    OS << formatv(" |{0}|", SrcMgr.getFormattedLocationNoOffset(Loc));
  OS << '\n';
}

// Print the template arguments of a class.
void DetailedRecordsEmitter::printTemplateArgs(const Record &Rec,
                                               raw_ostream &OS) {
  ArrayRef<const Init *> Args = Rec.getTemplateArgs();
  if (Args.empty()) {
    OS << "  Template args: (none)\n";
    return;
  }

  OS << "  Template args:\n";
  for (const Init *ArgName : Args) {
    const RecordVal *Value = Rec.getValue(ArgName);
    assert(Value && "Template argument value not found.");
    OS << "    ";
    Value->print(OS, false);
    OS << formatv("  |{0}|\n",
                  SrcMgr.getFormattedLocationNoOffset(Value->getLoc()));
  }
}

// Print the superclasses of a class or record. Indirect superclasses
// are enclosed in parentheses.
void DetailedRecordsEmitter::printSuperclasses(const Record &Rec,
                                               raw_ostream &OS) {
  ArrayRef<std::pair<const Record *, SMRange>> Superclasses =
      Rec.getSuperClasses();
  if (Superclasses.empty()) {
    OS << "  Superclasses: (none)\n";
    return;
  }

  OS << "  Superclasses:";
  for (const auto &[ClassRec, Loc] : Superclasses) {
    if (Rec.hasDirectSuperClass(ClassRec))
      OS << formatv(" {0}", ClassRec->getNameInitAsString());
    else
      OS << formatv(" ({0})", ClassRec->getNameInitAsString());
  }
  OS << '\n';
}

// Print the fields of a class or record, including their source locations.
void DetailedRecordsEmitter::printFields(const Record &Rec, raw_ostream &OS) {
  const auto &ValueList = Rec.getValues();
  if (ValueList.empty()) {
    OS << "  Fields: (none)\n";
    return;
  }

  OS << "  Fields:\n";
  for (const RecordVal &Value : ValueList)
    if (!Rec.isTemplateArg(Value.getNameInit())) {
      OS << "    ";
      Value.print(OS, false);
      OS << formatv("  |{0}|\n",
                    SrcMgr.getFormattedLocationNoOffset(Value.getLoc()));
    }
}

// This function is called by TableGen after parsing the files.
void llvm::EmitDetailedRecords(const RecordKeeper &RK, raw_ostream &OS) {
  // Instantiate the emitter class and invoke run().
  DetailedRecordsEmitter(RK).run(OS);
}
