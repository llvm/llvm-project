//===-- llvm/Assembly/Parser.h - Parser for VM assembly files ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  These classes are implemented by the lib/AsmParser library.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ASSEMBLY_PARSER_H
#define LLVM_ASSEMBLY_PARSER_H

#include <string>

namespace llvm {

class Module;
class ParseError;
class raw_ostream;

/// This function is the main interface to the LLVM Assembly Parser. It parses
/// an ASCII file that (presumably) contains LLVM Assembly code. It returns a
/// Module (intermediate representation) with the corresponding features. Note
/// that this does not verify that the generated Module is valid, so you should
/// run the verifier after parsing the file to check that it is okay.
/// @brief Parse LLVM Assembly from a file
Module *ParseAssemblyFile(
  const std::string &Filename, ///< The name of the file to parse
  ParseError &Error            ///< If not null, an object to return errors in.
);

/// The function is a secondary interface to the LLVM Assembly Parser. It parses
/// an ASCII string that (presumably) contains LLVM Assembly code. It returns a
/// Module (intermediate representation) with the corresponding features. Note
/// that this does not verify that the generated Module is valid, so you should
/// run the verifier after parsing the file to check that it is okay.
/// @brief Parse LLVM Assembly from a string
Module *ParseAssemblyString(
  const char *AsmString, ///< The string containing assembly
  Module *M,             ///< A module to add the assembly too.
  ParseError &Error      ///< If not null, an object to return errors in.
);

//===------------------------------------------------------------------------===
//                              Helper Classes
//===------------------------------------------------------------------------===

/// An instance of this class can be passed to ParseAssemblyFile or 
/// ParseAssemblyString functions in order to capture error information from
/// the parser.  It provides a standard way to print out the error message
/// including the file name and line number where the error occurred.
/// @brief An LLVM Assembly Parsing Error Object
class ParseError {
public:
  ParseError() : Filename("unknown"), Message("none"), LineNo(0), ColumnNo(0) {}
  ParseError(const ParseError &E);

  void setFilename(const std::string &F) { Filename = F; }
  
  inline const std::string &getRawMessage() const {   // Just the raw message.
    return Message;
  }

  inline const std::string &getFilename() const {
    return Filename;
  }

  void setError(const std::string &message, int lineNo = -1, int ColNo = -1,
                const std::string &FileContents = "") {
    Message = message;
    LineNo = lineNo; ColumnNo = ColNo;
    LineContents = FileContents;
  }

  // getErrorLocation - Return the line and column number of the error in the
  // input source file.  The source filename can be derived from the
  // ParserOptions in effect.  If positional information is not applicable,
  // these will return a value of -1.
  //
  inline void getErrorLocation(int &Line, int &Column) const {
    Line = LineNo; Column = ColumnNo;
  }
  
  void PrintError(const char *ProgName, raw_ostream &S);

private :
  std::string Filename;
  std::string Message;
  int LineNo, ColumnNo;                               // -1 if not relevant
  std::string LineContents;

  void operator=(const ParseError &E); // DO NOT IMPLEMENT
};

} // End llvm namespace

#endif
