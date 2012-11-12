//===-EDToken.h - LLVM Enhanced Disassembler --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines the interface for the Enhanced Disassembly library's token
// class.  The token is responsible for vending information about the token, 
// such as its type and logical value.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EDTOKEN_H
#define LLVM_EDTOKEN_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DataTypes.h"
#include <string>
#include <vector>

namespace llvm {
  
struct EDDisassembler;

/// EDToken - Encapsulates a single token, which can provide a string
///   representation of itself or interpret itself in various ways, depending
///   on the token type.
struct EDToken {
  enum tokenType {
    kTokenWhitespace,
    kTokenOpcode,
    kTokenLiteral,
    kTokenRegister,
    kTokenPunctuation
  };
  
  /// The parent disassembler
  EDDisassembler &Disassembler;

  /// The token's string representation
  llvm::StringRef Str;
  /// The token's string representation, but in a form suitable for export
  std::string PermStr;
  /// The type of the token, as exposed through the external API
  enum tokenType Type;
  /// The type of the token, as recorded by the syntax-specific tokenizer
  uint64_t LocalType;
  /// The operand corresponding to the token, or (unsigned int)-1 if not
  ///   part of an operand.
  int OperandID;
  
  /// The sign if the token is a literal (1 if negative, 0 otherwise)
  bool LiteralSign;
  /// The absolute value if the token is a literal
  uint64_t LiteralAbsoluteValue;
  /// The LLVM register ID if the token is a register name
  unsigned RegisterID;
  
  /// Constructor - Initializes an EDToken with the information common to all
  ///   tokens
  ///
  /// @arg str          - The string corresponding to the token
  /// @arg type         - The token's type as exposed through the public API
  /// @arg localType    - The token's type as recorded by the tokenizer
  /// @arg disassembler - The disassembler responsible for the token
  EDToken(llvm::StringRef str,
          enum tokenType type,
          uint64_t localType,
          EDDisassembler &disassembler);
  
  /// makeLiteral - Adds the information specific to a literal
  /// @arg sign           - The sign of the literal (1 if negative, 0 
  ///                       otherwise)
  ///
  /// @arg absoluteValue  - The absolute value of the literal
  void makeLiteral(bool sign, uint64_t absoluteValue);
  /// makeRegister - Adds the information specific to a register
  ///
  /// @arg registerID - The LLVM register ID
  void makeRegister(unsigned registerID);
  
  /// setOperandID - Links the token to a numbered operand
  ///
  /// @arg operandID  - The operand ID to link to
  void setOperandID(int operandID);
  
  ~EDToken();
  
  /// type - Returns the public type of the token
  enum tokenType type() const;
  /// localType - Returns the tokenizer-specific type of the token
  uint64_t localType() const;
  /// string - Returns the string representation of the token
  llvm::StringRef string() const;
  /// operandID - Returns the operand ID of the token
  int operandID() const;
  
  /// literalSign - Returns the sign of the token 
  ///   (1 if negative, 0 if positive or unsigned, -1 if it is not a literal)
  int literalSign() const;
  /// literalAbsoluteValue - Retrieves the absolute value of the token, and
  ///   returns -1 if the token is not a literal
  /// @arg value  - A reference to a value that is filled in with the absolute
  ///               value, if it is valid
  int literalAbsoluteValue(uint64_t &value) const;
  /// registerID - Retrieves the register ID of the token, and returns -1 if the
  ///   token is not a register
  ///
  /// @arg registerID - A reference to a value that is filled in with the 
  ///                   register ID, if it is valid
  int registerID(unsigned &registerID) const;
  
  /// tokenize - Tokenizes a string using the platform- and syntax-specific
  ///   tokenizer, and returns 0 on success (-1 on failure)
  ///
  /// @arg tokens       - A vector that will be filled in with pointers to
  ///                     allocated tokens
  /// @arg str          - The string, as outputted by the AsmPrinter
  /// @arg operandOrder - The order of the operands from the operandFlags array
  ///                     as they appear in str
  /// @arg disassembler - The disassembler for the desired target and
  //                      assembly syntax
  static int tokenize(std::vector<EDToken*> &tokens,
                      std::string &str,
                      const signed char *operandOrder,
                      EDDisassembler &disassembler);
  
  /// getString - Directs a character pointer to the string, returning 0 on
  ///   success (-1 on failure)
  /// @arg buf  - A reference to a pointer that is set to point to the string.
  ///   The string is still owned by the token.
  int getString(const char*& buf);
};

} // end namespace llvm
#endif
