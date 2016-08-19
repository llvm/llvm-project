//===- MILexer.h - Lexer for machine instructions -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the function that lexes the machine instruction source
// string.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_MIRPARSER_MILEXER_H
#define LLVM_LIB_CODEGEN_MIRPARSER_MILEXER_H

#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/STLExtras.h"
#include <functional>

namespace llvm {

class Twine;

/// A token produced by the machine instruction lexer.
struct MIToken {
  enum TokenKind {
    // Markers
    Eof,
    Error,
    Newline,

    // Tokens with no info.
    comma,
    equal,
    underscore,
    colon,
    coloncolon,
    dot,
    exclaim,
    lparen,
    rparen,
    lbrace,
    rbrace,
    plus,
    minus,
    less,
    greater,

    // Keywords
    kw_implicit,
    kw_implicit_define,
    kw_def,
    kw_dead,
    kw_killed,
    kw_undef,
    kw_internal,
    kw_early_clobber,
    kw_debug_use,
    kw_tied_def,
    kw_frame_setup,
    kw_debug_location,
    kw_cfi_same_value,
    kw_cfi_offset,
    kw_cfi_def_cfa_register,
    kw_cfi_def_cfa_offset,
    kw_cfi_def_cfa,
    kw_blockaddress,
    kw_intrinsic,
    kw_target_index,
    kw_half,
    kw_float,
    kw_double,
    kw_x86_fp80,
    kw_fp128,
    kw_ppc_fp128,
    kw_target_flags,
    kw_volatile,
    kw_non_temporal,
    kw_invariant,
    kw_align,
    kw_stack,
    kw_got,
    kw_jump_table,
    kw_constant_pool,
    kw_call_entry,
    kw_liveout,
    kw_address_taken,
    kw_landing_pad,
    kw_liveins,
    kw_successors,
    kw_floatpred,
    kw_intpred,

    // Named metadata keywords
    md_tbaa,
    md_alias_scope,
    md_noalias,
    md_range,

    // Identifier tokens
    Identifier,
    IntegerType,
    NamedRegister,
    MachineBasicBlockLabel,
    MachineBasicBlock,
    PointerType,
    ScalarType,
    StackObject,
    FixedStackObject,
    NamedGlobalValue,
    GlobalValue,
    ExternalSymbol,

    // Other tokens
    IntegerLiteral,
    FloatingPointLiteral,
    VirtualRegister,
    ConstantPoolItem,
    JumpTableIndex,
    NamedIRBlock,
    IRBlock,
    NamedIRValue,
    IRValue,
    QuotedIRValue, // `<constant value>`
    SubRegisterIndex
  };

private:
  TokenKind Kind;
  StringRef Range;
  StringRef StringValue;
  std::string StringValueStorage;
  APSInt IntVal;

public:
  MIToken() : Kind(Error) {}

  MIToken &reset(TokenKind Kind, StringRef Range);

  MIToken &setStringValue(StringRef StrVal);
  MIToken &setOwnedStringValue(std::string StrVal);
  MIToken &setIntegerValue(APSInt IntVal);

  TokenKind kind() const { return Kind; }

  bool isError() const { return Kind == Error; }

  bool isNewlineOrEOF() const { return Kind == Newline || Kind == Eof; }

  bool isErrorOrEOF() const { return Kind == Error || Kind == Eof; }

  bool isRegister() const {
    return Kind == NamedRegister || Kind == underscore ||
           Kind == VirtualRegister;
  }

  bool isRegisterFlag() const {
    return Kind == kw_implicit || Kind == kw_implicit_define ||
           Kind == kw_def || Kind == kw_dead || Kind == kw_killed ||
           Kind == kw_undef || Kind == kw_internal ||
           Kind == kw_early_clobber || Kind == kw_debug_use;
  }

  bool isMemoryOperandFlag() const {
    return Kind == kw_volatile || Kind == kw_non_temporal ||
           Kind == kw_invariant;
  }

  bool is(TokenKind K) const { return Kind == K; }

  bool isNot(TokenKind K) const { return Kind != K; }

  StringRef::iterator location() const { return Range.begin(); }

  StringRef range() const { return Range; }

  /// Return the token's string value.
  StringRef stringValue() const { return StringValue; }

  const APSInt &integerValue() const { return IntVal; }

  bool hasIntegerValue() const {
    return Kind == IntegerLiteral || Kind == MachineBasicBlock ||
           Kind == MachineBasicBlockLabel || Kind == StackObject ||
           Kind == FixedStackObject || Kind == GlobalValue ||
           Kind == VirtualRegister || Kind == ConstantPoolItem ||
           Kind == JumpTableIndex || Kind == IRBlock || Kind == IRValue;
  }
};

/// Consume a single machine instruction token in the given source and return
/// the remaining source string.
StringRef lexMIToken(
    StringRef Source, MIToken &Token,
    function_ref<void(StringRef::iterator, const Twine &)> ErrorCallback);

} // end namespace llvm

#endif
