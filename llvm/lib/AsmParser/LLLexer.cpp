//===- LLLexer.cpp - Lexer for .ll Files ----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implement the Lexer for .ll files.
//
//===----------------------------------------------------------------------===//

#include "LLLexer.h"
#include "ParserInternals.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/MathExtras.h"

#include <list>
#include "llvmAsmParser.h"

#include <cstring>
using namespace llvm;

//===----------------------------------------------------------------------===//
// Helper functions.
//===----------------------------------------------------------------------===//

// atoull - Convert an ascii string of decimal digits into the unsigned long
// long representation... this does not have to do input error checking,
// because we know that the input will be matched by a suitable regex...
//
static uint64_t atoull(const char *Buffer, const char *End) {
  uint64_t Result = 0;
  for (; Buffer != End; Buffer++) {
    uint64_t OldRes = Result;
    Result *= 10;
    Result += *Buffer-'0';
    if (Result < OldRes) {  // Uh, oh, overflow detected!!!
      GenerateError("constant bigger than 64 bits detected!");
      return 0;
    }
  }
  return Result;
}

static uint64_t HexIntToVal(const char *Buffer, const char *End) {
  uint64_t Result = 0;
  for (; Buffer != End; ++Buffer) {
    uint64_t OldRes = Result;
    Result *= 16;
    char C = *Buffer;
    if (C >= '0' && C <= '9')
      Result += C-'0';
    else if (C >= 'A' && C <= 'F')
      Result += C-'A'+10;
    else if (C >= 'a' && C <= 'f')
      Result += C-'a'+10;

    if (Result < OldRes) {   // Uh, oh, overflow detected!!!
      GenerateError("constant bigger than 64 bits detected!");
      return 0;
    }
  }
  return Result;
}

// HexToFP - Convert the ascii string in hexadecimal format to the floating
// point representation of it.
//
static double HexToFP(const char *Buffer, const char *End) {
  return BitsToDouble(HexIntToVal(Buffer, End)); // Cast Hex constant to double
}

static void HexToIntPair(const char *Buffer, const char *End, uint64_t Pair[2]){
  Pair[0] = 0;
  for (int i=0; i<16; i++, Buffer++) {
    assert(Buffer != End);
    Pair[0] *= 16;
    char C = *Buffer;
    if (C >= '0' && C <= '9')
      Pair[0] += C-'0';
    else if (C >= 'A' && C <= 'F')
      Pair[0] += C-'A'+10;
    else if (C >= 'a' && C <= 'f')
      Pair[0] += C-'a'+10;
  }
  Pair[1] = 0;
  for (int i=0; i<16 && Buffer != End; i++, Buffer++) {
    Pair[1] *= 16;
    char C = *Buffer;
    if (C >= '0' && C <= '9')
      Pair[1] += C-'0';
    else if (C >= 'A' && C <= 'F')
      Pair[1] += C-'A'+10;
    else if (C >= 'a' && C <= 'f')
      Pair[1] += C-'a'+10;
  }
  if (Buffer != End)
    GenerateError("constant bigger than 128 bits detected!");
}

// UnEscapeLexed - Run through the specified buffer and change \xx codes to the
// appropriate character.
static void UnEscapeLexed(std::string &Str) {
  if (Str.empty()) return;

  char *Buffer = &Str[0], *EndBuffer = Buffer+Str.size();
  char *BOut = Buffer;
  for (char *BIn = Buffer; BIn != EndBuffer; ) {
    if (BIn[0] == '\\') {
      if (BIn < EndBuffer-1 && BIn[1] == '\\') {
        *BOut++ = '\\'; // Two \ becomes one
        BIn += 2;
      } else if (BIn < EndBuffer-2 && isxdigit(BIn[1]) && isxdigit(BIn[2])) {
        char Tmp = BIn[3]; BIn[3] = 0;      // Terminate string
        *BOut = (char)strtol(BIn+1, 0, 16); // Convert to number
        BIn[3] = Tmp;                       // Restore character
        BIn += 3;                           // Skip over handled chars
        ++BOut;
      } else {
        *BOut++ = *BIn++;
      }
    } else {
      *BOut++ = *BIn++;
    }
  }
  Str.resize(BOut-Buffer);
}

/// isLabelChar - Return true for [-a-zA-Z$._0-9].
static bool isLabelChar(char C) {
  return isalnum(C) || C == '-' || C == '$' || C == '.' || C == '_';
}


/// isLabelTail - Return true if this pointer points to a valid end of a label.
static const char *isLabelTail(const char *CurPtr) {
  while (1) {
    if (CurPtr[0] == ':') return CurPtr+1;
    if (!isLabelChar(CurPtr[0])) return 0;
    ++CurPtr;
  }
}



//===----------------------------------------------------------------------===//
// Lexer definition.
//===----------------------------------------------------------------------===//

// FIXME: REMOVE THIS.
#define YYEOF 0
#define YYERROR -2

LLLexer::LLLexer(MemoryBuffer *StartBuf) : CurLineNo(1), CurBuf(StartBuf) {
  CurPtr = CurBuf->getBufferStart();
}

std::string LLLexer::getFilename() const {
  return CurBuf->getBufferIdentifier();
}

int LLLexer::getNextChar() {
  char CurChar = *CurPtr++;
  switch (CurChar) {
  default: return (unsigned char)CurChar;
  case 0:
    // A nul character in the stream is either the end of the current buffer or
    // a random nul in the file.  Disambiguate that here.
    if (CurPtr-1 != CurBuf->getBufferEnd())
      return 0;  // Just whitespace.

    // Otherwise, return end of file.
    --CurPtr;  // Another call to lex will return EOF again.
    return EOF;
  case '\n':
  case '\r':
    // Handle the newline character by ignoring it and incrementing the line
    // count.  However, be careful about 'dos style' files with \n\r in them.
    // Only treat a \n\r or \r\n as a single line.
    if ((*CurPtr == '\n' || (*CurPtr == '\r')) &&
        *CurPtr != CurChar)
      ++CurPtr;  // Eat the two char newline sequence.

    ++CurLineNo;
    return '\n';
  }
}


int LLLexer::LexToken() {
  TokStart = CurPtr;

  int CurChar = getNextChar();

  switch (CurChar) {
  default:
    // Handle letters: [a-zA-Z_]
    if (isalpha(CurChar) || CurChar == '_')
      return LexIdentifier();

    return CurChar;
  case EOF: return YYEOF;
  case 0:
  case ' ':
  case '\t':
  case '\n':
  case '\r':
    // Ignore whitespace.
    return LexToken();
  case '+': return LexPositive();
  case '@': return LexAt();
  case '%': return LexPercent();
  case '"': return LexQuote();
  case '.':
    if (const char *Ptr = isLabelTail(CurPtr)) {
      CurPtr = Ptr;
      llvmAsmlval.StrVal = new std::string(TokStart, CurPtr-1);
      return LABELSTR;
    }
    if (CurPtr[0] == '.' && CurPtr[1] == '.') {
      CurPtr += 2;
      return DOTDOTDOT;
    }
    return '.';
  case '$':
    if (const char *Ptr = isLabelTail(CurPtr)) {
      CurPtr = Ptr;
      llvmAsmlval.StrVal = new std::string(TokStart, CurPtr-1);
      return LABELSTR;
    }
    return '$';
  case ';':
    SkipLineComment();
    return LexToken();
  case '0': case '1': case '2': case '3': case '4':
  case '5': case '6': case '7': case '8': case '9':
  case '-':
    return LexDigitOrNegative();
  }
}

void LLLexer::SkipLineComment() {
  while (1) {
    if (CurPtr[0] == '\n' || CurPtr[0] == '\r' || getNextChar() == EOF)
      return;
  }
}

/// LexAt - Lex all tokens that start with an @ character:
///   AtStringConstant @\"[^\"]*\"
///   GlobalVarName    @[-a-zA-Z$._][-a-zA-Z$._0-9]*
///   GlobalVarID      @[0-9]+
int LLLexer::LexAt() {
  // Handle AtStringConstant: @\"[^\"]*\"
  if (CurPtr[0] == '"') {
    ++CurPtr;

    while (1) {
      int CurChar = getNextChar();

      if (CurChar == EOF) {
        GenerateError("End of file in global variable name");
        return YYERROR;
      }
      if (CurChar == '"') {
        llvmAsmlval.StrVal = new std::string(TokStart+2, CurPtr-1);
        UnEscapeLexed(*llvmAsmlval.StrVal);
        return ATSTRINGCONSTANT;
      }
    }
  }

  // Handle GlobalVarName: @[-a-zA-Z$._][-a-zA-Z$._0-9]*
  if (isalpha(CurPtr[0]) || CurPtr[0] == '-' || CurPtr[0] == '$' ||
      CurPtr[0] == '.' || CurPtr[0] == '_') {
    ++CurPtr;
    while (isalnum(CurPtr[0]) || CurPtr[0] == '-' || CurPtr[0] == '$' ||
           CurPtr[0] == '.' || CurPtr[0] == '_')
      ++CurPtr;

    llvmAsmlval.StrVal = new std::string(TokStart+1, CurPtr);   // Skip @
    return GLOBALVAR;
  }

  // Handle GlobalVarID: @[0-9]+
  if (isdigit(CurPtr[0])) {
    for (++CurPtr; isdigit(CurPtr[0]); ++CurPtr)
      /*empty*/;

    uint64_t Val = atoull(TokStart+1, CurPtr);
    if ((unsigned)Val != Val)
      GenerateError("Invalid value number (too large)!");
    llvmAsmlval.UIntVal = unsigned(Val);
    return GLOBALVAL_ID;
  }

  return '@';
}


/// LexPercent - Lex all tokens that start with a % character:
///   PctStringConstant  %\"[^\"]*\"
///   LocalVarName       %[-a-zA-Z$._][-a-zA-Z$._0-9]*
///   LocalVarID         %[0-9]+
int LLLexer::LexPercent() {
  // Handle PctStringConstant: %\"[^\"]*\"
  if (CurPtr[0] == '"') {
    ++CurPtr;

    while (1) {
      int CurChar = getNextChar();

      if (CurChar == EOF) {
        GenerateError("End of file in local variable name");
        return YYERROR;
      }
      if (CurChar == '"') {
        llvmAsmlval.StrVal = new std::string(TokStart+2, CurPtr-1);
        UnEscapeLexed(*llvmAsmlval.StrVal);
        return PCTSTRINGCONSTANT;
      }
    }
  }

  // Handle LocalVarName: %[-a-zA-Z$._][-a-zA-Z$._0-9]*
  if (isalpha(CurPtr[0]) || CurPtr[0] == '-' || CurPtr[0] == '$' ||
      CurPtr[0] == '.' || CurPtr[0] == '_') {
    ++CurPtr;
    while (isalnum(CurPtr[0]) || CurPtr[0] == '-' || CurPtr[0] == '$' ||
           CurPtr[0] == '.' || CurPtr[0] == '_')
      ++CurPtr;

    llvmAsmlval.StrVal = new std::string(TokStart+1, CurPtr);   // Skip %
    return LOCALVAR;
  }

  // Handle LocalVarID: %[0-9]+
  if (isdigit(CurPtr[0])) {
    for (++CurPtr; isdigit(CurPtr[0]); ++CurPtr)
      /*empty*/;

    uint64_t Val = atoull(TokStart+1, CurPtr);
    if ((unsigned)Val != Val)
      GenerateError("Invalid value number (too large)!");
    llvmAsmlval.UIntVal = unsigned(Val);
    return LOCALVAL_ID;
  }

  return '%';
}

/// LexQuote - Lex all tokens that start with a " character:
///   QuoteLabel        "[^"]+":
///   StringConstant    "[^"]*"
int LLLexer::LexQuote() {
  while (1) {
    int CurChar = getNextChar();

    if (CurChar == EOF) {
      GenerateError("End of file in quoted string");
      return YYERROR;
    }

    if (CurChar != '"') continue;

    if (CurPtr[0] != ':') {
      llvmAsmlval.StrVal = new std::string(TokStart+1, CurPtr-1);
      UnEscapeLexed(*llvmAsmlval.StrVal);
      return STRINGCONSTANT;
    }

    ++CurPtr;
    llvmAsmlval.StrVal = new std::string(TokStart+1, CurPtr-2);
    UnEscapeLexed(*llvmAsmlval.StrVal);
    return LABELSTR;
  }
}

static bool JustWhitespaceNewLine(const char *&Ptr) {
  const char *ThisPtr = Ptr;
  while (*ThisPtr == ' ' || *ThisPtr == '\t')
    ++ThisPtr;
  if (*ThisPtr == '\n' || *ThisPtr == '\r') {
    Ptr = ThisPtr;
    return true;
  }
  return false;
}


/// LexIdentifier: Handle several related productions:
///    Label           [-a-zA-Z$._0-9]+:
///    IntegerType     i[0-9]+
///    Keyword         sdiv, float, ...
///    HexIntConstant  [us]0x[0-9A-Fa-f]+
int LLLexer::LexIdentifier() {
  const char *StartChar = CurPtr;
  const char *IntEnd = CurPtr[-1] == 'i' ? 0 : StartChar;
  const char *KeywordEnd = 0;

  for (; isLabelChar(*CurPtr); ++CurPtr) {
    // If we decide this is an integer, remember the end of the sequence.
    if (!IntEnd && !isdigit(*CurPtr)) IntEnd = CurPtr;
    if (!KeywordEnd && !isalnum(*CurPtr) && *CurPtr != '_') KeywordEnd = CurPtr;
  }

  // If we stopped due to a colon, this really is a label.
  if (*CurPtr == ':') {
    llvmAsmlval.StrVal = new std::string(StartChar-1, CurPtr++);
    return LABELSTR;
  }

  // Otherwise, this wasn't a label.  If this was valid as an integer type,
  // return it.
  if (IntEnd == 0) IntEnd = CurPtr;
  if (IntEnd != StartChar) {
    CurPtr = IntEnd;
    uint64_t NumBits = atoull(StartChar, CurPtr);
    if (NumBits < IntegerType::MIN_INT_BITS ||
        NumBits > IntegerType::MAX_INT_BITS) {
      GenerateError("Bitwidth for integer type out of range!");
      return YYERROR;
    }
    const Type* Ty = IntegerType::get(NumBits);
    llvmAsmlval.PrimType = Ty;
    return INTTYPE;
  }

  // Otherwise, this was a letter sequence.  See which keyword this is.
  if (KeywordEnd == 0) KeywordEnd = CurPtr;
  CurPtr = KeywordEnd;
  --StartChar;
  unsigned Len = CurPtr-StartChar;
#define KEYWORD(STR, TOK) \
  if (Len == strlen(STR) && !memcmp(StartChar, STR, strlen(STR))) return TOK;

  KEYWORD("begin",     BEGINTOK);
  KEYWORD("end",       ENDTOK);
  KEYWORD("true",      TRUETOK);
  KEYWORD("false",     FALSETOK);
  KEYWORD("declare",   DECLARE);
  KEYWORD("define",    DEFINE);
  KEYWORD("global",    GLOBAL);
  KEYWORD("constant",  CONSTANT);

  KEYWORD("internal",  INTERNAL);
  KEYWORD("linkonce",  LINKONCE);
  KEYWORD("weak",      WEAK);
  KEYWORD("appending", APPENDING);
  KEYWORD("dllimport", DLLIMPORT);
  KEYWORD("dllexport", DLLEXPORT);
  KEYWORD("common", COMMON);
  KEYWORD("default", DEFAULT);
  KEYWORD("hidden", HIDDEN);
  KEYWORD("protected", PROTECTED);
  KEYWORD("extern_weak", EXTERN_WEAK);
  KEYWORD("external", EXTERNAL);
  KEYWORD("thread_local", THREAD_LOCAL);
  KEYWORD("zeroinitializer", ZEROINITIALIZER);
  KEYWORD("undef", UNDEF);
  KEYWORD("null", NULL_TOK);
  KEYWORD("to", TO);
  KEYWORD("tail", TAIL);
  KEYWORD("target", TARGET);
  KEYWORD("triple", TRIPLE);
  KEYWORD("deplibs", DEPLIBS);
  KEYWORD("datalayout", DATALAYOUT);
  KEYWORD("volatile", VOLATILE);
  KEYWORD("align", ALIGN);
  KEYWORD("addrspace", ADDRSPACE);
  KEYWORD("section", SECTION);
  KEYWORD("alias", ALIAS);
  KEYWORD("module", MODULE);
  KEYWORD("asm", ASM_TOK);
  KEYWORD("sideeffect", SIDEEFFECT);
  KEYWORD("gc", GC);

  KEYWORD("cc", CC_TOK);
  KEYWORD("ccc", CCC_TOK);
  KEYWORD("fastcc", FASTCC_TOK);
  KEYWORD("coldcc", COLDCC_TOK);
  KEYWORD("x86_stdcallcc", X86_STDCALLCC_TOK);
  KEYWORD("x86_fastcallcc", X86_FASTCALLCC_TOK);

  KEYWORD("signext", SIGNEXT);
  KEYWORD("zeroext", ZEROEXT);
  KEYWORD("inreg", INREG);
  KEYWORD("sret", SRET);
  KEYWORD("nounwind", NOUNWIND);
  KEYWORD("noreturn", NORETURN);
  KEYWORD("noalias", NOALIAS);
  KEYWORD("byval", BYVAL);
  KEYWORD("nest", NEST);
  KEYWORD("readnone", READNONE);
  KEYWORD("readonly", READONLY);

  KEYWORD("noinline", NOINLINE);
  KEYWORD("alwaysinline", ALWAYSINLINE);
  KEYWORD("optsize", OPTSIZE);

  KEYWORD("type", TYPE);
  KEYWORD("opaque", OPAQUE);

  KEYWORD("eq" , EQ);
  KEYWORD("ne" , NE);
  KEYWORD("slt", SLT);
  KEYWORD("sgt", SGT);
  KEYWORD("sle", SLE);
  KEYWORD("sge", SGE);
  KEYWORD("ult", ULT);
  KEYWORD("ugt", UGT);
  KEYWORD("ule", ULE);
  KEYWORD("uge", UGE);
  KEYWORD("oeq", OEQ);
  KEYWORD("one", ONE);
  KEYWORD("olt", OLT);
  KEYWORD("ogt", OGT);
  KEYWORD("ole", OLE);
  KEYWORD("oge", OGE);
  KEYWORD("ord", ORD);
  KEYWORD("uno", UNO);
  KEYWORD("ueq", UEQ);
  KEYWORD("une", UNE);
#undef KEYWORD

  // Keywords for types.
#define TYPEKEYWORD(STR, LLVMTY, TOK) \
  if (Len == strlen(STR) && !memcmp(StartChar, STR, strlen(STR))) { \
    llvmAsmlval.PrimType = LLVMTY; return TOK; }
  TYPEKEYWORD("void",      Type::VoidTy,  VOID);
  TYPEKEYWORD("float",     Type::FloatTy, FLOAT);
  TYPEKEYWORD("double",    Type::DoubleTy, DOUBLE);
  TYPEKEYWORD("x86_fp80",  Type::X86_FP80Ty, X86_FP80);
  TYPEKEYWORD("fp128",     Type::FP128Ty, FP128);
  TYPEKEYWORD("ppc_fp128", Type::PPC_FP128Ty, PPC_FP128);
  TYPEKEYWORD("label",     Type::LabelTy, LABEL);
#undef TYPEKEYWORD

  // Handle special forms for autoupgrading.  Drop these in LLVM 3.0.  This is
  // to avoid conflicting with the sext/zext instructions, below.
  if (Len == 4 && !memcmp(StartChar, "sext", 4)) {
    // Scan CurPtr ahead, seeing if there is just whitespace before the newline.
    if (JustWhitespaceNewLine(CurPtr))
      return SIGNEXT;
  } else if (Len == 4 && !memcmp(StartChar, "zext", 4)) {
    // Scan CurPtr ahead, seeing if there is just whitespace before the newline.
    if (JustWhitespaceNewLine(CurPtr))
      return ZEROEXT;
  }

  // Keywords for instructions.
#define INSTKEYWORD(STR, type, Enum, TOK) \
  if (Len == strlen(STR) && !memcmp(StartChar, STR, strlen(STR))) { \
    llvmAsmlval.type = Instruction::Enum; return TOK; }

  INSTKEYWORD("add",     BinaryOpVal, Add, ADD);
  INSTKEYWORD("sub",     BinaryOpVal, Sub, SUB);
  INSTKEYWORD("mul",     BinaryOpVal, Mul, MUL);
  INSTKEYWORD("udiv",    BinaryOpVal, UDiv, UDIV);
  INSTKEYWORD("sdiv",    BinaryOpVal, SDiv, SDIV);
  INSTKEYWORD("fdiv",    BinaryOpVal, FDiv, FDIV);
  INSTKEYWORD("urem",    BinaryOpVal, URem, UREM);
  INSTKEYWORD("srem",    BinaryOpVal, SRem, SREM);
  INSTKEYWORD("frem",    BinaryOpVal, FRem, FREM);
  INSTKEYWORD("shl",     BinaryOpVal, Shl, SHL);
  INSTKEYWORD("lshr",    BinaryOpVal, LShr, LSHR);
  INSTKEYWORD("ashr",    BinaryOpVal, AShr, ASHR);
  INSTKEYWORD("and",     BinaryOpVal, And, AND);
  INSTKEYWORD("or",      BinaryOpVal, Or , OR );
  INSTKEYWORD("xor",     BinaryOpVal, Xor, XOR);
  INSTKEYWORD("icmp",    OtherOpVal,  ICmp,  ICMP);
  INSTKEYWORD("fcmp",    OtherOpVal,  FCmp,  FCMP);

  INSTKEYWORD("phi",         OtherOpVal, PHI, PHI_TOK);
  INSTKEYWORD("call",        OtherOpVal, Call, CALL);
  INSTKEYWORD("trunc",       CastOpVal, Trunc, TRUNC);
  INSTKEYWORD("zext",        CastOpVal, ZExt, ZEXT);
  INSTKEYWORD("sext",        CastOpVal, SExt, SEXT);
  INSTKEYWORD("fptrunc",     CastOpVal, FPTrunc, FPTRUNC);
  INSTKEYWORD("fpext",       CastOpVal, FPExt, FPEXT);
  INSTKEYWORD("uitofp",      CastOpVal, UIToFP, UITOFP);
  INSTKEYWORD("sitofp",      CastOpVal, SIToFP, SITOFP);
  INSTKEYWORD("fptoui",      CastOpVal, FPToUI, FPTOUI);
  INSTKEYWORD("fptosi",      CastOpVal, FPToSI, FPTOSI);
  INSTKEYWORD("inttoptr",    CastOpVal, IntToPtr, INTTOPTR);
  INSTKEYWORD("ptrtoint",    CastOpVal, PtrToInt, PTRTOINT);
  INSTKEYWORD("bitcast",     CastOpVal, BitCast, BITCAST);
  INSTKEYWORD("select",      OtherOpVal, Select, SELECT);
  INSTKEYWORD("va_arg",      OtherOpVal, VAArg , VAARG);
  INSTKEYWORD("ret",         TermOpVal, Ret, RET);
  INSTKEYWORD("br",          TermOpVal, Br, BR);
  INSTKEYWORD("switch",      TermOpVal, Switch, SWITCH);
  INSTKEYWORD("invoke",      TermOpVal, Invoke, INVOKE);
  INSTKEYWORD("unwind",      TermOpVal, Unwind, UNWIND);
  INSTKEYWORD("unreachable", TermOpVal, Unreachable, UNREACHABLE);

  INSTKEYWORD("malloc",      MemOpVal, Malloc, MALLOC);
  INSTKEYWORD("alloca",      MemOpVal, Alloca, ALLOCA);
  INSTKEYWORD("free",        MemOpVal, Free, FREE);
  INSTKEYWORD("load",        MemOpVal, Load, LOAD);
  INSTKEYWORD("store",       MemOpVal, Store, STORE);
  INSTKEYWORD("getelementptr", MemOpVal, GetElementPtr, GETELEMENTPTR);

  INSTKEYWORD("extractelement", OtherOpVal, ExtractElement, EXTRACTELEMENT);
  INSTKEYWORD("insertelement", OtherOpVal, InsertElement, INSERTELEMENT);
  INSTKEYWORD("shufflevector", OtherOpVal, ShuffleVector, SHUFFLEVECTOR);
  INSTKEYWORD("getresult", OtherOpVal, ExtractValue, GETRESULT);
  INSTKEYWORD("extractvalue", OtherOpVal, ExtractValue, EXTRACTVALUE);
  INSTKEYWORD("insertvalue", OtherOpVal, InsertValue, INSERTVALUE);
#undef INSTKEYWORD

  // Check for [us]0x[0-9A-Fa-f]+ which are Hexadecimal constant generated by
  // the CFE to avoid forcing it to deal with 64-bit numbers.
  if ((TokStart[0] == 'u' || TokStart[0] == 's') &&
      TokStart[1] == '0' && TokStart[2] == 'x' && isxdigit(TokStart[3])) {
    int len = CurPtr-TokStart-3;
    uint32_t bits = len * 4;
    APInt Tmp(bits, TokStart+3, len, 16);
    uint32_t activeBits = Tmp.getActiveBits();
    if (activeBits > 0 && activeBits < bits)
      Tmp.trunc(activeBits);
    if (Tmp.getBitWidth() > 64) {
      llvmAsmlval.APIntVal = new APInt(Tmp);
      return TokStart[0] == 's' ? ESAPINTVAL : EUAPINTVAL;
    } else if (TokStart[0] == 's') {
      llvmAsmlval.SInt64Val = Tmp.getSExtValue();
      return ESINT64VAL;
    } else {
      llvmAsmlval.UInt64Val = Tmp.getZExtValue();
      return EUINT64VAL;
    }
  }

  // If this is "cc1234", return this as just "cc".
  if (TokStart[0] == 'c' && TokStart[1] == 'c') {
    CurPtr = TokStart+2;
    return CC_TOK;
  }

  // If this starts with "call", return it as CALL.  This is to support old
  // broken .ll files.  FIXME: remove this with LLVM 3.0.
  if (CurPtr-TokStart > 4 && !memcmp(TokStart, "call", 4)) {
    CurPtr = TokStart+4;
    llvmAsmlval.OtherOpVal = Instruction::Call;
    return CALL;
  }

  // Finally, if this isn't known, return just a single character.
  CurPtr = TokStart+1;
  return TokStart[0];
}


/// Lex0x: Handle productions that start with 0x, knowing that it matches and
/// that this is not a label:
///    HexFPConstant     0x[0-9A-Fa-f]+
///    HexFP80Constant   0xK[0-9A-Fa-f]+
///    HexFP128Constant  0xL[0-9A-Fa-f]+
///    HexPPC128Constant 0xM[0-9A-Fa-f]+
int LLLexer::Lex0x() {
  CurPtr = TokStart + 2;

  char Kind;
  if (CurPtr[0] >= 'K' && CurPtr[0] <= 'M') {
    Kind = *CurPtr++;
  } else {
    Kind = 'J';
  }

  if (!isxdigit(CurPtr[0])) {
    // Bad token, return it as just zero.
    CurPtr = TokStart+1;
    return '0';
  }

  while (isxdigit(CurPtr[0]))
    ++CurPtr;

  if (Kind == 'J') {
    // HexFPConstant - Floating point constant represented in IEEE format as a
    // hexadecimal number for when exponential notation is not precise enough.
    // Float and double only.
    llvmAsmlval.FPVal = new APFloat(HexToFP(TokStart+2, CurPtr));
    return FPVAL;
  }

  uint64_t Pair[2];
  HexToIntPair(TokStart+3, CurPtr, Pair);
  switch (Kind) {
  default: assert(0 && "Unknown kind!");
  case 'K':
    // F80HexFPConstant - x87 long double in hexadecimal format (10 bytes)
    llvmAsmlval.FPVal = new APFloat(APInt(80, 2, Pair));
    return FPVAL;
  case 'L':
    // F128HexFPConstant - IEEE 128-bit in hexadecimal format (16 bytes)
    llvmAsmlval.FPVal = new APFloat(APInt(128, 2, Pair), true);
    return FPVAL;
  case 'M':
    // PPC128HexFPConstant - PowerPC 128-bit in hexadecimal format (16 bytes)
    llvmAsmlval.FPVal = new APFloat(APInt(128, 2, Pair));
    return FPVAL;
  }
}

/// LexIdentifier: Handle several related productions:
///    Label             [-a-zA-Z$._0-9]+:
///    NInteger          -[0-9]+
///    FPConstant        [-+]?[0-9]+[.][0-9]*([eE][-+]?[0-9]+)?
///    PInteger          [0-9]+
///    HexFPConstant     0x[0-9A-Fa-f]+
///    HexFP80Constant   0xK[0-9A-Fa-f]+
///    HexFP128Constant  0xL[0-9A-Fa-f]+
///    HexPPC128Constant 0xM[0-9A-Fa-f]+
int LLLexer::LexDigitOrNegative() {
  // If the letter after the negative is a number, this is probably a label.
  if (!isdigit(TokStart[0]) && !isdigit(CurPtr[0])) {
    // Okay, this is not a number after the -, it's probably a label.
    if (const char *End = isLabelTail(CurPtr)) {
      llvmAsmlval.StrVal = new std::string(TokStart, End-1);
      CurPtr = End;
      return LABELSTR;
    }

    return CurPtr[-1];
  }

  // At this point, it is either a label, int or fp constant.

  // Skip digits, we have at least one.
  for (; isdigit(CurPtr[0]); ++CurPtr)
    /*empty*/;

  // Check to see if this really is a label afterall, e.g. "-1:".
  if (isLabelChar(CurPtr[0]) || CurPtr[0] == ':') {
    if (const char *End = isLabelTail(CurPtr)) {
      llvmAsmlval.StrVal = new std::string(TokStart, End-1);
      CurPtr = End;
      return LABELSTR;
    }
  }

  // If the next character is a '.', then it is a fp value, otherwise its
  // integer.
  if (CurPtr[0] != '.') {
    if (TokStart[0] == '0' && TokStart[1] == 'x')
      return Lex0x();
    unsigned Len = CurPtr-TokStart;
    uint32_t numBits = ((Len * 64) / 19) + 2;
    APInt Tmp(numBits, TokStart, Len, 10);
    if (TokStart[0] == '-') {
      uint32_t minBits = Tmp.getMinSignedBits();
      if (minBits > 0 && minBits < numBits)
        Tmp.trunc(minBits);
      if (Tmp.getBitWidth() > 64) {
        llvmAsmlval.APIntVal = new APInt(Tmp);
        return ESAPINTVAL;
      } else {
        llvmAsmlval.SInt64Val = Tmp.getSExtValue();
        return ESINT64VAL;
      }
    } else {
      uint32_t activeBits = Tmp.getActiveBits();
      if (activeBits > 0 && activeBits < numBits)
        Tmp.trunc(activeBits);
      if (Tmp.getBitWidth() > 64) {
        llvmAsmlval.APIntVal = new APInt(Tmp);
        return EUAPINTVAL;
      } else {
        llvmAsmlval.UInt64Val = Tmp.getZExtValue();
        return EUINT64VAL;
      }
    }
  }

  ++CurPtr;

  // Skip over [0-9]*([eE][-+]?[0-9]+)?
  while (isdigit(CurPtr[0])) ++CurPtr;

  if (CurPtr[0] == 'e' || CurPtr[0] == 'E') {
    if (isdigit(CurPtr[1]) ||
        ((CurPtr[1] == '-' || CurPtr[1] == '+') && isdigit(CurPtr[2]))) {
      CurPtr += 2;
      while (isdigit(CurPtr[0])) ++CurPtr;
    }
  }

  llvmAsmlval.FPVal = new APFloat(atof(TokStart));
  return FPVAL;
}

///    FPConstant  [-+]?[0-9]+[.][0-9]*([eE][-+]?[0-9]+)?
int LLLexer::LexPositive() {
  // If the letter after the negative is a number, this is probably not a
  // label.
  if (!isdigit(CurPtr[0]))
    return CurPtr[-1];

  // Skip digits.
  for (++CurPtr; isdigit(CurPtr[0]); ++CurPtr)
    /*empty*/;

  // At this point, we need a '.'.
  if (CurPtr[0] != '.') {
    CurPtr = TokStart+1;
    return TokStart[0];
  }

  ++CurPtr;

  // Skip over [0-9]*([eE][-+]?[0-9]+)?
  while (isdigit(CurPtr[0])) ++CurPtr;

  if (CurPtr[0] == 'e' || CurPtr[0] == 'E') {
    if (isdigit(CurPtr[1]) ||
        ((CurPtr[1] == '-' || CurPtr[1] == '+') && isdigit(CurPtr[2]))) {
      CurPtr += 2;
      while (isdigit(CurPtr[0])) ++CurPtr;
    }
  }

  llvmAsmlval.FPVal = new APFloat(atof(TokStart));
  return FPVAL;
}


//===----------------------------------------------------------------------===//
// Define the interface to this file.
//===----------------------------------------------------------------------===//

static LLLexer *TheLexer;

void InitLLLexer(llvm::MemoryBuffer *MB) {
  assert(TheLexer == 0 && "LL Lexer isn't reentrant yet");
  TheLexer = new LLLexer(MB);
}

int llvmAsmlex() {
  return TheLexer->LexToken();
}
const char *LLLgetTokenStart() { return TheLexer->getTokStart(); }
unsigned LLLgetTokenLength() { return TheLexer->getTokLength(); }
std::string LLLgetFilename() { return TheLexer->getFilename(); }
unsigned LLLgetLineNo() { return TheLexer->getLineNo(); }

void FreeLexer() {
  delete TheLexer;
  TheLexer = 0;
}
