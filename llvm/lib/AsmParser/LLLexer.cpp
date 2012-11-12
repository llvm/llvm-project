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
#include "llvm/DerivedTypes.h"
#include "llvm/Instruction.h"
#include "llvm/LLVMContext.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Assembly/Parser.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
using namespace llvm;

bool LLLexer::Error(LocTy ErrorLoc, const Twine &Msg) const {
  ErrorInfo = SM.GetMessage(ErrorLoc, SourceMgr::DK_Error, Msg);
  return true;
}

//===----------------------------------------------------------------------===//
// Helper functions.
//===----------------------------------------------------------------------===//

// atoull - Convert an ascii string of decimal digits into the unsigned long
// long representation... this does not have to do input error checking,
// because we know that the input will be matched by a suitable regex...
//
uint64_t LLLexer::atoull(const char *Buffer, const char *End) {
  uint64_t Result = 0;
  for (; Buffer != End; Buffer++) {
    uint64_t OldRes = Result;
    Result *= 10;
    Result += *Buffer-'0';
    if (Result < OldRes) {  // Uh, oh, overflow detected!!!
      Error("constant bigger than 64 bits detected!");
      return 0;
    }
  }
  return Result;
}

static char parseHexChar(char C) {
  if (C >= '0' && C <= '9')
    return C-'0';
  if (C >= 'A' && C <= 'F')
    return C-'A'+10;
  if (C >= 'a' && C <= 'f')
    return C-'a'+10;
  return 0;
}

uint64_t LLLexer::HexIntToVal(const char *Buffer, const char *End) {
  uint64_t Result = 0;
  for (; Buffer != End; ++Buffer) {
    uint64_t OldRes = Result;
    Result *= 16;
    Result += parseHexChar(*Buffer);

    if (Result < OldRes) {   // Uh, oh, overflow detected!!!
      Error("constant bigger than 64 bits detected!");
      return 0;
    }
  }
  return Result;
}

void LLLexer::HexToIntPair(const char *Buffer, const char *End,
                           uint64_t Pair[2]) {
  Pair[0] = 0;
  for (int i=0; i<16; i++, Buffer++) {
    assert(Buffer != End);
    Pair[0] *= 16;
    Pair[0] += parseHexChar(*Buffer);
  }
  Pair[1] = 0;
  for (int i=0; i<16 && Buffer != End; i++, Buffer++) {
    Pair[1] *= 16;
    Pair[1] += parseHexChar(*Buffer);
  }
  if (Buffer != End)
    Error("constant bigger than 128 bits detected!");
}

/// FP80HexToIntPair - translate an 80 bit FP80 number (20 hexits) into
/// { low64, high16 } as usual for an APInt.
void LLLexer::FP80HexToIntPair(const char *Buffer, const char *End,
                           uint64_t Pair[2]) {
  Pair[1] = 0;
  for (int i=0; i<4 && Buffer != End; i++, Buffer++) {
    assert(Buffer != End);
    Pair[1] *= 16;
    Pair[1] += parseHexChar(*Buffer);
  }
  Pair[0] = 0;
  for (int i=0; i<16; i++, Buffer++) {
    Pair[0] *= 16;
    Pair[0] += parseHexChar(*Buffer);
  }
  if (Buffer != End)
    Error("constant bigger than 128 bits detected!");
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
        *BOut = parseHexChar(BIn[1]) * 16 + parseHexChar(BIn[2]);
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

LLLexer::LLLexer(MemoryBuffer *StartBuf, SourceMgr &sm, SMDiagnostic &Err,
                 LLVMContext &C)
  : CurBuf(StartBuf), ErrorInfo(Err), SM(sm), Context(C), APFloatVal(0.0) {
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
  }
}


lltok::Kind LLLexer::LexToken() {
  TokStart = CurPtr;

  int CurChar = getNextChar();
  switch (CurChar) {
  default:
    // Handle letters: [a-zA-Z_]
    if (isalpha(CurChar) || CurChar == '_')
      return LexIdentifier();

    return lltok::Error;
  case EOF: return lltok::Eof;
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
      StrVal.assign(TokStart, CurPtr-1);
      return lltok::LabelStr;
    }
    if (CurPtr[0] == '.' && CurPtr[1] == '.') {
      CurPtr += 2;
      return lltok::dotdotdot;
    }
    return lltok::Error;
  case '$':
    if (const char *Ptr = isLabelTail(CurPtr)) {
      CurPtr = Ptr;
      StrVal.assign(TokStart, CurPtr-1);
      return lltok::LabelStr;
    }
    return lltok::Error;
  case ';':
    SkipLineComment();
    return LexToken();
  case '!': return LexExclaim();
  case '0': case '1': case '2': case '3': case '4':
  case '5': case '6': case '7': case '8': case '9':
  case '-':
    return LexDigitOrNegative();
  case '=': return lltok::equal;
  case '[': return lltok::lsquare;
  case ']': return lltok::rsquare;
  case '{': return lltok::lbrace;
  case '}': return lltok::rbrace;
  case '<': return lltok::less;
  case '>': return lltok::greater;
  case '(': return lltok::lparen;
  case ')': return lltok::rparen;
  case ',': return lltok::comma;
  case '*': return lltok::star;
  case '\\': return lltok::backslash;
  }
}

void LLLexer::SkipLineComment() {
  while (1) {
    if (CurPtr[0] == '\n' || CurPtr[0] == '\r' || getNextChar() == EOF)
      return;
  }
}

/// LexAt - Lex all tokens that start with an @ character:
///   GlobalVar   @\"[^\"]*\"
///   GlobalVar   @[-a-zA-Z$._][-a-zA-Z$._0-9]*
///   GlobalVarID @[0-9]+
lltok::Kind LLLexer::LexAt() {
  // Handle AtStringConstant: @\"[^\"]*\"
  if (CurPtr[0] == '"') {
    ++CurPtr;

    while (1) {
      int CurChar = getNextChar();

      if (CurChar == EOF) {
        Error("end of file in global variable name");
        return lltok::Error;
      }
      if (CurChar == '"') {
        StrVal.assign(TokStart+2, CurPtr-1);
        UnEscapeLexed(StrVal);
        return lltok::GlobalVar;
      }
    }
  }

  // Handle GlobalVarName: @[-a-zA-Z$._][-a-zA-Z$._0-9]*
  if (ReadVarName())
    return lltok::GlobalVar;

  // Handle GlobalVarID: @[0-9]+
  if (isdigit(CurPtr[0])) {
    for (++CurPtr; isdigit(CurPtr[0]); ++CurPtr)
      /*empty*/;

    uint64_t Val = atoull(TokStart+1, CurPtr);
    if ((unsigned)Val != Val)
      Error("invalid value number (too large)!");
    UIntVal = unsigned(Val);
    return lltok::GlobalID;
  }

  return lltok::Error;
}

/// ReadString - Read a string until the closing quote.
lltok::Kind LLLexer::ReadString(lltok::Kind kind) {
  const char *Start = CurPtr;
  while (1) {
    int CurChar = getNextChar();

    if (CurChar == EOF) {
      Error("end of file in string constant");
      return lltok::Error;
    }
    if (CurChar == '"') {
      StrVal.assign(Start, CurPtr-1);
      UnEscapeLexed(StrVal);
      return kind;
    }
  }
}

/// ReadVarName - Read the rest of a token containing a variable name.
bool LLLexer::ReadVarName() {
  const char *NameStart = CurPtr;
  if (isalpha(CurPtr[0]) || CurPtr[0] == '-' || CurPtr[0] == '$' ||
      CurPtr[0] == '.' || CurPtr[0] == '_') {
    ++CurPtr;
    while (isalnum(CurPtr[0]) || CurPtr[0] == '-' || CurPtr[0] == '$' ||
           CurPtr[0] == '.' || CurPtr[0] == '_')
      ++CurPtr;

    StrVal.assign(NameStart, CurPtr);
    return true;
  }
  return false;
}

/// LexPercent - Lex all tokens that start with a % character:
///   LocalVar   ::= %\"[^\"]*\"
///   LocalVar   ::= %[-a-zA-Z$._][-a-zA-Z$._0-9]*
///   LocalVarID ::= %[0-9]+
lltok::Kind LLLexer::LexPercent() {
  // Handle LocalVarName: %\"[^\"]*\"
  if (CurPtr[0] == '"') {
    ++CurPtr;
    return ReadString(lltok::LocalVar);
  }

  // Handle LocalVarName: %[-a-zA-Z$._][-a-zA-Z$._0-9]*
  if (ReadVarName())
    return lltok::LocalVar;

  // Handle LocalVarID: %[0-9]+
  if (isdigit(CurPtr[0])) {
    for (++CurPtr; isdigit(CurPtr[0]); ++CurPtr)
      /*empty*/;

    uint64_t Val = atoull(TokStart+1, CurPtr);
    if ((unsigned)Val != Val)
      Error("invalid value number (too large)!");
    UIntVal = unsigned(Val);
    return lltok::LocalVarID;
  }

  return lltok::Error;
}

/// LexQuote - Lex all tokens that start with a " character:
///   QuoteLabel        "[^"]+":
///   StringConstant    "[^"]*"
lltok::Kind LLLexer::LexQuote() {
  lltok::Kind kind = ReadString(lltok::StringConstant);
  if (kind == lltok::Error || kind == lltok::Eof)
    return kind;

  if (CurPtr[0] == ':') {
    ++CurPtr;
    kind = lltok::LabelStr;
  }

  return kind;
}

/// LexExclaim:
///    !foo
///    !
lltok::Kind LLLexer::LexExclaim() {
  // Lex a metadata name as a MetadataVar.
  if (isalpha(CurPtr[0]) || CurPtr[0] == '-' || CurPtr[0] == '$' ||
      CurPtr[0] == '.' || CurPtr[0] == '_' || CurPtr[0] == '\\') {
    ++CurPtr;
    while (isalnum(CurPtr[0]) || CurPtr[0] == '-' || CurPtr[0] == '$' ||
           CurPtr[0] == '.' || CurPtr[0] == '_' || CurPtr[0] == '\\')
      ++CurPtr;

    StrVal.assign(TokStart+1, CurPtr);   // Skip !
    UnEscapeLexed(StrVal);
    return lltok::MetadataVar;
  }
  return lltok::exclaim;
}
  
/// LexIdentifier: Handle several related productions:
///    Label           [-a-zA-Z$._0-9]+:
///    IntegerType     i[0-9]+
///    Keyword         sdiv, float, ...
///    HexIntConstant  [us]0x[0-9A-Fa-f]+
lltok::Kind LLLexer::LexIdentifier() {
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
    StrVal.assign(StartChar-1, CurPtr++);
    return lltok::LabelStr;
  }

  // Otherwise, this wasn't a label.  If this was valid as an integer type,
  // return it.
  if (IntEnd == 0) IntEnd = CurPtr;
  if (IntEnd != StartChar) {
    CurPtr = IntEnd;
    uint64_t NumBits = atoull(StartChar, CurPtr);
    if (NumBits < IntegerType::MIN_INT_BITS ||
        NumBits > IntegerType::MAX_INT_BITS) {
      Error("bitwidth for integer type out of range!");
      return lltok::Error;
    }
    TyVal = IntegerType::get(Context, NumBits);
    return lltok::Type;
  }

  // Otherwise, this was a letter sequence.  See which keyword this is.
  if (KeywordEnd == 0) KeywordEnd = CurPtr;
  CurPtr = KeywordEnd;
  --StartChar;
  unsigned Len = CurPtr-StartChar;
#define KEYWORD(STR) \
  if (Len == strlen(#STR) && !memcmp(StartChar, #STR, strlen(#STR))) \
    return lltok::kw_##STR;

  KEYWORD(true);    KEYWORD(false);
  KEYWORD(declare); KEYWORD(define);
  KEYWORD(global);  KEYWORD(constant);

  KEYWORD(private);
  KEYWORD(linker_private);
  KEYWORD(linker_private_weak);
  KEYWORD(linker_private_weak_def_auto); // FIXME: For backwards compatibility.
  KEYWORD(internal);
  KEYWORD(available_externally);
  KEYWORD(linkonce);
  KEYWORD(linkonce_odr);
  KEYWORD(linkonce_odr_auto_hide);
  KEYWORD(weak);
  KEYWORD(weak_odr);
  KEYWORD(appending);
  KEYWORD(dllimport);
  KEYWORD(dllexport);
  KEYWORD(common);
  KEYWORD(default);
  KEYWORD(hidden);
  KEYWORD(protected);
  KEYWORD(unnamed_addr);
  KEYWORD(extern_weak);
  KEYWORD(external);
  KEYWORD(thread_local);
  KEYWORD(localdynamic);
  KEYWORD(initialexec);
  KEYWORD(localexec);
  KEYWORD(zeroinitializer);
  KEYWORD(undef);
  KEYWORD(null);
  KEYWORD(to);
  KEYWORD(tail);
  KEYWORD(target);
  KEYWORD(triple);
  KEYWORD(unwind);
  KEYWORD(deplibs);
  KEYWORD(datalayout);
  KEYWORD(volatile);
  KEYWORD(atomic);
  KEYWORD(unordered);
  KEYWORD(monotonic);
  KEYWORD(acquire);
  KEYWORD(release);
  KEYWORD(acq_rel);
  KEYWORD(seq_cst);
  KEYWORD(singlethread);

  KEYWORD(nuw);
  KEYWORD(nsw);
  KEYWORD(exact);
  KEYWORD(inbounds);
  KEYWORD(align);
  KEYWORD(addrspace);
  KEYWORD(section);
  KEYWORD(alias);
  KEYWORD(module);
  KEYWORD(asm);
  KEYWORD(sideeffect);
  KEYWORD(alignstack);
  KEYWORD(inteldialect);
  KEYWORD(gc);

  KEYWORD(ccc);
  KEYWORD(fastcc);
  KEYWORD(coldcc);
  KEYWORD(x86_stdcallcc);
  KEYWORD(x86_fastcallcc);
  KEYWORD(x86_thiscallcc);
  KEYWORD(arm_apcscc);
  KEYWORD(arm_aapcscc);
  KEYWORD(arm_aapcs_vfpcc);
  KEYWORD(msp430_intrcc);
  KEYWORD(ptx_kernel);
  KEYWORD(ptx_device);
  KEYWORD(spir_kernel);
  KEYWORD(spir_func);
  KEYWORD(intel_ocl_bicc);

  KEYWORD(cc);
  KEYWORD(c);

  KEYWORD(signext);
  KEYWORD(zeroext);
  KEYWORD(inreg);
  KEYWORD(sret);
  KEYWORD(nounwind);
  KEYWORD(noreturn);
  KEYWORD(noalias);
  KEYWORD(nocapture);
  KEYWORD(byval);
  KEYWORD(nest);
  KEYWORD(readnone);
  KEYWORD(readonly);
  KEYWORD(uwtable);
  KEYWORD(returns_twice);

  KEYWORD(inlinehint);
  KEYWORD(noinline);
  KEYWORD(alwaysinline);
  KEYWORD(optsize);
  KEYWORD(ssp);
  KEYWORD(sspreq);
  KEYWORD(noredzone);
  KEYWORD(noimplicitfloat);
  KEYWORD(naked);
  KEYWORD(nonlazybind);
  KEYWORD(address_safety);
  KEYWORD(minsize);

  KEYWORD(type);
  KEYWORD(opaque);

  KEYWORD(eq); KEYWORD(ne); KEYWORD(slt); KEYWORD(sgt); KEYWORD(sle);
  KEYWORD(sge); KEYWORD(ult); KEYWORD(ugt); KEYWORD(ule); KEYWORD(uge);
  KEYWORD(oeq); KEYWORD(one); KEYWORD(olt); KEYWORD(ogt); KEYWORD(ole);
  KEYWORD(oge); KEYWORD(ord); KEYWORD(uno); KEYWORD(ueq); KEYWORD(une);

  KEYWORD(xchg); KEYWORD(nand); KEYWORD(max); KEYWORD(min); KEYWORD(umax);
  KEYWORD(umin);

  KEYWORD(x);
  KEYWORD(blockaddress);

  KEYWORD(personality);
  KEYWORD(cleanup);
  KEYWORD(catch);
  KEYWORD(filter);
#undef KEYWORD

  // Keywords for types.
#define TYPEKEYWORD(STR, LLVMTY) \
  if (Len == strlen(STR) && !memcmp(StartChar, STR, strlen(STR))) { \
    TyVal = LLVMTY; return lltok::Type; }
  TYPEKEYWORD("void",      Type::getVoidTy(Context));
  TYPEKEYWORD("half",      Type::getHalfTy(Context));
  TYPEKEYWORD("float",     Type::getFloatTy(Context));
  TYPEKEYWORD("double",    Type::getDoubleTy(Context));
  TYPEKEYWORD("x86_fp80",  Type::getX86_FP80Ty(Context));
  TYPEKEYWORD("fp128",     Type::getFP128Ty(Context));
  TYPEKEYWORD("ppc_fp128", Type::getPPC_FP128Ty(Context));
  TYPEKEYWORD("label",     Type::getLabelTy(Context));
  TYPEKEYWORD("metadata",  Type::getMetadataTy(Context));
  TYPEKEYWORD("x86_mmx",   Type::getX86_MMXTy(Context));
#undef TYPEKEYWORD

  // Keywords for instructions.
#define INSTKEYWORD(STR, Enum) \
  if (Len == strlen(#STR) && !memcmp(StartChar, #STR, strlen(#STR))) { \
    UIntVal = Instruction::Enum; return lltok::kw_##STR; }

  INSTKEYWORD(add,   Add);  INSTKEYWORD(fadd,   FAdd);
  INSTKEYWORD(sub,   Sub);  INSTKEYWORD(fsub,   FSub);
  INSTKEYWORD(mul,   Mul);  INSTKEYWORD(fmul,   FMul);
  INSTKEYWORD(udiv,  UDiv); INSTKEYWORD(sdiv,  SDiv); INSTKEYWORD(fdiv,  FDiv);
  INSTKEYWORD(urem,  URem); INSTKEYWORD(srem,  SRem); INSTKEYWORD(frem,  FRem);
  INSTKEYWORD(shl,   Shl);  INSTKEYWORD(lshr,  LShr); INSTKEYWORD(ashr,  AShr);
  INSTKEYWORD(and,   And);  INSTKEYWORD(or,    Or);   INSTKEYWORD(xor,   Xor);
  INSTKEYWORD(icmp,  ICmp); INSTKEYWORD(fcmp,  FCmp);

  INSTKEYWORD(phi,         PHI);
  INSTKEYWORD(call,        Call);
  INSTKEYWORD(trunc,       Trunc);
  INSTKEYWORD(zext,        ZExt);
  INSTKEYWORD(sext,        SExt);
  INSTKEYWORD(fptrunc,     FPTrunc);
  INSTKEYWORD(fpext,       FPExt);
  INSTKEYWORD(uitofp,      UIToFP);
  INSTKEYWORD(sitofp,      SIToFP);
  INSTKEYWORD(fptoui,      FPToUI);
  INSTKEYWORD(fptosi,      FPToSI);
  INSTKEYWORD(inttoptr,    IntToPtr);
  INSTKEYWORD(ptrtoint,    PtrToInt);
  INSTKEYWORD(bitcast,     BitCast);
  INSTKEYWORD(select,      Select);
  INSTKEYWORD(va_arg,      VAArg);
  INSTKEYWORD(ret,         Ret);
  INSTKEYWORD(br,          Br);
  INSTKEYWORD(switch,      Switch);
  INSTKEYWORD(indirectbr,  IndirectBr);
  INSTKEYWORD(invoke,      Invoke);
  INSTKEYWORD(resume,      Resume);
  INSTKEYWORD(unreachable, Unreachable);

  INSTKEYWORD(alloca,      Alloca);
  INSTKEYWORD(load,        Load);
  INSTKEYWORD(store,       Store);
  INSTKEYWORD(cmpxchg,     AtomicCmpXchg);
  INSTKEYWORD(atomicrmw,   AtomicRMW);
  INSTKEYWORD(fence,       Fence);
  INSTKEYWORD(getelementptr, GetElementPtr);

  INSTKEYWORD(extractelement, ExtractElement);
  INSTKEYWORD(insertelement,  InsertElement);
  INSTKEYWORD(shufflevector,  ShuffleVector);
  INSTKEYWORD(extractvalue,   ExtractValue);
  INSTKEYWORD(insertvalue,    InsertValue);
  INSTKEYWORD(landingpad,     LandingPad);
#undef INSTKEYWORD

  // Check for [us]0x[0-9A-Fa-f]+ which are Hexadecimal constant generated by
  // the CFE to avoid forcing it to deal with 64-bit numbers.
  if ((TokStart[0] == 'u' || TokStart[0] == 's') &&
      TokStart[1] == '0' && TokStart[2] == 'x' && isxdigit(TokStart[3])) {
    int len = CurPtr-TokStart-3;
    uint32_t bits = len * 4;
    APInt Tmp(bits, StringRef(TokStart+3, len), 16);
    uint32_t activeBits = Tmp.getActiveBits();
    if (activeBits > 0 && activeBits < bits)
      Tmp = Tmp.trunc(activeBits);
    APSIntVal = APSInt(Tmp, TokStart[0] == 'u');
    return lltok::APSInt;
  }

  // If this is "cc1234", return this as just "cc".
  if (TokStart[0] == 'c' && TokStart[1] == 'c') {
    CurPtr = TokStart+2;
    return lltok::kw_cc;
  }

  // Finally, if this isn't known, return an error.
  CurPtr = TokStart+1;
  return lltok::Error;
}


/// Lex0x: Handle productions that start with 0x, knowing that it matches and
/// that this is not a label:
///    HexFPConstant     0x[0-9A-Fa-f]+
///    HexFP80Constant   0xK[0-9A-Fa-f]+
///    HexFP128Constant  0xL[0-9A-Fa-f]+
///    HexPPC128Constant 0xM[0-9A-Fa-f]+
///    HexHalfConstant   0xH[0-9A-Fa-f]+
lltok::Kind LLLexer::Lex0x() {
  CurPtr = TokStart + 2;

  char Kind;
  if ((CurPtr[0] >= 'K' && CurPtr[0] <= 'M') || CurPtr[0] == 'H') {
    Kind = *CurPtr++;
  } else {
    Kind = 'J';
  }

  if (!isxdigit(CurPtr[0])) {
    // Bad token, return it as an error.
    CurPtr = TokStart+1;
    return lltok::Error;
  }

  while (isxdigit(CurPtr[0]))
    ++CurPtr;

  if (Kind == 'J') {
    // HexFPConstant - Floating point constant represented in IEEE format as a
    // hexadecimal number for when exponential notation is not precise enough.
    // Half, Float, and double only.
    APFloatVal = APFloat(BitsToDouble(HexIntToVal(TokStart+2, CurPtr)));
    return lltok::APFloat;
  }

  uint64_t Pair[2];
  switch (Kind) {
  default: llvm_unreachable("Unknown kind!");
  case 'K':
    // F80HexFPConstant - x87 long double in hexadecimal format (10 bytes)
    FP80HexToIntPair(TokStart+3, CurPtr, Pair);
    APFloatVal = APFloat(APInt(80, Pair));
    return lltok::APFloat;
  case 'L':
    // F128HexFPConstant - IEEE 128-bit in hexadecimal format (16 bytes)
    HexToIntPair(TokStart+3, CurPtr, Pair);
    APFloatVal = APFloat(APInt(128, Pair), true);
    return lltok::APFloat;
  case 'M':
    // PPC128HexFPConstant - PowerPC 128-bit in hexadecimal format (16 bytes)
    HexToIntPair(TokStart+3, CurPtr, Pair);
    APFloatVal = APFloat(APInt(128, Pair));
    return lltok::APFloat;
  case 'H':
    APFloatVal = APFloat(APInt(16,HexIntToVal(TokStart+3, CurPtr)));
    return lltok::APFloat;
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
lltok::Kind LLLexer::LexDigitOrNegative() {
  // If the letter after the negative is a number, this is probably a label.
  if (!isdigit(TokStart[0]) && !isdigit(CurPtr[0])) {
    // Okay, this is not a number after the -, it's probably a label.
    if (const char *End = isLabelTail(CurPtr)) {
      StrVal.assign(TokStart, End-1);
      CurPtr = End;
      return lltok::LabelStr;
    }

    return lltok::Error;
  }

  // At this point, it is either a label, int or fp constant.

  // Skip digits, we have at least one.
  for (; isdigit(CurPtr[0]); ++CurPtr)
    /*empty*/;

  // Check to see if this really is a label afterall, e.g. "-1:".
  if (isLabelChar(CurPtr[0]) || CurPtr[0] == ':') {
    if (const char *End = isLabelTail(CurPtr)) {
      StrVal.assign(TokStart, End-1);
      CurPtr = End;
      return lltok::LabelStr;
    }
  }

  // If the next character is a '.', then it is a fp value, otherwise its
  // integer.
  if (CurPtr[0] != '.') {
    if (TokStart[0] == '0' && TokStart[1] == 'x')
      return Lex0x();
    unsigned Len = CurPtr-TokStart;
    uint32_t numBits = ((Len * 64) / 19) + 2;
    APInt Tmp(numBits, StringRef(TokStart, Len), 10);
    if (TokStart[0] == '-') {
      uint32_t minBits = Tmp.getMinSignedBits();
      if (minBits > 0 && minBits < numBits)
        Tmp = Tmp.trunc(minBits);
      APSIntVal = APSInt(Tmp, false);
    } else {
      uint32_t activeBits = Tmp.getActiveBits();
      if (activeBits > 0 && activeBits < numBits)
        Tmp = Tmp.trunc(activeBits);
      APSIntVal = APSInt(Tmp, true);
    }
    return lltok::APSInt;
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

  APFloatVal = APFloat(std::atof(TokStart));
  return lltok::APFloat;
}

///    FPConstant  [-+]?[0-9]+[.][0-9]*([eE][-+]?[0-9]+)?
lltok::Kind LLLexer::LexPositive() {
  // If the letter after the negative is a number, this is probably not a
  // label.
  if (!isdigit(CurPtr[0]))
    return lltok::Error;

  // Skip digits.
  for (++CurPtr; isdigit(CurPtr[0]); ++CurPtr)
    /*empty*/;

  // At this point, we need a '.'.
  if (CurPtr[0] != '.') {
    CurPtr = TokStart+1;
    return lltok::Error;
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

  APFloatVal = APFloat(std::atof(TokStart));
  return lltok::APFloat;
}
