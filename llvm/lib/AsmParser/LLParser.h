//===-- LLParser.h - Parser Class -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the parser class for .ll files.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ASMPARSER_LLPARSER_H
#define LLVM_ASMPARSER_LLPARSER_H

#include "LLLexer.h"
#include "llvm/Type.h"
#include <map>

namespace llvm {
  class Module;
  class OpaqueType;
  class Function;
  class Value;
  class BasicBlock;
  class Instruction;
  class Constant;
  class GlobalValue;
  struct ValID;

  class LLParser {
  public:
    typedef LLLexer::LocTy LocTy;
  private:

    LLLexer Lex;
    Module *M;

    // Type resolution handling data structures.
    std::map<std::string, std::pair<PATypeHolder, LocTy> > ForwardRefTypes;
    std::map<unsigned, std::pair<PATypeHolder, LocTy> > ForwardRefTypeIDs;
    std::vector<PATypeHolder> NumberedTypes;

    struct UpRefRecord {
      /// Loc - This is the location of the upref.
      LocTy Loc;

      /// NestingLevel - The number of nesting levels that need to be popped
      /// before this type is resolved.
      unsigned NestingLevel;

      /// LastContainedTy - This is the type at the current binding level for
      /// the type.  Every time we reduce the nesting level, this gets updated.
      const Type *LastContainedTy;

      /// UpRefTy - This is the actual opaque type that the upreference is
      /// represented with.
      OpaqueType *UpRefTy;

      UpRefRecord(LocTy L, unsigned NL, OpaqueType *URTy)
        : Loc(L), NestingLevel(NL), LastContainedTy((Type*)URTy),
          UpRefTy(URTy) {}
    };
    std::vector<UpRefRecord> UpRefs;

    // Global Value reference information.
    std::map<std::string, std::pair<GlobalValue*, LocTy> > ForwardRefVals;
    std::map<unsigned, std::pair<GlobalValue*, LocTy> > ForwardRefValIDs;
    std::vector<GlobalValue*> NumberedVals;
  public:
    LLParser(MemoryBuffer *F, ParseError &Err, Module *m) : Lex(F, Err), M(m) {}
    bool Run();

  private:

    bool Error(LocTy L, const std::string &Msg) const {
      return Lex.Error(L, Msg);
    }
    bool TokError(const std::string &Msg) const {
      return Error(Lex.getLoc(), Msg);
    }

    /// GetGlobalVal - Get a value with the specified name or ID, creating a
    /// forward reference record if needed.  This can return null if the value
    /// exists but does not have the right type.
    GlobalValue *GetGlobalVal(const std::string &N, const Type *Ty, LocTy Loc);
    GlobalValue *GetGlobalVal(unsigned ID, const Type *Ty, LocTy Loc);

    // Helper Routines.
    bool ParseToken(lltok::Kind T, const char *ErrMsg);
    bool EatIfPresent(lltok::Kind T) {
      if (Lex.getKind() != T) return false;
      Lex.Lex();
      return true;
    }
    bool ParseOptionalToken(lltok::Kind T, bool &Present) {
      if (Lex.getKind() != T) {
        Present = false;
      } else {
        Lex.Lex();
        Present = true;
      }
      return false;
    }
    bool ParseStringConstant(std::string &Result);
    bool ParseUInt32(unsigned &Val);
    bool ParseUInt32(unsigned &Val, LocTy &Loc) {
      Loc = Lex.getLoc();
      return ParseUInt32(Val);
    }
    bool ParseOptionalAddrSpace(unsigned &AddrSpace);
    bool ParseOptionalAttrs(unsigned &Attrs, unsigned AttrKind);
    bool ParseOptionalLinkage(unsigned &Linkage, bool &HasLinkage);
    bool ParseOptionalLinkage(unsigned &Linkage) {
      bool HasLinkage; return ParseOptionalLinkage(Linkage, HasLinkage);
    }
    bool ParseOptionalVisibility(unsigned &Visibility);
    bool ParseOptionalCallingConv(unsigned &CC);
    bool ParseOptionalAlignment(unsigned &Alignment);
    bool ParseOptionalCommaAlignment(unsigned &Alignment);
    bool ParseIndexList(SmallVectorImpl<unsigned> &Indices);

    // Top-Level Entities
    bool ParseTopLevelEntities();
    bool ValidateEndOfModule();
    bool ParseTargetDefinition();
    bool ParseDepLibs();
    bool ParseModuleAsm();
    bool ParseUnnamedType();
    bool ParseNamedType();
    bool ParseDeclare();
    bool ParseDefine();

    bool ParseGlobalType(bool &IsConstant);
    bool ParseNamedGlobal();
    bool ParseGlobal(const std::string &Name, LocTy Loc, unsigned Linkage,
                     bool HasLinkage, unsigned Visibility);
    bool ParseAlias(const std::string &Name, LocTy Loc, unsigned Visibility);

    // Type Parsing.
    bool ParseType(PATypeHolder &Result);
    bool ParseType(PATypeHolder &Result, LocTy &Loc) {
      Loc = Lex.getLoc();
      return ParseType(Result);
    }
    bool ParseTypeRec(PATypeHolder &H);
    bool ParseStructType(PATypeHolder &H, bool Packed);
    bool ParseArrayVectorType(PATypeHolder &H, bool isVector);
    bool ParseFunctionType(PATypeHolder &Result);
    PATypeHolder HandleUpRefs(const Type *Ty);

    // Constants.
    bool ParseValID(ValID &ID);
    bool ConvertGlobalValIDToValue(const Type *Ty, ValID &ID, Constant *&V);
    bool ParseGlobalValue(const Type *Ty, Constant *&V);
    bool ParseGlobalTypeAndValue(Constant *&V);
    bool ParseGlobalValueVector(SmallVectorImpl<Constant*> &Elts);


    // Function Semantic Analysis.
    class PerFunctionState {
      LLParser &P;
      Function &F;
      std::map<std::string, std::pair<Value*, LocTy> > ForwardRefVals;
      std::map<unsigned, std::pair<Value*, LocTy> > ForwardRefValIDs;
      std::vector<Value*> NumberedVals;
    public:
      PerFunctionState(LLParser &p, Function &f);
      ~PerFunctionState();

      Function &getFunction() const { return F; }

      bool VerifyFunctionComplete();

      /// GetVal - Get a value with the specified name or ID, creating a
      /// forward reference record if needed.  This can return null if the value
      /// exists but does not have the right type.
      Value *GetVal(const std::string &Name, const Type *Ty, LocTy Loc);
      Value *GetVal(unsigned ID, const Type *Ty, LocTy Loc);

      /// SetInstName - After an instruction is parsed and inserted into its
      /// basic block, this installs its name.
      bool SetInstName(int NameID, const std::string &NameStr, LocTy NameLoc,
                       Instruction *Inst);

      /// GetBB - Get a basic block with the specified name or ID, creating a
      /// forward reference record if needed.  This can return null if the value
      /// is not a BasicBlock.
      BasicBlock *GetBB(const std::string &Name, LocTy Loc);
      BasicBlock *GetBB(unsigned ID, LocTy Loc);

      /// DefineBB - Define the specified basic block, which is either named or
      /// unnamed.  If there is an error, this returns null otherwise it returns
      /// the block being defined.
      BasicBlock *DefineBB(const std::string &Name, LocTy Loc);
    };

    bool ConvertValIDToValue(const Type *Ty, ValID &ID, Value *&V,
                             PerFunctionState &PFS);

    bool ParseValue(const Type *Ty, Value *&V, PerFunctionState &PFS);
    bool ParseValue(const Type *Ty, Value *&V, LocTy &Loc,
                    PerFunctionState &PFS) {
      Loc = Lex.getLoc();
      return ParseValue(Ty, V, PFS);
    }

    bool ParseTypeAndValue(Value *&V, PerFunctionState &PFS);
    bool ParseTypeAndValue(Value *&V, LocTy &Loc, PerFunctionState &PFS) {
      Loc = Lex.getLoc();
      return ParseTypeAndValue(V, PFS);
    }

    struct ParamInfo {
      LocTy Loc;
      Value *V;
      unsigned Attrs;
      ParamInfo(LocTy loc, Value *v, unsigned attrs)
        : Loc(loc), V(v), Attrs(attrs) {}
    };
    bool ParseParameterList(SmallVectorImpl<ParamInfo> &ArgList,
                            PerFunctionState &PFS);

    // Function Parsing.
    struct ArgInfo {
      LocTy Loc;
      PATypeHolder Type;
      unsigned Attrs;
      std::string Name;
      ArgInfo(LocTy L, PATypeHolder Ty, unsigned Attr, const std::string &N)
        : Loc(L), Type(Ty), Attrs(Attr), Name(N) {}
    };
    bool ParseArgumentList(std::vector<ArgInfo> &ArgList,
                           bool &isVarArg, bool inType);
    bool ParseFunctionHeader(Function *&Fn, bool isDefine);
    bool ParseFunctionBody(Function &Fn);
    bool ParseBasicBlock(PerFunctionState &PFS);

    // Instruction Parsing.
    bool ParseInstruction(Instruction *&Inst, BasicBlock *BB,
                          PerFunctionState &PFS);
    bool ParseCmpPredicate(unsigned &Pred, unsigned Opc);

    bool ParseRet(Instruction *&Inst, BasicBlock *BB, PerFunctionState &PFS);
    bool ParseBr(Instruction *&Inst, PerFunctionState &PFS);
    bool ParseSwitch(Instruction *&Inst, PerFunctionState &PFS);
    bool ParseInvoke(Instruction *&Inst, PerFunctionState &PFS);

    bool ParseArithmetic(Instruction *&I, PerFunctionState &PFS, unsigned Opc,
                         unsigned OperandType);
    bool ParseLogical(Instruction *&I, PerFunctionState &PFS, unsigned Opc);
    bool ParseCompare(Instruction *&I, PerFunctionState &PFS, unsigned Opc);
    bool ParseCast(Instruction *&I, PerFunctionState &PFS, unsigned Opc);
    bool ParseSelect(Instruction *&I, PerFunctionState &PFS);
    bool ParseVA_Arg(Instruction *&I, PerFunctionState &PFS);
    bool ParseExtractElement(Instruction *&I, PerFunctionState &PFS);
    bool ParseInsertElement(Instruction *&I, PerFunctionState &PFS);
    bool ParseShuffleVector(Instruction *&I, PerFunctionState &PFS);
    bool ParsePHI(Instruction *&I, PerFunctionState &PFS);
    bool ParseCall(Instruction *&I, PerFunctionState &PFS, bool isTail);
    bool ParseAlloc(Instruction *&I, PerFunctionState &PFS, unsigned Opc);
    bool ParseFree(Instruction *&I, PerFunctionState &PFS);
    bool ParseLoad(Instruction *&I, PerFunctionState &PFS, bool isVolatile);
    bool ParseStore(Instruction *&I, PerFunctionState &PFS, bool isVolatile);
    bool ParseGetResult(Instruction *&I, PerFunctionState &PFS);
    bool ParseGetElementPtr(Instruction *&I, PerFunctionState &PFS);
    bool ParseExtractValue(Instruction *&I, PerFunctionState &PFS);
    bool ParseInsertValue(Instruction *&I, PerFunctionState &PFS);
  };
} // End llvm namespace

#endif
