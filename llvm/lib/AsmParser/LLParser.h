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
#include "llvm/Attributes.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/Type.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/ValueHandle.h"
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
  class MDString;
  class MDNode;
  class StructType;

  /// ValID - Represents a reference of a definition of some sort with no type.
  /// There are several cases where we have to parse the value but where the
  /// type can depend on later context.  This may either be a numeric reference
  /// or a symbolic (%var) reference.  This is just a discriminated union.
  struct ValID {
    enum {
      t_LocalID, t_GlobalID,      // ID in UIntVal.
      t_LocalName, t_GlobalName,  // Name in StrVal.
      t_APSInt, t_APFloat,        // Value in APSIntVal/APFloatVal.
      t_Null, t_Undef, t_Zero,    // No value.
      t_EmptyArray,               // No value:  []
      t_Constant,                 // Value in ConstantVal.
      t_InlineAsm,                // Value in StrVal/StrVal2/UIntVal.
      t_MDNode,                   // Value in MDNodeVal.
      t_MDString,                 // Value in MDStringVal.
      t_ConstantStruct,           // Value in ConstantStructElts.
      t_PackedConstantStruct      // Value in ConstantStructElts.
    } Kind;
    
    LLLexer::LocTy Loc;
    unsigned UIntVal;
    std::string StrVal, StrVal2;
    APSInt APSIntVal;
    APFloat APFloatVal;
    Constant *ConstantVal;
    MDNode *MDNodeVal;
    MDString *MDStringVal;
    Constant **ConstantStructElts;
    
    ValID() : Kind(t_LocalID), APFloatVal(0.0) {}
    ~ValID() {
      if (Kind == t_ConstantStruct || Kind == t_PackedConstantStruct)
        delete [] ConstantStructElts;
    }
    
    bool operator<(const ValID &RHS) const {
      if (Kind == t_LocalID || Kind == t_GlobalID)
        return UIntVal < RHS.UIntVal;
      assert((Kind == t_LocalName || Kind == t_GlobalName ||
              Kind == t_ConstantStruct || Kind == t_PackedConstantStruct) && 
             "Ordering not defined for this ValID kind yet");
      return StrVal < RHS.StrVal;
    }
  };
  
  class LLParser {
  public:
    typedef LLLexer::LocTy LocTy;
  private:
    LLVMContext &Context;
    LLLexer Lex;
    Module *M;
    
    // Instruction metadata resolution.  Each instruction can have a list of
    // MDRef info associated with them.
    //
    // The simpler approach of just creating temporary MDNodes and then calling
    // RAUW on them when the definition is processed doesn't work because some
    // instruction metadata kinds, such as dbg, get stored in the IR in an
    // "optimized" format which doesn't participate in the normal value use
    // lists. This means that RAUW doesn't work, even on temporary MDNodes
    // which otherwise support RAUW. Instead, we defer resolving MDNode
    // references until the definitions have been processed.
    struct MDRef {
      SMLoc Loc;
      unsigned MDKind, MDSlot;
    };
    DenseMap<Instruction*, std::vector<MDRef> > ForwardRefInstMetadata;

    // Type resolution handling data structures.  The location is set when we
    // have processed a use of the type but not a definition yet.
    StringMap<std::pair<Type*, LocTy> > NamedTypes;
    std::vector<std::pair<Type*, LocTy> > NumberedTypes;
    
    std::vector<TrackingVH<MDNode> > NumberedMetadata;
    std::map<unsigned, std::pair<TrackingVH<MDNode>, LocTy> > ForwardRefMDNodes;

    // Global Value reference information.
    std::map<std::string, std::pair<GlobalValue*, LocTy> > ForwardRefVals;
    std::map<unsigned, std::pair<GlobalValue*, LocTy> > ForwardRefValIDs;
    std::vector<GlobalValue*> NumberedVals;
    
    // References to blockaddress.  The key is the function ValID, the value is
    // a list of references to blocks in that function.
    std::map<ValID, std::vector<std::pair<ValID, GlobalValue*> > >
      ForwardRefBlockAddresses;
    
  public:
    LLParser(MemoryBuffer *F, SourceMgr &SM, SMDiagnostic &Err, Module *m) : 
      Context(m->getContext()), Lex(F, SM, Err, m->getContext()),
      M(m) {}
    bool Run();

    LLVMContext &getContext() { return Context; }

  private:

    bool Error(LocTy L, const Twine &Msg) const {
      return Lex.Error(L, Msg);
    }
    bool TokError(const Twine &Msg) const {
      return Error(Lex.getLoc(), Msg);
    }

    /// GetGlobalVal - Get a value with the specified name or ID, creating a
    /// forward reference record if needed.  This can return null if the value
    /// exists but does not have the right type.
    GlobalValue *GetGlobalVal(const std::string &N, Type *Ty, LocTy Loc);
    GlobalValue *GetGlobalVal(unsigned ID, Type *Ty, LocTy Loc);

    // Helper Routines.
    bool ParseToken(lltok::Kind T, const char *ErrMsg);
    bool EatIfPresent(lltok::Kind T) {
      if (Lex.getKind() != T) return false;
      Lex.Lex();
      return true;
    }
    bool ParseOptionalToken(lltok::Kind T, bool &Present, LocTy *Loc = 0) {
      if (Lex.getKind() != T) {
        Present = false;
      } else {
        if (Loc)
          *Loc = Lex.getLoc();
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

    bool ParseTLSModel(GlobalVariable::ThreadLocalMode &TLM);
    bool ParseOptionalThreadLocal(GlobalVariable::ThreadLocalMode &TLM);
    bool ParseOptionalAddrSpace(unsigned &AddrSpace);
    bool ParseOptionalAttrs(AttrBuilder &Attrs, unsigned AttrKind);
    bool ParseOptionalLinkage(unsigned &Linkage, bool &HasLinkage);
    bool ParseOptionalLinkage(unsigned &Linkage) {
      bool HasLinkage; return ParseOptionalLinkage(Linkage, HasLinkage);
    }
    bool ParseOptionalVisibility(unsigned &Visibility);
    bool ParseOptionalCallingConv(CallingConv::ID &CC);
    bool ParseOptionalAlignment(unsigned &Alignment);
    bool ParseScopeAndOrdering(bool isAtomic, SynchronizationScope &Scope,
                               AtomicOrdering &Ordering);
    bool ParseOptionalStackAlignment(unsigned &Alignment);
    bool ParseOptionalCommaAlign(unsigned &Alignment, bool &AteExtraComma);
    bool ParseIndexList(SmallVectorImpl<unsigned> &Indices,bool &AteExtraComma);
    bool ParseIndexList(SmallVectorImpl<unsigned> &Indices) {
      bool AteExtraComma;
      if (ParseIndexList(Indices, AteExtraComma)) return true;
      if (AteExtraComma)
        return TokError("expected index");
      return false;
    }

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
    bool ParseUnnamedGlobal();
    bool ParseNamedGlobal();
    bool ParseGlobal(const std::string &Name, LocTy Loc, unsigned Linkage,
                     bool HasLinkage, unsigned Visibility);
    bool ParseAlias(const std::string &Name, LocTy Loc, unsigned Visibility);
    bool ParseStandaloneMetadata();
    bool ParseNamedMetadata();
    bool ParseMDString(MDString *&Result);
    bool ParseMDNodeID(MDNode *&Result);
    bool ParseMDNodeID(MDNode *&Result, unsigned &SlotNo);

    // Type Parsing.
    bool ParseType(Type *&Result, bool AllowVoid = false);
    bool ParseType(Type *&Result, LocTy &Loc, bool AllowVoid = false) {
      Loc = Lex.getLoc();
      return ParseType(Result, AllowVoid);
    }
    bool ParseAnonStructType(Type *&Result, bool Packed);
    bool ParseStructBody(SmallVectorImpl<Type*> &Body);
    bool ParseStructDefinition(SMLoc TypeLoc, StringRef Name,
                               std::pair<Type*, LocTy> &Entry,
                               Type *&ResultTy);

    bool ParseArrayVectorType(Type *&Result, bool isVector);
    bool ParseFunctionType(Type *&Result);

    // Function Semantic Analysis.
    class PerFunctionState {
      LLParser &P;
      Function &F;
      std::map<std::string, std::pair<Value*, LocTy> > ForwardRefVals;
      std::map<unsigned, std::pair<Value*, LocTy> > ForwardRefValIDs;
      std::vector<Value*> NumberedVals;
      
      /// FunctionNumber - If this is an unnamed function, this is the slot
      /// number of it, otherwise it is -1.
      int FunctionNumber;
    public:
      PerFunctionState(LLParser &p, Function &f, int FunctionNumber);
      ~PerFunctionState();

      Function &getFunction() const { return F; }

      bool FinishFunction();

      /// GetVal - Get a value with the specified name or ID, creating a
      /// forward reference record if needed.  This can return null if the value
      /// exists but does not have the right type.
      Value *GetVal(const std::string &Name, Type *Ty, LocTy Loc);
      Value *GetVal(unsigned ID, Type *Ty, LocTy Loc);

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

    bool ConvertValIDToValue(Type *Ty, ValID &ID, Value *&V,
                             PerFunctionState *PFS);

    bool ParseValue(Type *Ty, Value *&V, PerFunctionState *PFS);
    bool ParseValue(Type *Ty, Value *&V, PerFunctionState &PFS) {
      return ParseValue(Ty, V, &PFS);
    }
    bool ParseValue(Type *Ty, Value *&V, LocTy &Loc,
                    PerFunctionState &PFS) {
      Loc = Lex.getLoc();
      return ParseValue(Ty, V, &PFS);
    }

    bool ParseTypeAndValue(Value *&V, PerFunctionState *PFS);
    bool ParseTypeAndValue(Value *&V, PerFunctionState &PFS) {
      return ParseTypeAndValue(V, &PFS);
    }
    bool ParseTypeAndValue(Value *&V, LocTy &Loc, PerFunctionState &PFS) {
      Loc = Lex.getLoc();
      return ParseTypeAndValue(V, PFS);
    }
    bool ParseTypeAndBasicBlock(BasicBlock *&BB, LocTy &Loc,
                                PerFunctionState &PFS);
    bool ParseTypeAndBasicBlock(BasicBlock *&BB, PerFunctionState &PFS) {
      LocTy Loc;
      return ParseTypeAndBasicBlock(BB, Loc, PFS);
    }


    struct ParamInfo {
      LocTy Loc;
      Value *V;
      Attributes Attrs;
      ParamInfo(LocTy loc, Value *v, Attributes attrs)
        : Loc(loc), V(v), Attrs(attrs) {}
    };
    bool ParseParameterList(SmallVectorImpl<ParamInfo> &ArgList,
                            PerFunctionState &PFS);

    // Constant Parsing.
    bool ParseValID(ValID &ID, PerFunctionState *PFS = NULL);
    bool ParseGlobalValue(Type *Ty, Constant *&V);
    bool ParseGlobalTypeAndValue(Constant *&V);
    bool ParseGlobalValueVector(SmallVectorImpl<Constant*> &Elts);
    bool ParseMetadataListValue(ValID &ID, PerFunctionState *PFS);
    bool ParseMetadataValue(ValID &ID, PerFunctionState *PFS);
    bool ParseMDNodeVector(SmallVectorImpl<Value*> &, PerFunctionState *PFS);
    bool ParseInstructionMetadata(Instruction *Inst, PerFunctionState *PFS);

    // Function Parsing.
    struct ArgInfo {
      LocTy Loc;
      Type *Ty;
      Attributes Attrs;
      std::string Name;
      ArgInfo(LocTy L, Type *ty, Attributes Attr, const std::string &N)
        : Loc(L), Ty(ty), Attrs(Attr), Name(N) {}
    };
    bool ParseArgumentList(SmallVectorImpl<ArgInfo> &ArgList, bool &isVarArg);
    bool ParseFunctionHeader(Function *&Fn, bool isDefine);
    bool ParseFunctionBody(Function &Fn);
    bool ParseBasicBlock(PerFunctionState &PFS);

    // Instruction Parsing.  Each instruction parsing routine can return with a
    // normal result, an error result, or return having eaten an extra comma.
    enum InstResult { InstNormal = 0, InstError = 1, InstExtraComma = 2 };
    int ParseInstruction(Instruction *&Inst, BasicBlock *BB,
                         PerFunctionState &PFS);
    bool ParseCmpPredicate(unsigned &Pred, unsigned Opc);

    bool ParseRet(Instruction *&Inst, BasicBlock *BB, PerFunctionState &PFS);
    bool ParseBr(Instruction *&Inst, PerFunctionState &PFS);
    bool ParseSwitch(Instruction *&Inst, PerFunctionState &PFS);
    bool ParseIndirectBr(Instruction *&Inst, PerFunctionState &PFS);
    bool ParseInvoke(Instruction *&Inst, PerFunctionState &PFS);
    bool ParseResume(Instruction *&Inst, PerFunctionState &PFS);

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
    int ParsePHI(Instruction *&I, PerFunctionState &PFS);
    bool ParseLandingPad(Instruction *&I, PerFunctionState &PFS);
    bool ParseCall(Instruction *&I, PerFunctionState &PFS, bool isTail);
    int ParseAlloc(Instruction *&I, PerFunctionState &PFS);
    int ParseLoad(Instruction *&I, PerFunctionState &PFS);
    int ParseStore(Instruction *&I, PerFunctionState &PFS);
    int ParseCmpXchg(Instruction *&I, PerFunctionState &PFS);
    int ParseAtomicRMW(Instruction *&I, PerFunctionState &PFS);
    int ParseFence(Instruction *&I, PerFunctionState &PFS);
    int ParseGetElementPtr(Instruction *&I, PerFunctionState &PFS);
    int ParseExtractValue(Instruction *&I, PerFunctionState &PFS);
    int ParseInsertValue(Instruction *&I, PerFunctionState &PFS);
    
    bool ResolveForwardRefBlockAddresses(Function *TheFn, 
                             std::vector<std::pair<ValID, GlobalValue*> > &Refs,
                                         PerFunctionState *PFS);
  };
} // End llvm namespace

#endif
