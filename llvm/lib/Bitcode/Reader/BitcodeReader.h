//===- BitcodeReader.h - Internal BitcodeReader impl ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header defines the BitcodeReader class.
//
//===----------------------------------------------------------------------===//

#ifndef BITCODE_READER_H
#define BITCODE_READER_H

#include "llvm/GVMaterializer.h"
#include "llvm/Attributes.h"
#include "llvm/Type.h"
#include "llvm/OperandTraits.h"
#include "llvm/Bitcode/BitstreamReader.h"
#include "llvm/Bitcode/LLVMBitCodes.h"
#include "llvm/Support/ValueHandle.h"
#include "llvm/ADT/DenseMap.h"
#include <vector>

namespace llvm {
  class MemoryBuffer;
  class LLVMContext;
  
//===----------------------------------------------------------------------===//
//                          BitcodeReaderValueList Class
//===----------------------------------------------------------------------===//

class BitcodeReaderValueList {
  std::vector<WeakVH> ValuePtrs;
  
  /// ResolveConstants - As we resolve forward-referenced constants, we add
  /// information about them to this vector.  This allows us to resolve them in
  /// bulk instead of resolving each reference at a time.  See the code in
  /// ResolveConstantForwardRefs for more information about this.
  ///
  /// The key of this vector is the placeholder constant, the value is the slot
  /// number that holds the resolved value.
  typedef std::vector<std::pair<Constant*, unsigned> > ResolveConstantsTy;
  ResolveConstantsTy ResolveConstants;
  LLVMContext &Context;
public:
  BitcodeReaderValueList(LLVMContext &C) : Context(C) {}
  ~BitcodeReaderValueList() {
    assert(ResolveConstants.empty() && "Constants not resolved?");
  }

  // vector compatibility methods
  unsigned size() const { return ValuePtrs.size(); }
  void resize(unsigned N) { ValuePtrs.resize(N); }
  void push_back(Value *V) {
    ValuePtrs.push_back(V);
  }
  
  void clear() {
    assert(ResolveConstants.empty() && "Constants not resolved?");
    ValuePtrs.clear();
  }
  
  Value *operator[](unsigned i) const {
    assert(i < ValuePtrs.size());
    return ValuePtrs[i];
  }
  
  Value *back() const { return ValuePtrs.back(); }
    void pop_back() { ValuePtrs.pop_back(); }
  bool empty() const { return ValuePtrs.empty(); }
  void shrinkTo(unsigned N) {
    assert(N <= size() && "Invalid shrinkTo request!");
    ValuePtrs.resize(N);
  }
  
  Constant *getConstantFwdRef(unsigned Idx, Type *Ty);
  Value *getValueFwdRef(unsigned Idx, Type *Ty);
  
  void AssignValue(Value *V, unsigned Idx);
  
  /// ResolveConstantForwardRefs - Once all constants are read, this method bulk
  /// resolves any forward references.
  void ResolveConstantForwardRefs();
};


//===----------------------------------------------------------------------===//
//                          BitcodeReaderMDValueList Class
//===----------------------------------------------------------------------===//

class BitcodeReaderMDValueList {
  std::vector<WeakVH> MDValuePtrs;
  
  LLVMContext &Context;
public:
  BitcodeReaderMDValueList(LLVMContext& C) : Context(C) {}

  // vector compatibility methods
  unsigned size() const       { return MDValuePtrs.size(); }
  void resize(unsigned N)     { MDValuePtrs.resize(N); }
  void push_back(Value *V)    { MDValuePtrs.push_back(V);  }
  void clear()                { MDValuePtrs.clear();  }
  Value *back() const         { return MDValuePtrs.back(); }
  void pop_back()             { MDValuePtrs.pop_back(); }
  bool empty() const          { return MDValuePtrs.empty(); }
  
  Value *operator[](unsigned i) const {
    assert(i < MDValuePtrs.size());
    return MDValuePtrs[i];
  }
  
  void shrinkTo(unsigned N) {
    assert(N <= size() && "Invalid shrinkTo request!");
    MDValuePtrs.resize(N);
  }

  Value *getValueFwdRef(unsigned Idx);
  void AssignValue(Value *V, unsigned Idx);
};

class BitcodeReader : public GVMaterializer {
  LLVMContext &Context;
  Module *TheModule;
  MemoryBuffer *Buffer;
  bool BufferOwned;
  OwningPtr<BitstreamReader> StreamFile;
  BitstreamCursor Stream;
  DataStreamer *LazyStreamer;
  uint64_t NextUnreadBit;
  bool SeenValueSymbolTable;
  
  const char *ErrorString;
  
  std::vector<Type*> TypeList;
  BitcodeReaderValueList ValueList;
  BitcodeReaderMDValueList MDValueList;
  SmallVector<Instruction *, 64> InstructionList;
  SmallVector<SmallVector<uint64_t, 64>, 64> UseListRecords;

  std::vector<std::pair<GlobalVariable*, unsigned> > GlobalInits;
  std::vector<std::pair<GlobalAlias*, unsigned> > AliasInits;
  
  /// MAttributes - The set of attributes by index.  Index zero in the
  /// file is for null, and is thus not represented here.  As such all indices
  /// are off by one.
  std::vector<AttrListPtr> MAttributes;
  
  /// FunctionBBs - While parsing a function body, this is a list of the basic
  /// blocks for the function.
  std::vector<BasicBlock*> FunctionBBs;
  
  // When reading the module header, this list is populated with functions that
  // have bodies later in the file.
  std::vector<Function*> FunctionsWithBodies;

  // When intrinsic functions are encountered which require upgrading they are 
  // stored here with their replacement function.
  typedef std::vector<std::pair<Function*, Function*> > UpgradedIntrinsicMap;
  UpgradedIntrinsicMap UpgradedIntrinsics;

  // Map the bitcode's custom MDKind ID to the Module's MDKind ID.
  DenseMap<unsigned, unsigned> MDKindMap;
  
  // Several operations happen after the module header has been read, but
  // before function bodies are processed. This keeps track of whether
  // we've done this yet.
  bool SeenFirstFunctionBody;
  
  /// DeferredFunctionInfo - When function bodies are initially scanned, this
  /// map contains info about where to find deferred function body in the
  /// stream.
  DenseMap<Function*, uint64_t> DeferredFunctionInfo;
  
  /// BlockAddrFwdRefs - These are blockaddr references to basic blocks.  These
  /// are resolved lazily when functions are loaded.
  typedef std::pair<unsigned, GlobalVariable*> BlockAddrRefTy;
  DenseMap<Function*, std::vector<BlockAddrRefTy> > BlockAddrFwdRefs;

  /// UseRelativeIDs - Indicates that we are using a new encoding for
  /// instruction operands where most operands in the current
  /// FUNCTION_BLOCK are encoded relative to the instruction number,
  /// for a more compact encoding.  Some instruction operands are not
  /// relative to the instruction ID: basic block numbers, and types.
  /// Once the old style function blocks have been phased out, we would
  /// not need this flag.
  bool UseRelativeIDs;

public:
  explicit BitcodeReader(MemoryBuffer *buffer, LLVMContext &C)
    : Context(C), TheModule(0), Buffer(buffer), BufferOwned(false),
      LazyStreamer(0), NextUnreadBit(0), SeenValueSymbolTable(false),
      ErrorString(0), ValueList(C), MDValueList(C),
      SeenFirstFunctionBody(false), UseRelativeIDs(false) {
  }
  explicit BitcodeReader(DataStreamer *streamer, LLVMContext &C)
    : Context(C), TheModule(0), Buffer(0), BufferOwned(false),
      LazyStreamer(streamer), NextUnreadBit(0), SeenValueSymbolTable(false),
      ErrorString(0), ValueList(C), MDValueList(C),
      SeenFirstFunctionBody(false), UseRelativeIDs(false) {
  }
  ~BitcodeReader() {
    FreeState();
  }

  void materializeForwardReferencedFunctions();

  void FreeState();
  
  /// setBufferOwned - If this is true, the reader will destroy the MemoryBuffer
  /// when the reader is destroyed.
  void setBufferOwned(bool Owned) { BufferOwned = Owned; }
  
  virtual bool isMaterializable(const GlobalValue *GV) const;
  virtual bool isDematerializable(const GlobalValue *GV) const;
  virtual bool Materialize(GlobalValue *GV, std::string *ErrInfo = 0);
  virtual bool MaterializeModule(Module *M, std::string *ErrInfo = 0);
  virtual void Dematerialize(GlobalValue *GV);

  bool Error(const char *Str) {
    ErrorString = Str;
    return true;
  }
  const char *getErrorString() const { return ErrorString; }
  
  /// @brief Main interface to parsing a bitcode buffer.
  /// @returns true if an error occurred.
  bool ParseBitcodeInto(Module *M);

  /// @brief Cheap mechanism to just extract module triple
  /// @returns true if an error occurred.
  bool ParseTriple(std::string &Triple);

  static uint64_t decodeSignRotatedValue(uint64_t V);

private:
  Type *getTypeByID(unsigned ID);
  Value *getFnValueByID(unsigned ID, Type *Ty) {
    if (Ty && Ty->isMetadataTy())
      return MDValueList.getValueFwdRef(ID);
    return ValueList.getValueFwdRef(ID, Ty);
  }
  BasicBlock *getBasicBlock(unsigned ID) const {
    if (ID >= FunctionBBs.size()) return 0; // Invalid ID
    return FunctionBBs[ID];
  }
  AttrListPtr getAttributes(unsigned i) const {
    if (i-1 < MAttributes.size())
      return MAttributes[i-1];
    return AttrListPtr();
  }
  
  /// getValueTypePair - Read a value/type pair out of the specified record from
  /// slot 'Slot'.  Increment Slot past the number of slots used in the record.
  /// Return true on failure.
  bool getValueTypePair(SmallVector<uint64_t, 64> &Record, unsigned &Slot,
                        unsigned InstNum, Value *&ResVal) {
    if (Slot == Record.size()) return true;
    unsigned ValNo = (unsigned)Record[Slot++];
    // Adjust the ValNo, if it was encoded relative to the InstNum.
    if (UseRelativeIDs)
      ValNo = InstNum - ValNo;
    if (ValNo < InstNum) {
      // If this is not a forward reference, just return the value we already
      // have.
      ResVal = getFnValueByID(ValNo, 0);
      return ResVal == 0;
    } else if (Slot == Record.size()) {
      return true;
    }

    unsigned TypeNo = (unsigned)Record[Slot++];
    ResVal = getFnValueByID(ValNo, getTypeByID(TypeNo));
    return ResVal == 0;
  }

  /// popValue - Read a value out of the specified record from slot 'Slot'.
  /// Increment Slot past the number of slots used by the value in the record.
  /// Return true if there is an error.
  bool popValue(SmallVector<uint64_t, 64> &Record, unsigned &Slot,
                unsigned InstNum, Type *Ty, Value *&ResVal) {
    if (getValue(Record, Slot, InstNum, Ty, ResVal))
      return true;
    // All values currently take a single record slot.
    ++Slot;
    return false;
  }

  /// getValue -- Like popValue, but does not increment the Slot number.
  bool getValue(SmallVector<uint64_t, 64> &Record, unsigned Slot,
                unsigned InstNum, Type *Ty, Value *&ResVal) {
    ResVal = getValue(Record, Slot, InstNum, Ty);
    return ResVal == 0;
  }

  /// getValue -- Version of getValue that returns ResVal directly,
  /// or 0 if there is an error.
  Value *getValue(SmallVector<uint64_t, 64> &Record, unsigned Slot,
                  unsigned InstNum, Type *Ty) {
    if (Slot == Record.size()) return 0;
    unsigned ValNo = (unsigned)Record[Slot];
    // Adjust the ValNo, if it was encoded relative to the InstNum.
    if (UseRelativeIDs)
      ValNo = InstNum - ValNo;
    return getFnValueByID(ValNo, Ty);
  }

  /// getValueSigned -- Like getValue, but decodes signed VBRs.
  Value *getValueSigned(SmallVector<uint64_t, 64> &Record, unsigned Slot,
                        unsigned InstNum, Type *Ty) {
    if (Slot == Record.size()) return 0;
    unsigned ValNo = (unsigned)decodeSignRotatedValue(Record[Slot]);
    // Adjust the ValNo, if it was encoded relative to the InstNum.
    if (UseRelativeIDs)
      ValNo = InstNum - ValNo;
    return getFnValueByID(ValNo, Ty);
  }

  bool ParseModule(bool Resume);
  bool ParseAttributeBlock();
  bool ParseTypeTable();
  bool ParseTypeTableBody();

  bool ParseValueSymbolTable();
  bool ParseConstants();
  bool RememberAndSkipFunctionBody();
  bool ParseFunctionBody(Function *F);
  bool GlobalCleanup();
  bool ResolveGlobalAndAliasInits();
  bool ParseMetadata();
  bool ParseMetadataAttachment();
  bool ParseModuleTriple(std::string &Triple);
  bool ParseUseLists();
  bool InitStream();
  bool InitStreamFromBuffer();
  bool InitLazyStream();
  bool FindFunctionInStream(Function *F,
         DenseMap<Function*, uint64_t>::iterator DeferredFunctionInfoIterator);
};
  
} // End llvm namespace

#endif
