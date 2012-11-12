//===--- llvm/Analysis/DebugInfo.h - Debug Information Helpers --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a bunch of datatypes that are useful for creating and
// walking debug info in LLVM IR form. They essentially provide wrappers around
// the information in the global variables that's needed when constructing the
// DWARF information.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_DEBUGINFO_H
#define LLVM_ANALYSIS_DEBUGINFO_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Dwarf.h"

namespace llvm {
  class BasicBlock;
  class Constant;
  class Function;
  class GlobalVariable;
  class Module;
  class Type;
  class Value;
  class DbgDeclareInst;
  class Instruction;
  class MDNode;
  class NamedMDNode;
  class LLVMContext;
  class raw_ostream;

  class DIFile;
  class DISubprogram;
  class DILexicalBlock;
  class DILexicalBlockFile;
  class DIVariable;
  class DIType;
  class DIObjCProperty;

  /// DIDescriptor - A thin wraper around MDNode to access encoded debug info.
  /// This should not be stored in a container, because the underlying MDNode
  /// may change in certain situations.
  class DIDescriptor {
  public:
    enum {
      FlagPrivate            = 1 << 0,
      FlagProtected          = 1 << 1,
      FlagFwdDecl            = 1 << 2,
      FlagAppleBlock         = 1 << 3,
      FlagBlockByrefStruct   = 1 << 4,
      FlagVirtual            = 1 << 5,
      FlagArtificial         = 1 << 6,
      FlagExplicit           = 1 << 7,
      FlagPrototyped         = 1 << 8,
      FlagObjcClassComplete  = 1 << 9,
      FlagObjectPointer      = 1 << 10
    };
  protected:
    const MDNode *DbgNode;

    StringRef getStringField(unsigned Elt) const;
    unsigned getUnsignedField(unsigned Elt) const {
      return (unsigned)getUInt64Field(Elt);
    }
    uint64_t getUInt64Field(unsigned Elt) const;
    DIDescriptor getDescriptorField(unsigned Elt) const;

    template <typename DescTy>
    DescTy getFieldAs(unsigned Elt) const {
      return DescTy(getDescriptorField(Elt));
    }

    GlobalVariable *getGlobalVariableField(unsigned Elt) const;
    Constant *getConstantField(unsigned Elt) const;
    Function *getFunctionField(unsigned Elt) const;
    void replaceFunctionField(unsigned Elt, Function *F);

  public:
    explicit DIDescriptor() : DbgNode(0) {}
    explicit DIDescriptor(const MDNode *N) : DbgNode(N) {}
    explicit DIDescriptor(const DIFile F);
    explicit DIDescriptor(const DISubprogram F);
    explicit DIDescriptor(const DILexicalBlockFile F);
    explicit DIDescriptor(const DILexicalBlock F);
    explicit DIDescriptor(const DIVariable F);
    explicit DIDescriptor(const DIType F);

    bool Verify() const { return DbgNode != 0; }

    operator MDNode *() const { return const_cast<MDNode*>(DbgNode); }
    MDNode *operator ->() const { return const_cast<MDNode*>(DbgNode); }

    unsigned getVersion() const {
      return getUnsignedField(0) & LLVMDebugVersionMask;
    }

    unsigned getTag() const {
      return getUnsignedField(0) & ~LLVMDebugVersionMask;
    }

    bool isDerivedType() const;
    bool isCompositeType() const;
    bool isBasicType() const;
    bool isVariable() const;
    bool isSubprogram() const;
    bool isGlobalVariable() const;
    bool isScope() const;
    bool isFile() const;
    bool isCompileUnit() const;
    bool isNameSpace() const;
    bool isLexicalBlockFile() const;
    bool isLexicalBlock() const;
    bool isSubrange() const;
    bool isEnumerator() const;
    bool isType() const;
    bool isGlobal() const;
    bool isUnspecifiedParameter() const;
    bool isTemplateTypeParameter() const;
    bool isTemplateValueParameter() const;
    bool isObjCProperty() const;

    /// print - print descriptor.
    void print(raw_ostream &OS) const;

    /// dump - print descriptor to dbgs() with a newline.
    void dump() const;
  };

  /// DISubrange - This is used to represent ranges, for array bounds.
  class DISubrange : public DIDescriptor {
    friend class DIDescriptor;
    void printInternal(raw_ostream &OS) const;
  public:
    explicit DISubrange(const MDNode *N = 0) : DIDescriptor(N) {}

    uint64_t getLo() const { return getUInt64Field(1); }
    uint64_t getHi() const { return getUInt64Field(2); }
  };

  /// DIArray - This descriptor holds an array of descriptors.
  class DIArray : public DIDescriptor {
  public:
    explicit DIArray(const MDNode *N = 0)
      : DIDescriptor(N) {}

    unsigned getNumElements() const;
    DIDescriptor getElement(unsigned Idx) const {
      return getDescriptorField(Idx);
    }
  };

  /// DIScope - A base class for various scopes.
  class DIScope : public DIDescriptor {
  protected:
    friend class DIDescriptor;
    void printInternal(raw_ostream &OS) const;
  public:
    explicit DIScope(const MDNode *N = 0) : DIDescriptor (N) {}

    StringRef getFilename() const;
    StringRef getDirectory() const;
  };

  /// DICompileUnit - A wrapper for a compile unit.
  class DICompileUnit : public DIScope {
    friend class DIDescriptor;
    void printInternal(raw_ostream &OS) const;
  public:
    explicit DICompileUnit(const MDNode *N = 0) : DIScope(N) {}

    unsigned getLanguage() const   { return getUnsignedField(2); }
    StringRef getFilename() const  { return getStringField(3);   }
    StringRef getDirectory() const { return getStringField(4);   }
    StringRef getProducer() const  { return getStringField(5);   }

    /// isMain - Each input file is encoded as a separate compile unit in LLVM
    /// debugging information output. However, many target specific tool chains
    /// prefer to encode only one compile unit in an object file. In this
    /// situation, the LLVM code generator will include  debugging information
    /// entities in the compile unit that is marked as main compile unit. The
    /// code generator accepts maximum one main compile unit per module. If a
    /// module does not contain any main compile unit then the code generator
    /// will emit multiple compile units in the output object file.

    bool isMain() const                { return getUnsignedField(6) != 0; }
    bool isOptimized() const           { return getUnsignedField(7) != 0; }
    StringRef getFlags() const       { return getStringField(8);   }
    unsigned getRunTimeVersion() const { return getUnsignedField(9); }

    DIArray getEnumTypes() const;
    DIArray getRetainedTypes() const;
    DIArray getSubprograms() const;
    DIArray getGlobalVariables() const;

    /// Verify - Verify that a compile unit is well formed.
    bool Verify() const;
  };

  /// DIFile - This is a wrapper for a file.
  class DIFile : public DIScope {
    friend class DIDescriptor;
    void printInternal(raw_ostream &OS) const {} // FIXME: Output something?
  public:
    explicit DIFile(const MDNode *N = 0) : DIScope(N) {
      if (DbgNode && !isFile())
        DbgNode = 0;
    }
    StringRef getFilename() const  { return getStringField(1);   }
    StringRef getDirectory() const { return getStringField(2);   }
    DICompileUnit getCompileUnit() const{ 
      assert (getVersion() <= LLVMDebugVersion10  && "Invalid CompileUnit!");
      return getFieldAs<DICompileUnit>(3); 
    }
  };

  /// DIEnumerator - A wrapper for an enumerator (e.g. X and Y in 'enum {X,Y}').
  /// FIXME: it seems strange that this doesn't have either a reference to the
  /// type/precision or a file/line pair for location info.
  class DIEnumerator : public DIDescriptor {
    friend class DIDescriptor;
    void printInternal(raw_ostream &OS) const;
  public:
    explicit DIEnumerator(const MDNode *N = 0) : DIDescriptor(N) {}

    StringRef getName() const        { return getStringField(1); }
    uint64_t getEnumValue() const      { return getUInt64Field(2); }
  };

  /// DIType - This is a wrapper for a type.
  /// FIXME: Types should be factored much better so that CV qualifiers and
  /// others do not require a huge and empty descriptor full of zeros.
  class DIType : public DIScope {
  protected:
    friend class DIDescriptor;
    void printInternal(raw_ostream &OS) const;
    // This ctor is used when the Tag has already been validated by a derived
    // ctor.
    DIType(const MDNode *N, bool, bool) : DIScope(N) {}
  public:
    /// Verify - Verify that a type descriptor is well formed.
    bool Verify() const;
    explicit DIType(const MDNode *N);
    explicit DIType() {}

    DIScope getContext() const          { return getFieldAs<DIScope>(1); }
    StringRef getName() const           { return getStringField(2);     }
    DICompileUnit getCompileUnit() const{ 
      assert (getVersion() <= LLVMDebugVersion10 && "Invalid getCompileUnit!");
     if (getVersion() == llvm::LLVMDebugVersion7)
       return getFieldAs<DICompileUnit>(3);
     
     return getFieldAs<DIFile>(3).getCompileUnit();
    }
    DIFile getFile() const              { return getFieldAs<DIFile>(3); }
    unsigned getLineNumber() const      { return getUnsignedField(4); }
    uint64_t getSizeInBits() const      { return getUInt64Field(5); }
    uint64_t getAlignInBits() const     { return getUInt64Field(6); }
    // FIXME: Offset is only used for DW_TAG_member nodes.  Making every type
    // carry this is just plain insane.
    uint64_t getOffsetInBits() const    { return getUInt64Field(7); }
    unsigned getFlags() const           { return getUnsignedField(8); }
    bool isPrivate() const {
      return (getFlags() & FlagPrivate) != 0;
    }
    bool isProtected() const {
      return (getFlags() & FlagProtected) != 0;
    }
    bool isForwardDecl() const {
      return (getFlags() & FlagFwdDecl) != 0;
    }
    // isAppleBlock - Return true if this is the Apple Blocks extension.
    bool isAppleBlockExtension() const {
      return (getFlags() & FlagAppleBlock) != 0;
    }
    bool isBlockByrefStruct() const {
      return (getFlags() & FlagBlockByrefStruct) != 0;
    }
    bool isVirtual() const {
      return (getFlags() & FlagVirtual) != 0;
    }
    bool isArtificial() const {
      return (getFlags() & FlagArtificial) != 0;
    }
    bool isObjectPointer() const {
      return (getFlags() & FlagObjectPointer) != 0;
    }
    bool isObjcClassComplete() const {
      return (getFlags() & FlagObjcClassComplete) != 0;
    }
    bool isValid() const {
      return DbgNode && (isBasicType() || isDerivedType() || isCompositeType());
    }
    StringRef getDirectory() const  { 
      if (getVersion() == llvm::LLVMDebugVersion7)
        return getCompileUnit().getDirectory();

      return getFieldAs<DIFile>(3).getDirectory();
    }
    StringRef getFilename() const  { 
      if (getVersion() == llvm::LLVMDebugVersion7)
        return getCompileUnit().getFilename();

      return getFieldAs<DIFile>(3).getFilename();
    }

    /// isUnsignedDIType - Return true if type encoding is unsigned.
    bool isUnsignedDIType();

    /// replaceAllUsesWith - Replace all uses of debug info referenced by
    /// this descriptor.
    void replaceAllUsesWith(DIDescriptor &D);
    void replaceAllUsesWith(MDNode *D);
  };

  /// DIBasicType - A basic type, like 'int' or 'float'.
  class DIBasicType : public DIType {
  public:
    explicit DIBasicType(const MDNode *N = 0) : DIType(N) {}

    unsigned getEncoding() const { return getUnsignedField(9); }

    /// Verify - Verify that a basic type descriptor is well formed.
    bool Verify() const;
  };

  /// DIDerivedType - A simple derived type, like a const qualified type,
  /// a typedef, a pointer or reference, etc.
  class DIDerivedType : public DIType {
    friend class DIDescriptor;
    void printInternal(raw_ostream &OS) const;
  protected:
    explicit DIDerivedType(const MDNode *N, bool, bool)
      : DIType(N, true, true) {}
  public:
    explicit DIDerivedType(const MDNode *N = 0)
      : DIType(N, true, true) {}

    DIType getTypeDerivedFrom() const { return getFieldAs<DIType>(9); }

    /// getOriginalTypeSize - If this type is derived from a base type then
    /// return base type size.
    uint64_t getOriginalTypeSize() const;

    /// getObjCProperty - Return property node, if this ivar is 
    /// associated with one.
    MDNode *getObjCProperty() const;

    StringRef getObjCPropertyName() const { 
      if (getVersion() > LLVMDebugVersion11)
        return StringRef();
      return getStringField(10); 
    }
    StringRef getObjCPropertyGetterName() const {
      assert (getVersion() <= LLVMDebugVersion11  && "Invalid Request");
      return getStringField(11);
    }
    StringRef getObjCPropertySetterName() const {
      assert (getVersion() <= LLVMDebugVersion11  && "Invalid Request");
      return getStringField(12);
    }
    bool isReadOnlyObjCProperty() {
      assert (getVersion() <= LLVMDebugVersion11  && "Invalid Request");
      return (getUnsignedField(13) & dwarf::DW_APPLE_PROPERTY_readonly) != 0;
    }
    bool isReadWriteObjCProperty() {
      assert (getVersion() <= LLVMDebugVersion11  && "Invalid Request");
      return (getUnsignedField(13) & dwarf::DW_APPLE_PROPERTY_readwrite) != 0;
    }
    bool isAssignObjCProperty() {
      assert (getVersion() <= LLVMDebugVersion11  && "Invalid Request");
      return (getUnsignedField(13) & dwarf::DW_APPLE_PROPERTY_assign) != 0;
    }
    bool isRetainObjCProperty() {
      assert (getVersion() <= LLVMDebugVersion11  && "Invalid Request");
      return (getUnsignedField(13) & dwarf::DW_APPLE_PROPERTY_retain) != 0;
    }
    bool isCopyObjCProperty() {
      assert (getVersion() <= LLVMDebugVersion11  && "Invalid Request");
      return (getUnsignedField(13) & dwarf::DW_APPLE_PROPERTY_copy) != 0;
    }
    bool isNonAtomicObjCProperty() {
      assert (getVersion() <= LLVMDebugVersion11  && "Invalid Request");
      return (getUnsignedField(13) & dwarf::DW_APPLE_PROPERTY_nonatomic) != 0;
    }

    /// Verify - Verify that a derived type descriptor is well formed.
    bool Verify() const;
  };

  /// DICompositeType - This descriptor holds a type that can refer to multiple
  /// other types, like a function or struct.
  /// FIXME: Why is this a DIDerivedType??
  class DICompositeType : public DIDerivedType {
    friend class DIDescriptor;
    void printInternal(raw_ostream &OS) const;
  public:
    explicit DICompositeType(const MDNode *N = 0)
      : DIDerivedType(N, true, true) {
      if (N && !isCompositeType())
        DbgNode = 0;
    }

    DIArray getTypeArray() const { return getFieldAs<DIArray>(10); }
    unsigned getRunTimeLang() const { return getUnsignedField(11); }
    DICompositeType getContainingType() const {
      return getFieldAs<DICompositeType>(12);
    }
    DIArray getTemplateParams() const { return getFieldAs<DIArray>(13); }

    /// Verify - Verify that a composite type descriptor is well formed.
    bool Verify() const;
  };

  /// DITemplateTypeParameter - This is a wrapper for template type parameter.
  class DITemplateTypeParameter : public DIDescriptor {
  public:
    explicit DITemplateTypeParameter(const MDNode *N = 0) : DIDescriptor(N) {}

    DIScope getContext() const       { return getFieldAs<DIScope>(1); }
    StringRef getName() const        { return getStringField(2); }
    DIType getType() const           { return getFieldAs<DIType>(3); }
    StringRef getFilename() const    { 
      return getFieldAs<DIFile>(4).getFilename();
    }
    StringRef getDirectory() const   { 
      return getFieldAs<DIFile>(4).getDirectory();
    }
    unsigned getLineNumber() const   { return getUnsignedField(5); }
    unsigned getColumnNumber() const { return getUnsignedField(6); }
  };

  /// DITemplateValueParameter - This is a wrapper for template value parameter.
  class DITemplateValueParameter : public DIDescriptor {
  public:
    explicit DITemplateValueParameter(const MDNode *N = 0) : DIDescriptor(N) {}

    DIScope getContext() const       { return getFieldAs<DIScope>(1); }
    StringRef getName() const        { return getStringField(2); }
    DIType getType() const           { return getFieldAs<DIType>(3); }
    uint64_t getValue() const         { return getUInt64Field(4); }
    StringRef getFilename() const    { 
      return getFieldAs<DIFile>(5).getFilename();
    }
    StringRef getDirectory() const   { 
      return getFieldAs<DIFile>(5).getDirectory();
    }
    unsigned getLineNumber() const   { return getUnsignedField(6); }
    unsigned getColumnNumber() const { return getUnsignedField(7); }
  };

  /// DISubprogram - This is a wrapper for a subprogram (e.g. a function).
  class DISubprogram : public DIScope {
    friend class DIDescriptor;
    void printInternal(raw_ostream &OS) const;
  public:
    explicit DISubprogram(const MDNode *N = 0) : DIScope(N) {}

    DIScope getContext() const          { return getFieldAs<DIScope>(2); }
    StringRef getName() const         { return getStringField(3); }
    StringRef getDisplayName() const  { return getStringField(4); }
    StringRef getLinkageName() const  { return getStringField(5); }
    DICompileUnit getCompileUnit() const{ 
      assert (getVersion() <= LLVMDebugVersion10 && "Invalid getCompileUnit!");
      if (getVersion() == llvm::LLVMDebugVersion7)
        return getFieldAs<DICompileUnit>(6);

      return getFieldAs<DIFile>(6).getCompileUnit(); 
    }
    unsigned getLineNumber() const      { return getUnsignedField(7); }
    DICompositeType getType() const { return getFieldAs<DICompositeType>(8); }

    /// getReturnTypeName - Subprogram return types are encoded either as
    /// DIType or as DICompositeType.
    StringRef getReturnTypeName() const {
      DICompositeType DCT(getFieldAs<DICompositeType>(8));
      if (DCT.Verify()) {
        DIArray A = DCT.getTypeArray();
        DIType T(A.getElement(0));
        return T.getName();
      }
      DIType T(getFieldAs<DIType>(8));
      return T.getName();
    }

    /// isLocalToUnit - Return true if this subprogram is local to the current
    /// compile unit, like 'static' in C.
    unsigned isLocalToUnit() const     { return getUnsignedField(9); }
    unsigned isDefinition() const      { return getUnsignedField(10); }

    unsigned getVirtuality() const { return getUnsignedField(11); }
    unsigned getVirtualIndex() const { return getUnsignedField(12); }

    DICompositeType getContainingType() const {
      return getFieldAs<DICompositeType>(13);
    }

    unsigned isArtificial() const    { 
      if (getVersion() <= llvm::LLVMDebugVersion8)
        return getUnsignedField(14); 
      return (getUnsignedField(14) & FlagArtificial) != 0;
    }
    /// isPrivate - Return true if this subprogram has "private"
    /// access specifier.
    bool isPrivate() const    { 
      if (getVersion() <= llvm::LLVMDebugVersion8)
        return false;
      return (getUnsignedField(14) & FlagPrivate) != 0;
    }
    /// isProtected - Return true if this subprogram has "protected"
    /// access specifier.
    bool isProtected() const    { 
      if (getVersion() <= llvm::LLVMDebugVersion8)
        return false;
      return (getUnsignedField(14) & FlagProtected) != 0;
    }
    /// isExplicit - Return true if this subprogram is marked as explicit.
    bool isExplicit() const    { 
      if (getVersion() <= llvm::LLVMDebugVersion8)
        return false;
      return (getUnsignedField(14) & FlagExplicit) != 0;
    }
    /// isPrototyped - Return true if this subprogram is prototyped.
    bool isPrototyped() const    { 
      if (getVersion() <= llvm::LLVMDebugVersion8)
        return false;
      return (getUnsignedField(14) & FlagPrototyped) != 0;
    }

    unsigned isOptimized() const;

    StringRef getFilename() const    { 
      if (getVersion() == llvm::LLVMDebugVersion7)
        return getCompileUnit().getFilename();

      return getFieldAs<DIFile>(6).getFilename(); 
    }

    StringRef getDirectory() const   { 
      if (getVersion() == llvm::LLVMDebugVersion7)
        return getCompileUnit().getFilename();

      return getFieldAs<DIFile>(6).getDirectory(); 
    }

    /// getScopeLineNumber - Get the beginning of the scope of the
    /// function, not necessarily where the name of the program
    /// starts.
    unsigned getScopeLineNumber() const { return getUnsignedField(20); }

    /// Verify - Verify that a subprogram descriptor is well formed.
    bool Verify() const;

    /// describes - Return true if this subprogram provides debugging
    /// information for the function F.
    bool describes(const Function *F);

    Function *getFunction() const { return getFunctionField(16); }
    void replaceFunction(Function *F) { replaceFunctionField(16, F); }
    DIArray getTemplateParams() const { return getFieldAs<DIArray>(17); }
    DISubprogram getFunctionDeclaration() const {
      return getFieldAs<DISubprogram>(18);
    }
    MDNode *getVariablesNodes() const;
    DIArray getVariables() const;
  };

  /// DIGlobalVariable - This is a wrapper for a global variable.
  class DIGlobalVariable : public DIDescriptor {
    friend class DIDescriptor;
    void printInternal(raw_ostream &OS) const;
  public:
    explicit DIGlobalVariable(const MDNode *N = 0) : DIDescriptor(N) {}

    DIScope getContext() const          { return getFieldAs<DIScope>(2); }
    StringRef getName() const         { return getStringField(3); }
    StringRef getDisplayName() const  { return getStringField(4); }
    StringRef getLinkageName() const  { return getStringField(5); }
    DICompileUnit getCompileUnit() const{ 
      assert (getVersion() <= LLVMDebugVersion10 && "Invalid getCompileUnit!");
      if (getVersion() == llvm::LLVMDebugVersion7)
        return getFieldAs<DICompileUnit>(6);

      DIFile F = getFieldAs<DIFile>(6); 
      return F.getCompileUnit();
    }
    StringRef getFilename() const {
      if (getVersion() <= llvm::LLVMDebugVersion10)
        return getContext().getFilename();
      return getFieldAs<DIFile>(6).getFilename();
    } 
    StringRef getDirectory() const {
      if (getVersion() <= llvm::LLVMDebugVersion10)
        return getContext().getDirectory();
      return getFieldAs<DIFile>(6).getDirectory();

    } 

    unsigned getLineNumber() const      { return getUnsignedField(7); }
    DIType getType() const              { return getFieldAs<DIType>(8); }
    unsigned isLocalToUnit() const      { return getUnsignedField(9); }
    unsigned isDefinition() const       { return getUnsignedField(10); }

    GlobalVariable *getGlobal() const { return getGlobalVariableField(11); }
    Constant *getConstant() const   { return getConstantField(11); }

    /// Verify - Verify that a global variable descriptor is well formed.
    bool Verify() const;
  };

  /// DIVariable - This is a wrapper for a variable (e.g. parameter, local,
  /// global etc).
  class DIVariable : public DIDescriptor {
    friend class DIDescriptor;
    void printInternal(raw_ostream &OS) const;
  public:
    explicit DIVariable(const MDNode *N = 0)
      : DIDescriptor(N) {}

    DIScope getContext() const          { return getFieldAs<DIScope>(1); }
    StringRef getName() const           { return getStringField(2);     }
    DICompileUnit getCompileUnit() const { 
      assert (getVersion() <= LLVMDebugVersion10 && "Invalid getCompileUnit!");
      if (getVersion() == llvm::LLVMDebugVersion7)
        return getFieldAs<DICompileUnit>(3);

      DIFile F = getFieldAs<DIFile>(3); 
      return F.getCompileUnit();
    }
    unsigned getLineNumber() const      { 
      return (getUnsignedField(4) << 8) >> 8; 
    }
    unsigned getArgNumber() const       {
      unsigned L = getUnsignedField(4); 
      return L >> 24;
    }
    DIType getType() const              { return getFieldAs<DIType>(5); }
    
    /// isArtificial - Return true if this variable is marked as "artificial".
    bool isArtificial() const    { 
      if (getVersion() <= llvm::LLVMDebugVersion8)
        return false;
      return (getUnsignedField(6) & FlagArtificial) != 0;
    }

    bool isObjectPointer() const {
      return (getUnsignedField(6) & FlagObjectPointer) != 0;
    }

    /// getInlinedAt - If this variable is inlined then return inline location.
    MDNode *getInlinedAt() const;

    /// Verify - Verify that a variable descriptor is well formed.
    bool Verify() const;

    /// HasComplexAddr - Return true if the variable has a complex address.
    bool hasComplexAddress() const {
      return getNumAddrElements() > 0;
    }

    unsigned getNumAddrElements() const;
    
    uint64_t getAddrElement(unsigned Idx) const {
      if (getVersion() <= llvm::LLVMDebugVersion8)
        return getUInt64Field(Idx+6);
      if (getVersion() == llvm::LLVMDebugVersion9)
        return getUInt64Field(Idx+7);
      return getUInt64Field(Idx+8);
    }

    /// isBlockByrefVariable - Return true if the variable was declared as
    /// a "__block" variable (Apple Blocks).
    bool isBlockByrefVariable() const {
      return getType().isBlockByrefStruct();
    }

    /// isInlinedFnArgument - Return trule if this variable provides debugging
    /// information for an inlined function arguments.
    bool isInlinedFnArgument(const Function *CurFn);

    void printExtendedName(raw_ostream &OS) const;
  };

  /// DILexicalBlock - This is a wrapper for a lexical block.
  class DILexicalBlock : public DIScope {
  public:
    explicit DILexicalBlock(const MDNode *N = 0) : DIScope(N) {}
    DIScope getContext() const       { return getFieldAs<DIScope>(1);      }
    unsigned getLineNumber() const   { return getUnsignedField(2);         }
    unsigned getColumnNumber() const { return getUnsignedField(3);         }
    StringRef getDirectory() const {
      StringRef dir = getFieldAs<DIFile>(4).getDirectory();
      return !dir.empty() ? dir : getContext().getDirectory();
    }
    StringRef getFilename() const {
      StringRef filename = getFieldAs<DIFile>(4).getFilename();
      return !filename.empty() ? filename : getContext().getFilename();
    }
  };

  /// DILexicalBlockFile - This is a wrapper for a lexical block with
  /// a filename change.
  class DILexicalBlockFile : public DIScope {
  public:
    explicit DILexicalBlockFile(const MDNode *N = 0) : DIScope(N) {}
    DIScope getContext() const { return getScope().getContext(); }
    unsigned getLineNumber() const { return getScope().getLineNumber(); }
    unsigned getColumnNumber() const { return getScope().getColumnNumber(); }
    StringRef getDirectory() const {
      StringRef dir = getFieldAs<DIFile>(2).getDirectory();
      return !dir.empty() ? dir : getContext().getDirectory();
    }
    StringRef getFilename() const {
      StringRef filename = getFieldAs<DIFile>(2).getFilename();
      assert(!filename.empty() && "Why'd you create this then?");
      return filename;
    }
    DILexicalBlock getScope() const { return getFieldAs<DILexicalBlock>(1); }
  };

  /// DINameSpace - A wrapper for a C++ style name space.
  class DINameSpace : public DIScope { 
  public:
    explicit DINameSpace(const MDNode *N = 0) : DIScope(N) {}
    DIScope getContext() const     { return getFieldAs<DIScope>(1);      }
    StringRef getName() const      { return getStringField(2);           }
    StringRef getDirectory() const  { 
      return getFieldAs<DIFile>(3).getDirectory();
    }
    StringRef getFilename() const  { 
      return getFieldAs<DIFile>(3).getFilename();
    }
    DICompileUnit getCompileUnit() const{ 
      assert (getVersion() <= LLVMDebugVersion10 && "Invalid getCompileUnit!");
      if (getVersion() == llvm::LLVMDebugVersion7)
        return getFieldAs<DICompileUnit>(3);

      return getFieldAs<DIFile>(3).getCompileUnit(); 
    }
    unsigned getLineNumber() const { return getUnsignedField(4);         }
    bool Verify() const;
  };

  /// DILocation - This object holds location information. This object
  /// is not associated with any DWARF tag.
  class DILocation : public DIDescriptor {
  public:
    explicit DILocation(const MDNode *N) : DIDescriptor(N) { }

    unsigned getLineNumber() const     { return getUnsignedField(0); }
    unsigned getColumnNumber() const   { return getUnsignedField(1); }
    DIScope  getScope() const          { return getFieldAs<DIScope>(2); }
    DILocation getOrigLocation() const { return getFieldAs<DILocation>(3); }
    StringRef getFilename() const    { return getScope().getFilename(); }
    StringRef getDirectory() const   { return getScope().getDirectory(); }
    bool Verify() const;
  };

  class DIObjCProperty : public DIDescriptor {
    friend class DIDescriptor;
    void printInternal(raw_ostream &OS) const;
  public:
    explicit DIObjCProperty(const MDNode *N) : DIDescriptor(N) { }

    StringRef getObjCPropertyName() const { return getStringField(1); }
    DIFile getFile() const { return getFieldAs<DIFile>(2); }
    unsigned getLineNumber() const { return getUnsignedField(3); }

    StringRef getObjCPropertyGetterName() const {
      return getStringField(4);
    }
    StringRef getObjCPropertySetterName() const {
      return getStringField(5);
    }
    bool isReadOnlyObjCProperty() {
      return (getUnsignedField(6) & dwarf::DW_APPLE_PROPERTY_readonly) != 0;
    }
    bool isReadWriteObjCProperty() {
      return (getUnsignedField(6) & dwarf::DW_APPLE_PROPERTY_readwrite) != 0;
    }
    bool isAssignObjCProperty() {
      return (getUnsignedField(6) & dwarf::DW_APPLE_PROPERTY_assign) != 0;
    }
    bool isRetainObjCProperty() {
      return (getUnsignedField(6) & dwarf::DW_APPLE_PROPERTY_retain) != 0;
    }
    bool isCopyObjCProperty() {
      return (getUnsignedField(6) & dwarf::DW_APPLE_PROPERTY_copy) != 0;
    }
    bool isNonAtomicObjCProperty() {
      return (getUnsignedField(6) & dwarf::DW_APPLE_PROPERTY_nonatomic) != 0;
    }

    DIType getType() const { return getFieldAs<DIType>(7); }

    /// Verify - Verify that a derived type descriptor is well formed.
    bool Verify() const;
  };

  /// getDISubprogram - Find subprogram that is enclosing this scope.
  DISubprogram getDISubprogram(const MDNode *Scope);

  /// getDICompositeType - Find underlying composite type.
  DICompositeType getDICompositeType(DIType T);

  /// isSubprogramContext - Return true if Context is either a subprogram
  /// or another context nested inside a subprogram.
  bool isSubprogramContext(const MDNode *Context);

  /// getOrInsertFnSpecificMDNode - Return a NameMDNode that is suitable
  /// to hold function specific information.
  NamedMDNode *getOrInsertFnSpecificMDNode(Module &M, DISubprogram SP);

  /// getFnSpecificMDNode - Return a NameMDNode, if available, that is 
  /// suitable to hold function specific information.
  NamedMDNode *getFnSpecificMDNode(const Module &M, DISubprogram SP);

  /// createInlinedVariable - Create a new inlined variable based on current
  /// variable.
  /// @param DV            Current Variable.
  /// @param InlinedScope  Location at current variable is inlined.
  DIVariable createInlinedVariable(MDNode *DV, MDNode *InlinedScope,
                                   LLVMContext &VMContext);

  /// cleanseInlinedVariable - Remove inlined scope from the variable.
  DIVariable cleanseInlinedVariable(MDNode *DV, LLVMContext &VMContext);

  class DebugInfoFinder {
  public:
    /// processModule - Process entire module and collect debug info
    /// anchors.
    void processModule(Module &M);

  private:
    /// processType - Process DIType.
    void processType(DIType DT);

    /// processLexicalBlock - Process DILexicalBlock.
    void processLexicalBlock(DILexicalBlock LB);

    /// processSubprogram - Process DISubprogram.
    void processSubprogram(DISubprogram SP);

    /// processDeclare - Process DbgDeclareInst.
    void processDeclare(DbgDeclareInst *DDI);

    /// processLocation - Process DILocation.
    void processLocation(DILocation Loc);

    /// addCompileUnit - Add compile unit into CUs.
    bool addCompileUnit(DICompileUnit CU);

    /// addGlobalVariable - Add global variable into GVs.
    bool addGlobalVariable(DIGlobalVariable DIG);

    // addSubprogram - Add subprogram into SPs.
    bool addSubprogram(DISubprogram SP);

    /// addType - Add type into Tys.
    bool addType(DIType DT);

  public:
    typedef SmallVector<MDNode *, 8>::const_iterator iterator;
    iterator compile_unit_begin()    const { return CUs.begin(); }
    iterator compile_unit_end()      const { return CUs.end(); }
    iterator subprogram_begin()      const { return SPs.begin(); }
    iterator subprogram_end()        const { return SPs.end(); }
    iterator global_variable_begin() const { return GVs.begin(); }
    iterator global_variable_end()   const { return GVs.end(); }
    iterator type_begin()            const { return TYs.begin(); }
    iterator type_end()              const { return TYs.end(); }

    unsigned compile_unit_count()    const { return CUs.size(); }
    unsigned global_variable_count() const { return GVs.size(); }
    unsigned subprogram_count()      const { return SPs.size(); }
    unsigned type_count()            const { return TYs.size(); }

  private:
    SmallVector<MDNode *, 8> CUs;  // Compile Units
    SmallVector<MDNode *, 8> SPs;  // Subprograms
    SmallVector<MDNode *, 8> GVs;  // Global Variables;
    SmallVector<MDNode *, 8> TYs;  // Types
    SmallPtrSet<MDNode *, 64> NodesSeen;
  };
} // end namespace llvm

#endif
