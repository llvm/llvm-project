//===--- llvm/DIBuilder.h - Debug Information Builder -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a DIBuilder that is useful for creating debugging 
// information entries in LLVM IR form.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_DIBUILDER_H
#define LLVM_ANALYSIS_DIBUILDER_H

#include "llvm/Support/DataTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
  class BasicBlock;
  class Instruction;
  class Function;
  class Module;
  class Value;
  class LLVMContext;
  class MDNode;
  class StringRef;
  class DIDescriptor;
  class DIFile;
  class DIEnumerator;
  class DIType;
  class DIArray;
  class DIGlobalVariable;
  class DINameSpace;
  class DIVariable;
  class DISubrange;
  class DILexicalBlockFile;
  class DILexicalBlock;
  class DISubprogram;
  class DITemplateTypeParameter;
  class DITemplateValueParameter;
  class DIObjCProperty;

  class DIBuilder {
    private:
    Module &M;
    LLVMContext & VMContext;
    MDNode *TheCU;

    MDNode *TempEnumTypes;
    MDNode *TempRetainTypes;
    MDNode *TempSubprograms;
    MDNode *TempGVs;

    Function *DeclareFn;     // llvm.dbg.declare
    Function *ValueFn;       // llvm.dbg.value

    SmallVector<Value *, 4> AllEnumTypes;
    SmallVector<Value *, 4> AllRetainTypes;
    SmallVector<Value *, 4> AllSubprograms;
    SmallVector<Value *, 4> AllGVs;

    DIBuilder(const DIBuilder &) LLVM_DELETED_FUNCTION;
    void operator=(const DIBuilder &) LLVM_DELETED_FUNCTION;

    public:
    explicit DIBuilder(Module &M);
    const MDNode *getCU() { return TheCU; }
    enum ComplexAddrKind { OpPlus=1, OpDeref };

    /// finalize - Construct any deferred debug info descriptors.
    void finalize();

    /// createCompileUnit - A CompileUnit provides an anchor for all debugging
    /// information generated during this instance of compilation.
    /// @param Lang     Source programming language, eg. dwarf::DW_LANG_C99
    /// @param File     File name
    /// @param Dir      Directory
    /// @param Producer String identify producer of debugging information. 
    ///                 Usuall this is a compiler version string.
    /// @param isOptimized A boolean flag which indicates whether optimization
    ///                    is ON or not.
    /// @param Flags    This string lists command line options. This string is 
    ///                 directly embedded in debug info output which may be used
    ///                 by a tool analyzing generated debugging information.
    /// @param RV       This indicates runtime version for languages like 
    ///                 Objective-C.
    void createCompileUnit(unsigned Lang, StringRef File, StringRef Dir, 
                           StringRef Producer,
                           bool isOptimized, StringRef Flags, unsigned RV);

    /// createFile - Create a file descriptor to hold debugging information
    /// for a file.
    DIFile createFile(StringRef Filename, StringRef Directory);
                           
    /// createEnumerator - Create a single enumerator value.
    DIEnumerator createEnumerator(StringRef Name, uint64_t Val);

    /// createNullPtrType - Create C++0x nullptr type.
    DIType createNullPtrType(StringRef Name);

    /// createBasicType - Create debugging information entry for a basic 
    /// type.
    /// @param Name        Type name.
    /// @param SizeInBits  Size of the type.
    /// @param AlignInBits Type alignment.
    /// @param Encoding    DWARF encoding code, e.g. dwarf::DW_ATE_float.
    DIType createBasicType(StringRef Name, uint64_t SizeInBits, 
                           uint64_t AlignInBits, unsigned Encoding);

    /// createQualifiedType - Create debugging information entry for a qualified
    /// type, e.g. 'const int'.
    /// @param Tag         Tag identifing type, e.g. dwarf::TAG_volatile_type
    /// @param FromTy      Base Type.
    DIType createQualifiedType(unsigned Tag, DIType FromTy);

    /// createPointerType - Create debugging information entry for a pointer.
    /// @param PointeeTy   Type pointed by this pointer.
    /// @param SizeInBits  Size.
    /// @param AlignInBits Alignment. (optional)
    /// @param Name        Pointer type name. (optional)
    DIType createPointerType(DIType PointeeTy, uint64_t SizeInBits,
                             uint64_t AlignInBits = 0, 
                             StringRef Name = StringRef());

    /// createReferenceType - Create debugging information entry for a c++
    /// style reference or rvalue reference type.
    DIType createReferenceType(unsigned Tag, DIType RTy);

    /// createTypedef - Create debugging information entry for a typedef.
    /// @param Ty          Original type.
    /// @param Name        Typedef name.
    /// @param File        File where this type is defined.
    /// @param LineNo      Line number.
    /// @param Context     The surrounding context for the typedef.
    DIType createTypedef(DIType Ty, StringRef Name, DIFile File, 
                         unsigned LineNo, DIDescriptor Context);

    /// createFriend - Create debugging information entry for a 'friend'.
    DIType createFriend(DIType Ty, DIType FriendTy);

    /// createInheritance - Create debugging information entry to establish
    /// inheritance relationship between two types.
    /// @param Ty           Original type.
    /// @param BaseTy       Base type. Ty is inherits from base.
    /// @param BaseOffset   Base offset.
    /// @param Flags        Flags to describe inheritance attribute, 
    ///                     e.g. private
    DIType createInheritance(DIType Ty, DIType BaseTy, uint64_t BaseOffset,
                             unsigned Flags);

    /// createMemberType - Create debugging information entry for a member.
    /// @param Scope        Member scope.
    /// @param Name         Member name.
    /// @param File         File where this member is defined.
    /// @param LineNo       Line number.
    /// @param SizeInBits   Member size.
    /// @param AlignInBits  Member alignment.
    /// @param OffsetInBits Member offset.
    /// @param Flags        Flags to encode member attribute, e.g. private
    /// @param Ty           Parent type.
    DIType createMemberType(DIDescriptor Scope, StringRef Name, DIFile File,
                            unsigned LineNo, uint64_t SizeInBits, 
                            uint64_t AlignInBits, uint64_t OffsetInBits, 
                            unsigned Flags, DIType Ty);

    /// createObjCIVar - Create debugging information entry for Objective-C
    /// instance variable.
    /// @param Name         Member name.
    /// @param File         File where this member is defined.
    /// @param LineNo       Line number.
    /// @param SizeInBits   Member size.
    /// @param AlignInBits  Member alignment.
    /// @param OffsetInBits Member offset.
    /// @param Flags        Flags to encode member attribute, e.g. private
    /// @param Ty           Parent type.
    /// @param PropertyName Name of the Objective C property associated with
    ///                     this ivar.
    /// @param PropertyGetterName Name of the Objective C property getter
    ///                           selector.
    /// @param PropertySetterName Name of the Objective C property setter
    ///                           selector.
    /// @param PropertyAttributes Objective C property attributes.
    DIType createObjCIVar(StringRef Name, DIFile File,
                          unsigned LineNo, uint64_t SizeInBits, 
                          uint64_t AlignInBits, uint64_t OffsetInBits, 
                          unsigned Flags, DIType Ty,
                          StringRef PropertyName = StringRef(),
                          StringRef PropertyGetterName = StringRef(),
                          StringRef PropertySetterName = StringRef(),
                          unsigned PropertyAttributes = 0);

    /// createObjCIVar - Create debugging information entry for Objective-C
    /// instance variable.
    /// @param Name         Member name.
    /// @param File         File where this member is defined.
    /// @param LineNo       Line number.
    /// @param SizeInBits   Member size.
    /// @param AlignInBits  Member alignment.
    /// @param OffsetInBits Member offset.
    /// @param Flags        Flags to encode member attribute, e.g. private
    /// @param Ty           Parent type.
    /// @param PropertyNode Property associated with this ivar.
    DIType createObjCIVar(StringRef Name, DIFile File,
                          unsigned LineNo, uint64_t SizeInBits, 
                          uint64_t AlignInBits, uint64_t OffsetInBits, 
                          unsigned Flags, DIType Ty,
                          MDNode *PropertyNode);

    /// createObjCProperty - Create debugging information entry for Objective-C
    /// property.
    /// @param Name         Property name.
    /// @param File         File where this property is defined.
    /// @param LineNumber   Line number.
    /// @param GetterName   Name of the Objective C property getter selector.
    /// @param SetterName   Name of the Objective C property setter selector.
    /// @param PropertyAttributes Objective C property attributes.
    /// @param Ty           Type.
    DIObjCProperty createObjCProperty(StringRef Name,
                                      DIFile File, unsigned LineNumber,
                                      StringRef GetterName,
                                      StringRef SetterName,
                                      unsigned PropertyAttributes,
                                      DIType Ty);
      
    /// createClassType - Create debugging information entry for a class.
    /// @param Scope        Scope in which this class is defined.
    /// @param Name         class name.
    /// @param File         File where this member is defined.
    /// @param LineNumber   Line number.
    /// @param SizeInBits   Member size.
    /// @param AlignInBits  Member alignment.
    /// @param OffsetInBits Member offset.
    /// @param Flags        Flags to encode member attribute, e.g. private
    /// @param Elements     class members.
    /// @param VTableHolder Debug info of the base class that contains vtable
    ///                     for this type. This is used in 
    ///                     DW_AT_containing_type. See DWARF documentation
    ///                     for more info.
    /// @param TemplateParms Template type parameters.
    DIType createClassType(DIDescriptor Scope, StringRef Name, DIFile File,
                           unsigned LineNumber, uint64_t SizeInBits,
                           uint64_t AlignInBits, uint64_t OffsetInBits,
                           unsigned Flags, DIType DerivedFrom, 
                           DIArray Elements, MDNode *VTableHolder = 0,
                           MDNode *TemplateParms = 0);

    /// createStructType - Create debugging information entry for a struct.
    /// @param Scope        Scope in which this struct is defined.
    /// @param Name         Struct name.
    /// @param File         File where this member is defined.
    /// @param LineNumber   Line number.
    /// @param SizeInBits   Member size.
    /// @param AlignInBits  Member alignment.
    /// @param Flags        Flags to encode member attribute, e.g. private
    /// @param Elements     Struct elements.
    /// @param RunTimeLang  Optional parameter, Objective-C runtime version.
    DIType createStructType(DIDescriptor Scope, StringRef Name, DIFile File,
                            unsigned LineNumber, uint64_t SizeInBits,
                            uint64_t AlignInBits, unsigned Flags,
                            DIArray Elements, unsigned RunTimeLang = 0);

    /// createUnionType - Create debugging information entry for an union.
    /// @param Scope        Scope in which this union is defined.
    /// @param Name         Union name.
    /// @param File         File where this member is defined.
    /// @param LineNumber   Line number.
    /// @param SizeInBits   Member size.
    /// @param AlignInBits  Member alignment.
    /// @param Flags        Flags to encode member attribute, e.g. private
    /// @param Elements     Union elements.
    /// @param RunTimeLang  Optional parameter, Objective-C runtime version.
    DIType createUnionType(DIDescriptor Scope, StringRef Name, DIFile File,
                           unsigned LineNumber, uint64_t SizeInBits,
                           uint64_t AlignInBits, unsigned Flags,
                           DIArray Elements, unsigned RunTimeLang = 0);

    /// createTemplateTypeParameter - Create debugging information for template
    /// type parameter.
    /// @param Scope        Scope in which this type is defined.
    /// @param Name         Type parameter name.
    /// @param Ty           Parameter type.
    /// @param File         File where this type parameter is defined.
    /// @param LineNo       Line number.
    /// @param ColumnNo     Column Number.
    DITemplateTypeParameter
    createTemplateTypeParameter(DIDescriptor Scope, StringRef Name, DIType Ty,
                                MDNode *File = 0, unsigned LineNo = 0,
                                unsigned ColumnNo = 0);

    /// createTemplateValueParameter - Create debugging information for template
    /// value parameter.
    /// @param Scope        Scope in which this type is defined.
    /// @param Name         Value parameter name.
    /// @param Ty           Parameter type.
    /// @param Value        Constant parameter value.
    /// @param File         File where this type parameter is defined.
    /// @param LineNo       Line number.
    /// @param ColumnNo     Column Number.
    DITemplateValueParameter
    createTemplateValueParameter(DIDescriptor Scope, StringRef Name, DIType Ty,
                                 uint64_t Value,
                                 MDNode *File = 0, unsigned LineNo = 0,
                                 unsigned ColumnNo = 0);

    /// createArrayType - Create debugging information entry for an array.
    /// @param Size         Array size.
    /// @param AlignInBits  Alignment.
    /// @param Ty           Element type.
    /// @param Subscripts   Subscripts.
    DIType createArrayType(uint64_t Size, uint64_t AlignInBits, 
                           DIType Ty, DIArray Subscripts);

    /// createVectorType - Create debugging information entry for a vector type.
    /// @param Size         Array size.
    /// @param AlignInBits  Alignment.
    /// @param Ty           Element type.
    /// @param Subscripts   Subscripts.
    DIType createVectorType(uint64_t Size, uint64_t AlignInBits, 
                            DIType Ty, DIArray Subscripts);

    /// createEnumerationType - Create debugging information entry for an 
    /// enumeration.
    /// @param Scope        Scope in which this enumeration is defined.
    /// @param Name         Union name.
    /// @param File         File where this member is defined.
    /// @param LineNumber   Line number.
    /// @param SizeInBits   Member size.
    /// @param AlignInBits  Member alignment.
    /// @param Elements     Enumeration elements.
    DIType createEnumerationType(DIDescriptor Scope, StringRef Name, 
                                 DIFile File, unsigned LineNumber, 
                                 uint64_t SizeInBits, uint64_t AlignInBits,
                                 DIArray Elements, DIType ClassType);

    /// createSubroutineType - Create subroutine type.
    /// @param File           File in which this subroutine is defined.
    /// @param ParameterTypes An array of subroutine parameter types. This
    ///                       includes return type at 0th index.
    DIType createSubroutineType(DIFile File, DIArray ParameterTypes);

    /// createArtificialType - Create a new DIType with "artificial" flag set.
    DIType createArtificialType(DIType Ty);

    /// createObjectPointerType - Create a new DIType with the "object pointer"
    /// flag set.
    DIType createObjectPointerType(DIType Ty);

    /// createTemporaryType - Create a temporary forward-declared type.
    DIType createTemporaryType();
    DIType createTemporaryType(DIFile F);

    /// createForwardDecl - Create a temporary forward-declared type.
    DIType createForwardDecl(unsigned Tag, StringRef Name, DIDescriptor Scope,
                             DIFile F, unsigned Line, unsigned RuntimeLang = 0,
                             uint64_t SizeInBits = 0, uint64_t AlignInBits = 0);

    /// retainType - Retain DIType in a module even if it is not referenced 
    /// through debug info anchors.
    void retainType(DIType T);

    /// createUnspecifiedParameter - Create unspeicified type descriptor
    /// for a subroutine type.
    DIDescriptor createUnspecifiedParameter();

    /// getOrCreateArray - Get a DIArray, create one if required.
    DIArray getOrCreateArray(ArrayRef<Value *> Elements);

    /// getOrCreateSubrange - Create a descriptor for a value range.  This
    /// implicitly uniques the values returned.
    DISubrange getOrCreateSubrange(int64_t Lo, int64_t Hi);

    /// createGlobalVariable - Create a new descriptor for the specified global.
    /// @param Name        Name of the variable.
    /// @param File        File where this variable is defined.
    /// @param LineNo      Line number.
    /// @param Ty          Variable Type.
    /// @param isLocalToUnit Boolean flag indicate whether this variable is
    ///                      externally visible or not.
    /// @param Val         llvm::Value of the variable.
    DIGlobalVariable
    createGlobalVariable(StringRef Name, DIFile File, unsigned LineNo,
                         DIType Ty, bool isLocalToUnit, llvm::Value *Val);


    /// createStaticVariable - Create a new descriptor for the specified 
    /// variable.
    /// @param Context     Variable scope.
    /// @param Name        Name of the variable.
    /// @param LinkageName Mangled  name of the variable.
    /// @param File        File where this variable is defined.
    /// @param LineNo      Line number.
    /// @param Ty          Variable Type.
    /// @param isLocalToUnit Boolean flag indicate whether this variable is
    ///                      externally visible or not.
    /// @param Val         llvm::Value of the variable.
    DIGlobalVariable
    createStaticVariable(DIDescriptor Context, StringRef Name, 
                         StringRef LinkageName, DIFile File, unsigned LineNo, 
                         DIType Ty, bool isLocalToUnit, llvm::Value *Val);


    /// createLocalVariable - Create a new descriptor for the specified 
    /// local variable.
    /// @param Tag         Dwarf TAG. Usually DW_TAG_auto_variable or
    ///                    DW_TAG_arg_variable.
    /// @param Scope       Variable scope.
    /// @param Name        Variable name.
    /// @param File        File where this variable is defined.
    /// @param LineNo      Line number.
    /// @param Ty          Variable Type
    /// @param AlwaysPreserve Boolean. Set to true if debug info for this
    ///                       variable should be preserved in optimized build.
    /// @param Flags          Flags, e.g. artificial variable.
    /// @param ArgNo       If this variable is an arugment then this argument's
    ///                    number. 1 indicates 1st argument.
    DIVariable createLocalVariable(unsigned Tag, DIDescriptor Scope,
                                   StringRef Name,
                                   DIFile File, unsigned LineNo,
                                   DIType Ty, bool AlwaysPreserve = false,
                                   unsigned Flags = 0,
                                   unsigned ArgNo = 0);


    /// createComplexVariable - Create a new descriptor for the specified
    /// variable which has a complex address expression for its address.
    /// @param Tag         Dwarf TAG. Usually DW_TAG_auto_variable or
    ///                    DW_TAG_arg_variable.
    /// @param Scope       Variable scope.
    /// @param Name        Variable name.
    /// @param F           File where this variable is defined.
    /// @param LineNo      Line number.
    /// @param Ty          Variable Type
    /// @param Addr        An array of complex address operations.
    /// @param ArgNo       If this variable is an arugment then this argument's
    ///                    number. 1 indicates 1st argument.
    DIVariable createComplexVariable(unsigned Tag, DIDescriptor Scope,
                                     StringRef Name, DIFile F, unsigned LineNo,
                                     DIType Ty, ArrayRef<Value *> Addr,
                                     unsigned ArgNo = 0);

    /// createFunction - Create a new descriptor for the specified subprogram.
    /// See comments in DISubprogram for descriptions of these fields.
    /// @param Scope         Function scope.
    /// @param Name          Function name.
    /// @param LinkageName   Mangled function name.
    /// @param File          File where this variable is defined.
    /// @param LineNo        Line number.
    /// @param Ty            Function type.
    /// @param isLocalToUnit True if this function is not externally visible..
    /// @param isDefinition  True if this is a function definition.
    /// @param ScopeLine     Set to the beginning of the scope this starts
    /// @param Flags         e.g. is this function prototyped or not.
    ///                      This flags are used to emit dwarf attributes.
    /// @param isOptimized   True if optimization is ON.
    /// @param Fn            llvm::Function pointer.
    /// @param TParam        Function template parameters.
    DISubprogram createFunction(DIDescriptor Scope, StringRef Name,
                                StringRef LinkageName,
                                DIFile File, unsigned LineNo,
                                DIType Ty, bool isLocalToUnit,
                                bool isDefinition,
                                unsigned ScopeLine,
                                unsigned Flags = 0,
                                bool isOptimized = false,
                                Function *Fn = 0,
                                MDNode *TParam = 0,
                                MDNode *Decl = 0);

    /// createMethod - Create a new descriptor for the specified C++ method.
    /// See comments in DISubprogram for descriptions of these fields.
    /// @param Scope         Function scope.
    /// @param Name          Function name.
    /// @param LinkageName   Mangled function name.
    /// @param File          File where this variable is defined.
    /// @param LineNo        Line number.
    /// @param Ty            Function type.
    /// @param isLocalToUnit True if this function is not externally visible..
    /// @param isDefinition  True if this is a function definition.
    /// @param Virtuality    Attributes describing virtualness. e.g. pure 
    ///                      virtual function.
    /// @param VTableIndex   Index no of this method in virtual table.
    /// @param VTableHolder  Type that holds vtable.
    /// @param Flags         e.g. is this function prototyped or not.
    ///                      This flags are used to emit dwarf attributes.
    /// @param isOptimized   True if optimization is ON.
    /// @param Fn            llvm::Function pointer.
    /// @param TParam        Function template parameters.
    DISubprogram createMethod(DIDescriptor Scope, StringRef Name,
                              StringRef LinkageName,
                              DIFile File, unsigned LineNo,
                              DIType Ty, bool isLocalToUnit,
                              bool isDefinition,
                              unsigned Virtuality = 0, unsigned VTableIndex = 0,
                              MDNode *VTableHolder = 0,
                              unsigned Flags = 0,
                              bool isOptimized = false,
                              Function *Fn = 0,
                              MDNode *TParam = 0);

    /// createNameSpace - This creates new descriptor for a namespace
    /// with the specified parent scope.
    /// @param Scope       Namespace scope
    /// @param Name        Name of this namespace
    /// @param File        Source file
    /// @param LineNo      Line number
    DINameSpace createNameSpace(DIDescriptor Scope, StringRef Name,
                                DIFile File, unsigned LineNo);


    /// createLexicalBlockFile - This creates a descriptor for a lexical
    /// block with a new file attached. This merely extends the existing
    /// lexical block as it crosses a file.
    /// @param Scope       Lexical block.
    /// @param File        Source file.
    DILexicalBlockFile createLexicalBlockFile(DIDescriptor Scope,
                                              DIFile File);
    
    /// createLexicalBlock - This creates a descriptor for a lexical block
    /// with the specified parent context.
    /// @param Scope       Parent lexical scope.
    /// @param File        Source file
    /// @param Line        Line number
    /// @param Col         Column number
    DILexicalBlock createLexicalBlock(DIDescriptor Scope, DIFile File,
                                      unsigned Line, unsigned Col);

    /// insertDeclare - Insert a new llvm.dbg.declare intrinsic call.
    /// @param Storage     llvm::Value of the variable
    /// @param VarInfo     Variable's debug info descriptor.
    /// @param InsertAtEnd Location for the new intrinsic.
    Instruction *insertDeclare(llvm::Value *Storage, DIVariable VarInfo,
                               BasicBlock *InsertAtEnd);

    /// insertDeclare - Insert a new llvm.dbg.declare intrinsic call.
    /// @param Storage      llvm::Value of the variable
    /// @param VarInfo      Variable's debug info descriptor.
    /// @param InsertBefore Location for the new intrinsic.
    Instruction *insertDeclare(llvm::Value *Storage, DIVariable VarInfo,
                               Instruction *InsertBefore);


    /// insertDbgValueIntrinsic - Insert a new llvm.dbg.value intrinsic call.
    /// @param Val          llvm::Value of the variable
    /// @param Offset       Offset
    /// @param VarInfo      Variable's debug info descriptor.
    /// @param InsertAtEnd Location for the new intrinsic.
    Instruction *insertDbgValueIntrinsic(llvm::Value *Val, uint64_t Offset,
                                         DIVariable VarInfo, 
                                         BasicBlock *InsertAtEnd);
    
    /// insertDbgValueIntrinsic - Insert a new llvm.dbg.value intrinsic call.
    /// @param Val          llvm::Value of the variable
    /// @param Offset       Offset
    /// @param VarInfo      Variable's debug info descriptor.
    /// @param InsertBefore Location for the new intrinsic.
    Instruction *insertDbgValueIntrinsic(llvm::Value *Val, uint64_t Offset,
                                         DIVariable VarInfo, 
                                         Instruction *InsertBefore);

  };
} // end namespace llvm

#endif
