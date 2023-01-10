//===-- LLVMOpBase.td - LLVM IR dialect shared definitions -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains shared definitions for the LLVM IR dialect and its
// subdialects.
//
//===----------------------------------------------------------------------===//

#ifndef LLVMIR_OP_BASE
#define LLVMIR_OP_BASE

include "mlir/Dialect/LLVMIR/LLVMOpsInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// LLVM Dialect.
//===----------------------------------------------------------------------===//

def LLVM_Dialect : Dialect {
  let name = "llvm";
  let cppNamespace = "::mlir::LLVM";

  let useDefaultAttributePrinterParser = 1;
  let hasRegionArgAttrVerify = 1;
  let hasRegionResultAttrVerify = 1;
  let hasOperationAttrVerify = 1;
  let useFoldAPI = kEmitFoldAdaptorFolder;

  let extraClassDeclaration = [{
    /// Name of the data layout attributes.
    static StringRef getDataLayoutAttrName() { return "llvm.data_layout"; }
    static StringRef getAlignAttrName() { return "llvm.align"; }
    static StringRef getNoAliasAttrName() { return "llvm.noalias"; }
    static StringRef getReadonlyAttrName() { return "llvm.readonly"; }
    static StringRef getNoAliasScopesAttrName() { return "noalias_scopes"; }
    static StringRef getAliasScopesAttrName() { return "alias_scopes"; }
    static StringRef getLoopAttrName() { return "llvm.loop"; }
    static StringRef getParallelAccessAttrName() { return "parallel_access"; }
    static StringRef getLoopOptionsAttrName() { return "options"; }
    static StringRef getAccessGroupsAttrName() { return "access_groups"; }
    static StringRef getStructAttrsAttrName() { return "llvm.struct_attrs"; }
    static StringRef getByValAttrName() { return "llvm.byval"; }
    static StringRef getByRefAttrName() { return "llvm.byref"; }
    static StringRef getStructRetAttrName() { return "llvm.sret"; }
    static StringRef getInAllocaAttrName() { return "llvm.inalloca"; }
    static StringRef getNoUndefAttrName() { return "llvm.noundef"; }
    static StringRef getSExtAttrName() { return "llvm.signext"; }
    static StringRef getZExtAttrName() { return "llvm.zeroext"; }
    static StringRef getTBAAAttrName() { return "llvm.tbaa"; }

    /// Verifies if the attribute is a well-formed value for "llvm.struct_attrs"
    static LogicalResult verifyStructAttr(
        Operation *op, Attribute attr, Type annotatedType);

    /// Verifies if the given string is a well-formed data layout descriptor.
    /// Uses `reportError` to report errors.
    static LogicalResult verifyDataLayoutString(
        StringRef descr, llvm::function_ref<void (const Twine &)> reportError);

    /// Name of the target triple attribute.
    static StringRef getTargetTripleAttrName() { return "llvm.target_triple"; }

    /// Name of the C wrapper emission attribute.
    static StringRef getEmitCWrapperAttrName() {
      return "llvm.emit_c_interface";
    }

    /// Returns `true` if the given type is compatible with the LLVM dialect.
    static bool isCompatibleType(Type);

    /// Name of the attribute mapped to LLVM's 'readnone' function attribute.
    /// It is allowed on any FunctionOpInterface operations.
    static StringRef getReadnoneAttrName() {
      return "llvm.readnone";
    }

    Type parseType(DialectAsmParser &p) const override;
    void printType(Type, DialectAsmPrinter &p) const override;

  private:
    /// Register all types.
    void registerTypes();

    /// A cache storing compatible LLVM types that have been verified. This
    /// can save us lots of verification time if there are many occurrences
    /// of some deeply-nested aggregate types in the program.
    ThreadLocalCache<DenseSet<Type>> compatibleTypes;

    /// Register the attributes of this dialect.
    void registerAttributes();
  }];
}

//===----------------------------------------------------------------------===//
// LLVM dialect type constraints.
//===----------------------------------------------------------------------===//

// LLVM dialect type.
def LLVM_Type : DialectType<LLVM_Dialect,
                            CPred<"::mlir::LLVM::isCompatibleOuterType($_self)">,
                            "LLVM dialect-compatible type">;

// Type constraint accepting LLVM token type.
def LLVM_TokenType : Type<
  CPred<"$_self.isa<::mlir::LLVM::LLVMTokenType>()">,
  "LLVM token type">,
  BuildableType<"::mlir::LLVM::LLVMTokenType::get($_builder.getContext())">;

// Type constraint accepting LLVM primitive types, i.e. all types except void
// and function.
def LLVM_PrimitiveType : Type<
  And<[LLVM_Type.predicate,
       CPred<"!$_self.isa<::mlir::LLVM::LLVMVoidType, "
                         "::mlir::LLVM::LLVMFunctionType>()">]>,
  "primitive LLVM type">;

// Type constraint accepting any LLVM function type.
def LLVM_FunctionType : Type<CPred<"$_self.isa<::mlir::LLVM::LLVMFunctionType>()">,
                         "LLVM function type", "::mlir::LLVM::LLVMFunctionType">;

// Type constraint accepting any LLVM floating point type.
def LLVM_AnyFloat : Type<
  CPred<"::mlir::LLVM::isCompatibleFloatingPointType($_self)">,
  "floating point LLVM type">;

// Type constraint accepting any LLVM pointer type.
def LLVM_AnyPointer : Type<CPred<"$_self.isa<::mlir::LLVM::LLVMPointerType>()">,
                          "LLVM pointer type", "::mlir::LLVM::LLVMPointerType">;

// Type constraint accepting LLVM pointer type with an additional constraint
// on the element type.
class LLVM_PointerTo<Type pointee> : Type<
  And<[LLVM_AnyPointer.predicate,
       Or<[CPred<"$_self.cast<::mlir::LLVM::LLVMPointerType>().isOpaque()">,
           SubstLeaves<
             "$_self",
             "$_self.cast<::mlir::LLVM::LLVMPointerType>().getElementType()",
             pointee.predicate>]>]>,
  "LLVM pointer to " # pointee.summary, "::mlir::LLVM::LLVMPointerType">;

// Type constraints accepting LLVM pointer type to integer of a specific width.
class LLVM_IntPtrBase<int width, int addressSpace = 0> : Type<
  LLVM_PointerTo<I<width>>.predicate,
  "LLVM pointer to " # I<width>.summary>,
  BuildableType<"::mlir::LLVM::LLVMPointerType::get("
                "::mlir::IntegerType::get($_builder.getContext(), "
                # width #"), "# addressSpace #")">;

def LLVM_i8Ptr : LLVM_IntPtrBase<8>;

// Type constraint accepting any LLVM structure type.
def LLVM_AnyStruct : Type<CPred<"$_self.isa<::mlir::LLVM::LLVMStructType>()">,
                         "LLVM structure type">;

// Type constraint accepting opaque LLVM structure type.
def LLVM_OpaqueStruct : Type<
  And<[LLVM_AnyStruct.predicate,
       CPred<"$_self.cast<::mlir::LLVM::LLVMStructType>().isOpaque()">]>>;

// Type constraint accepting any LLVM type that can be loaded or stored, i.e. a
// type that has size (not void, function or opaque struct type).
def LLVM_LoadableType : Type<
  Or<[And<[LLVM_PrimitiveType.predicate, Neg<LLVM_OpaqueStruct.predicate>]>,
      LLVM_PointerElementTypeInterface.predicate]>,
  "LLVM type with size">;

// Type constraint accepting any LLVM aggregate type, i.e. structure or array.
def LLVM_AnyAggregate : Type<
  CPred<"$_self.isa<::mlir::LLVM::LLVMStructType, "
                   "::mlir::LLVM::LLVMArrayType>()">,
  "LLVM aggregate type">;

// Type constraint accepting any LLVM non-aggregate type, i.e. not structure or
// array.
def LLVM_AnyNonAggregate : Type<And<[LLVM_Type.predicate,
                                     Neg<LLVM_AnyAggregate.predicate>]>,
                               "LLVM-compatible non-aggregate type">;

// Type constraint accepting any LLVM vector type.
def LLVM_AnyVector : Type<CPred<"::mlir::LLVM::isCompatibleVectorType($_self)">,
                         "LLVM dialect-compatible vector type">;

// Type constraint accepting any LLVM fixed-length vector type.
def LLVM_AnyFixedVector : Type<CPred<
                                "!::mlir::LLVM::isScalableVectorType($_self)">,
                                "LLVM dialect-compatible fixed-length vector type">;

// Type constraint accepting any LLVM scalable vector type.
def LLVM_AnyScalableVector : Type<CPred<
                                "::mlir::LLVM::isScalableVectorType($_self)">,
                                "LLVM dialect-compatible scalable vector type">;

// Type constraint accepting an LLVM vector type with an additional constraint
// on the vector element type.
class LLVM_VectorOf<Type element> : Type<
  And<[LLVM_AnyVector.predicate,
       SubstLeaves<
         "$_self",
         "::mlir::LLVM::getVectorElementType($_self)",
         element.predicate>]>,
  "LLVM dialect-compatible vector of " # element.summary>;

// Type constraint accepting a constrained type, or a vector of such types.
class LLVM_ScalarOrVectorOf<Type element> :
    AnyTypeOf<[element, LLVM_VectorOf<element>]>;

// Base class for LLVM operations. Defines the interface to the llvm::IRBuilder
// used to translate to proper LLVM IR and the interface to the mlir::OpBuilder
// used to import from LLVM IR.
class LLVM_OpBase<Dialect dialect, string mnemonic, list<Trait> traits = []> :
    Op<dialect, mnemonic, traits> {
  // A pattern for constructing the LLVM IR Instruction (or other Value) that
  // corresponds to this op.  This pattern can use `builder` to refer to an
  // `llvm::IRBuilder<>` instance, $-names of arguments and results and the
  // following special variable names:
  //   - $_resultType - substituted with the LLVM IR type of the result;
  //   - $_numOperands - substituted with the number of operands (including
  //                     the variadic ones);
  //   - $_hasResult - substituted with a check that a variadic-result op does
  //                   have a result (LLVM ops can have 0 or 1 result);
  //   - $_location - mlir::Location object of the instruction.
  // Additionally, `$$` can be used to produce the dollar character.
  string llvmBuilder = "";

  // A builder to construct the MLIR LLVM dialect operation given the matching
  // LLVM IR instruction `inst` and its operands `llvmOperands`. The
  // following $-variables exist:
  //   - $name - substituted by the remapped `inst` operand value at the index
  //             of the MLIR operation argument with the given name, or if the
  //             name matches the result name, by a reference to store the
  //             result of the newly created MLIR operation to;
  //   - $_op - substituted by a reference to store the newly created MLIR
  //            operation (only for MLIR operations that return no result);
  //   - $_int_attr - substituted by a call to an integer attribute matcher;
  //   - $_var_attr - substituted by a call to a variable attribute matcher;
  //   - $_resultType - substituted with the MLIR result type;
  //   - $_location - substituted with the MLIR location;
  //   - $_builder - substituted with the MLIR builder;
  //   - $_qualCppClassName - substitiuted with the MLIR operation class name.
  // Always either store a reference to the result of the newly created
  // operation, or to the operation itself if it does not return a result.
  // Additionally, `$$` can be used to produce the dollar character.
  string mlirBuilder = "";

  // An array that specifies a mapping from MLIR argument indices to LLVM IR
  // operand indices. The mapping is necessary since argument and operand
  // indices do not always match. If not defined, the array is set to the
  // identity permutation. An operation may define any custom index permutation
  // and set a specific argument index to -1 if it does not map to an LLVM IR
  // operand.
  list<int> llvmArgIndices = [];
}

//===----------------------------------------------------------------------===//
// Base classes for LLVM dialect operations.
//===----------------------------------------------------------------------===//

// Base class for LLVM operations. All operations get an "llvm." prefix in
// their name automatically. LLVM operations have either zero or one result,
// this class is specialized below for both cases and should not be used
// directly.
class LLVM_Op<string mnemonic, list<Trait> traits = []> :
    LLVM_OpBase<LLVM_Dialect, mnemonic, traits>;

// Case of the LLVM enum attribute backed by I64Attr with customized string
// representation that corresponds to what is visible in the textual IR form.
// The parameters are as follows:
//   - `cppSym`: name of the C++ enumerant for this case in MLIR API;
//   - `irSym`: keyword used in the custom form of MLIR operation;
//   - `llvmSym`: name of the C++ enumerant for this case in LLVM API.
// For example, `LLVM_EnumAttrCase<"Weak", "weak", "WeakAnyLinkage">` is usable
// as `<MlirEnumName>::Weak` in MLIR API, `WeakAnyLinkage` in LLVM API and
// is printed/parsed as `weak` in MLIR custom textual format.
class LLVM_EnumAttrCase<string cppSym, string irSym, string llvmSym, int val> :
    I64EnumAttrCase<cppSym, val, irSym> {

  // The name of the equivalent enumerant in LLVM.
  string llvmEnumerant = llvmSym;
}

// LLVM enum attribute backed by I64Attr with string representation
// corresponding to what is visible in the textual IR form.
// The parameters are as follows:
//   - `name`: name of the C++ enum class in MLIR API;
//   - `llvmName`: name of the C++ enum in LLVM API;
//   - `description`: textual description for documentation purposes;
//   - `cases`: list of enum cases.
// For example, `LLVM_EnumAttr<Linkage, "::llvm::GlobalValue::LinkageTypes`
// produces `mlir::LLVM::Linkage` enum class in MLIR API that corresponds to (a
// subset of) values in the `llvm::GlobalValue::LinkageTypes` in LLVM API.
class LLVM_EnumAttr<string name, string llvmName, string description,
                    list<LLVM_EnumAttrCase> cases> :
    I64EnumAttr<name, description, cases> {

  // The equivalent enum class name in LLVM.
  string llvmClassName = llvmName;
}

// LLVM_CEnumAttr is functionally identical to LLVM_EnumAttr, but to be used for
// non-class enums.
class LLVM_CEnumAttr<string name, string llvmNS, string description,
      list<LLVM_EnumAttrCase> cases> :
    I64EnumAttr<name, description, cases> {
  string llvmClassName = llvmNS;
}

// For every value in the list, substitutes the value in the place of "$0" in
// "pattern" and stores the list of strings as "lst".
class ListIntSubst<string pattern, list<int> values> {
  list<string> lst = !foreach(x, values,
                              !subst("$0", !cast<string>(x), pattern));
}

// Patterns with code obtaining the LLVM IR type of the given operand or result
// of operation. "$0" is expected to be replaced by the position of the operand
// or result in the operation.
def LLVM_IntrPatterns {
  string operand =
    [{moduleTranslation.convertType(opInst.getOperand($0).getType())}];
  string result =
    [{moduleTranslation.convertType(opInst.getResult($0).getType())}];
  string structResult =
    [{moduleTranslation.convertType(
        opInst.getResult(0).getType().cast<LLVM::LLVMStructType>()
              .getBody()[$0])}];
}


// Base class for LLVM intrinsics operation. It is similar to LLVM_Op, but
// provides the "llvmBuilder" field for constructing the intrinsic.
// The builder relies on the contents of "overloadedResults" and
// "overloadedOperands" lists that contain the positions of intrinsic results
// and operands that are overloadable in the LLVM sense, that is their types
// must be passed in during the construction of the intrinsic declaration to
// differentiate between differently-typed versions of the intrinsic.
// If the intrinsic has multiple results, this will eventually be packed into a
// single struct result. In this case, the types of any overloaded results need
// to be accessed via the LLVMStructType, instead of directly via the result.
// "opName" contains the name of the operation to be associated with the
// intrinsic and "enumName" contains the name of the intrinsic as appears in
// `llvm::Intrinsic` enum; one usually wants these to be related. Additionally,
// the base class also defines the "mlirBuilder" field to support the inverse
// translation starting from an LLVM IR intrinsic.
class LLVM_IntrOpBase<Dialect dialect, string opName, string enumName,
                      list<int> overloadedResults, list<int> overloadedOperands,
                      list<Trait> traits, int numResults,
                      bit requiresAccessGroup = 0, bit requiresAliasScope = 0,
                      bit requiresFastmath = 0>
    : LLVM_OpBase<dialect, opName, !listconcat(
        !if(!gt(requiresFastmath, 0),
            [DeclareOpInterfaceMethods<FastmathFlagsInterface>],
            []),
        traits)>,
      Results<!if(!gt(numResults, 0), (outs LLVM_Type:$res), (outs))> {
  string resultPattern = !if(!gt(numResults, 1),
                             LLVM_IntrPatterns.structResult,
                             LLVM_IntrPatterns.result);
  string llvmEnumName = enumName;
  let llvmBuilder = [{
    llvm::Module *module = builder.GetInsertBlock()->getModule();
    llvm::Function *fn = llvm::Intrinsic::getDeclaration(
        module,
        llvm::Intrinsic::}] # enumName # [{,
        { }] # !interleave(!listconcat(
            ListIntSubst<resultPattern, overloadedResults>.lst,
            ListIntSubst<LLVM_IntrPatterns.operand,
                         overloadedOperands>.lst), ", ") # [{
        });
    auto operands = moduleTranslation.lookupValues(opInst.getOperands());
    }] # [{auto *inst = builder.CreateCall(fn, operands);
    }] # !if(!gt(requiresAccessGroup, 0),
      "moduleTranslation.setAccessGroupsMetadata(op, inst);",
      "(void) inst;")
    # !if(!gt(requiresAliasScope, 0),
      "moduleTranslation.setAliasScopeMetadata(op, inst);",
      "(void) inst;")
    # !if(!gt(numResults, 0), "$res = inst;", "");

  string mlirBuilder = [{
    FailureOr<SmallVector<Value>> mlirOperands =
      moduleImport.convertValues(llvmOperands);
    if (failed(mlirOperands))
      return failure();
    SmallVector<Type> resultTypes =
    }] # !if(!gt(numResults, 0), "{$_resultType};", "{};") # [{
    auto op = $_builder.create<$_qualCppClassName>(
      $_location, resultTypes, *mlirOperands);
    }] # !if(!gt(requiresFastmath, 0),
      "moduleImport.setFastmathFlagsAttr(inst, op);", "")
    # !if(!gt(numResults, 0), "$res = op;", "$_op = op;");
}

// Base class for LLVM intrinsic operations, should not be used directly. Places
// the intrinsic into the LLVM dialect and prefixes its name with "intr.".
class LLVM_IntrOp<string mnem, list<int> overloadedResults,
                  list<int> overloadedOperands, list<Trait> traits,
                  int numResults, bit requiresAccessGroup = 0,
                  bit requiresAliasScope = 0, bit requiresFastmath = 0>
    : LLVM_IntrOpBase<LLVM_Dialect, "intr." # mnem, !subst(".", "_", mnem),
                      overloadedResults, overloadedOperands, traits,
                      numResults, requiresAccessGroup, requiresAliasScope,
                      requiresFastmath>;

// Base class for LLVM intrinsic operations returning no results. Places the
// intrinsic into the LLVM dialect and prefixes its name with "intr.".
//
// Sample use: derive an entry from this class and populate the fields.
//
//    def LLVM_Name : LLVM_ZeroResultIntrOp<"name", [0], [Pure]>,
//                    Arguments<(ins LLVM_Type, LLVM_Type)>;
//
// The mnemonic will be prefixed with "llvm.intr.", where the "llvm." part comes
// from the LLVM dialect. The overloadedOperands list contains the indices of
// the operands the type of which will be passed in the LLVM IR intrinsic
// builder. In the example above, the Op has two arguments, but only the first
// one (as indicated by `[0]`) is necessary to resolve the overloaded intrinsic.
// The Op has no results.
class LLVM_ZeroResultIntrOp<string mnem, list<int> overloadedOperands = [],
                            list<Trait> traits = []>
    : LLVM_IntrOp<mnem, [], overloadedOperands, traits, 0>;

// Base class for LLVM intrinsic operations returning one result. Places the
// intrinsic into the LLVM dialect and prefixes its name with "intr.". This is
// similar to LLVM_ZeroResultIntrOp but allows one to define Ops returning one
// result, called "res". Additionally, the overloadedResults list should contain
// "0" if the result must be used to resolve overloaded intrinsics, or remain
// empty otherwise.
class LLVM_OneResultIntrOp<string mnem, list<int> overloadedResults = [],
                           list<int> overloadedOperands = [],
                           list<Trait> traits = [],
                           bit requiresFastmath = 0>
    : LLVM_IntrOp<mnem, overloadedResults, overloadedOperands, traits, 1,
                  /*requiresAccessGroup=*/0, /*requiresAliasScope=*/0,
                  requiresFastmath>;

def LLVM_OneResultOpBuilder :
  OpBuilder<(ins "Type":$resultType, "ValueRange":$operands,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    if (resultType) $_state.addTypes(resultType);
    $_state.addOperands(operands);
    for (auto namedAttr : attributes)
      $_state.addAttribute(namedAttr.getName(), namedAttr.getValue());
  }]>;

def LLVM_ZeroResultOpBuilder :
  OpBuilder<(ins "ValueRange":$operands,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addOperands(operands);
    for (auto namedAttr : attributes)
      $_state.addAttribute(namedAttr.getName(), namedAttr.getValue());
  }]>;

// Compatibility builder that takes an instance of wrapped llvm::VoidType
// to indicate no result.
def LLVM_VoidResultTypeOpBuilder :
  OpBuilder<(ins "Type":$resultType, "ValueRange":$operands,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    assert(isCompatibleType(resultType) && "result must be an LLVM type");
    assert(resultType.isa<LLVMVoidType>() &&
           "for zero-result operands, only 'void' is accepted as result type");
    build($_builder, $_state, operands, attributes);
  }]>;


// Opaque builder used for terminator operations that contain successors.
def LLVM_TerminatorPassthroughOpBuilder :
  OpBuilder<(ins "ValueRange":$operands, "SuccessorRange":$destinations,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attributes),
  [{
    $_state.addOperands(operands);
    $_state.addSuccessors(destinations);
    $_state.addAttributes(attributes);
  }]>;


#endif  // LLVMIR_OP_BASE
