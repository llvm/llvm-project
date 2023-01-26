//===- FunctionInterfaces.td - Function interfaces --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for interfaces that support the definition of
// "function-like" operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_FUNCTIONINTERFACES_TD_
#define MLIR_IR_FUNCTIONINTERFACES_TD_

include "mlir/IR/SymbolInterfaces.td"

//===----------------------------------------------------------------------===//
// FunctionOpInterface
//===----------------------------------------------------------------------===//

def FunctionOpInterface : OpInterface<"FunctionOpInterface", [Symbol]> {
  let cppNamespace = "::mlir";
  let description = [{
    This interfaces provides support for interacting with operations that
    behave like functions. In particular, these operations:

      - must be symbols, i.e. have the `Symbol` trait.
      - must have a single region, that may be comprised with multiple blocks,
        that corresponds to the function body.
        * when this region is empty, the operation corresponds to an external
          function.
        * leading arguments of the first block of the region are treated as
          function arguments.

    The function, aside from implementing the various interface methods,
    should have the following ODS arguments:

      - `function_type` (required)
        * A TypeAttr that holds the signature type of the function.

      - `arg_attrs` (optional)
        * An ArrayAttr of DictionaryAttr that contains attribute dictionaries
          for each of the function arguments.

      - `res_attrs` (optional)
        * An ArrayAttr of DictionaryAttr that contains attribute dictionaries
          for each of the function results.
  }];
  let methods = [
    InterfaceMethod<[{
      Returns the type of the function.
    }],
    "::mlir::Type", "getFunctionType">,
    InterfaceMethod<[{
      Set the type of the function. This method should perform an unsafe
      modification to the function type; it should not update argument or
      result attributes.
    }],
    "void", "setFunctionTypeAttr", (ins "::mlir::TypeAttr":$type)>,

    InterfaceMethod<[{
      Get the array of argument attribute dictionaries. The method should return
      an array attribute containing only dictionary attributes equal in number
      to the number of function arguments. Alternatively, the method can return
      null to indicate that the function has no argument attributes.
    }],
    "::mlir::ArrayAttr", "getArgAttrsAttr">,
    InterfaceMethod<[{
      Get the array of result attribute dictionaries. The method should return
      an array attribute containing only dictionary attributes equal in number
      to the number of function results. Alternatively, the method can return
      null to indicate that the function has no result attributes.
    }],
    "::mlir::ArrayAttr", "getResAttrsAttr">,
    InterfaceMethod<[{
      Set the array of argument attribute dictionaries.
    }],
    "void", "setArgAttrsAttr", (ins "::mlir::ArrayAttr":$attrs)>,
    InterfaceMethod<[{
      Set the array of result attribute dictionaries.
    }],
    "void", "setResAttrsAttr", (ins "::mlir::ArrayAttr":$attrs)>,
    InterfaceMethod<[{
      Remove the array of argument attribute dictionaries. This is the same as
      setting all argument attributes to an empty dictionary. The method should
      return the removed attribute.
    }],
    "::mlir::Attribute", "removeArgAttrsAttr">,
    InterfaceMethod<[{
      Remove the array of result attribute dictionaries. This is the same as
      setting all result attributes to an empty dictionary. The method should
      return the removed attribute.
    }],
    "::mlir::Attribute", "removeResAttrsAttr">,

    InterfaceMethod<[{
      Returns the function argument types based exclusively on
      the type (to allow for this method may be called on function
      declarations).
    }],
    "::llvm::ArrayRef<::mlir::Type>", "getArgumentTypes">,
    InterfaceMethod<[{
      Returns the function result types based exclusively on
      the type (to allow for this method may be called on function
      declarations).
    }],
    "::llvm::ArrayRef<::mlir::Type>", "getResultTypes">,
    InterfaceMethod<[{
      Returns a clone of the function type with the given argument and
      result types.

      Note: The default implementation assumes the function type has
            an appropriate clone method:
              `Type clone(ArrayRef<Type> inputs, ArrayRef<Type> results)`
    }],
    "::mlir::Type", "cloneTypeWith", (ins
      "::mlir::TypeRange":$inputs, "::mlir::TypeRange":$results
    ), /*methodBody=*/[{}], /*defaultImplementation=*/[{
      return $_op.getFunctionType().clone(inputs, results);
    }]>,

    InterfaceMethod<[{
      Verify the contents of the body of this function.

      Note: The default implementation merely checks that if the entry block
      exists, it has the same number and type of arguments as the function type.
    }],
    "::mlir::LogicalResult", "verifyBody", (ins),
    /*methodBody=*/[{}], /*defaultImplementation=*/[{
      if ($_op.isExternal())
        return success();
      ArrayRef<Type> fnInputTypes = $_op.getArgumentTypes();
      // NOTE: This should just be $_op.front() but access generically
      // because the interface methods defined here may be shadowed in
      // arbitrary ways. https://github.com/llvm/llvm-project/issues/54807
      Block &entryBlock = $_op->getRegion(0).front();

      unsigned numArguments = fnInputTypes.size();
      if (entryBlock.getNumArguments() != numArguments)
        return $_op.emitOpError("entry block must have ")
              << numArguments << " arguments to match function signature";

      for (unsigned i = 0, e = fnInputTypes.size(); i != e; ++i) {
        Type argType = entryBlock.getArgument(i).getType();
        if (fnInputTypes[i] != argType) {
          return $_op.emitOpError("type of entry block argument #")
                << i << '(' << argType
                << ") must match the type of the corresponding argument in "
                << "function signature(" << fnInputTypes[i] << ')';
        }
      }

      return success();
    }]>,
    InterfaceMethod<[{
      Verify the type attribute of the function for derived op-specific
      invariants.
    }],
    "::mlir::LogicalResult", "verifyType", (ins),
    /*methodBody=*/[{}], /*defaultImplementation=*/[{
      return success();
    }]>,
  ];

  let extraTraitClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // Builders
    //===------------------------------------------------------------------===//

    /// Build the function with the given name, attributes, and type. This
    /// builder also inserts an entry block into the function body with the
    /// given argument types.
    static void buildWithEntryBlock(
        OpBuilder &builder, OperationState &state, StringRef name, Type type,
        ArrayRef<NamedAttribute> attrs, TypeRange inputTypes) {
      state.addAttribute(SymbolTable::getSymbolAttrName(),
                        builder.getStringAttr(name));
      state.addAttribute(ConcreteOp::getFunctionTypeAttrName(state.name),
                        TypeAttr::get(type));
      state.attributes.append(attrs.begin(), attrs.end());

      // Add the function body.
      Region *bodyRegion = state.addRegion();
      Block *body = new Block();
      bodyRegion->push_back(body);
      for (Type input : inputTypes)
        body->addArgument(input, state.location);
    }
  }];
  let extraSharedClassDeclaration = [{
    /// Block list iterator types.
    using BlockListType = Region::BlockListType;
    using iterator = BlockListType::iterator;
    using reverse_iterator = BlockListType::reverse_iterator;

    /// Block argument iterator types.
    using BlockArgListType = Region::BlockArgListType;
    using args_iterator = BlockArgListType::iterator;

    //===------------------------------------------------------------------===//
    // Body Handling
    //===------------------------------------------------------------------===//

    /// Returns true if this function is external, i.e. it has no body.
    bool isExternal() { return empty(); }

    /// Return the region containing the body of this function.
    Region &getFunctionBody() { return $_op->getRegion(0); }

    /// Delete all blocks from this function.
    void eraseBody() {
      getFunctionBody().dropAllReferences();
      getFunctionBody().getBlocks().clear();
    }

    /// Return the list of blocks within the function body.
    BlockListType &getBlocks() { return getFunctionBody().getBlocks(); }

    iterator begin() { return getFunctionBody().begin(); }
    iterator end() { return getFunctionBody().end(); }
    reverse_iterator rbegin() { return getFunctionBody().rbegin(); }
    reverse_iterator rend() { return getFunctionBody().rend(); }

    /// Returns true if this function has no blocks within the body.
    bool empty() { return getFunctionBody().empty(); }

    /// Push a new block to the back of the body region.
    void push_back(Block *block) { getFunctionBody().push_back(block); }

    /// Push a new block to the front of the body region.
    void push_front(Block *block) { getFunctionBody().push_front(block); }

    /// Return the last block in the body region.
    Block &back() { return getFunctionBody().back(); }

    /// Return the first block in the body region.
    Block &front() { return getFunctionBody().front(); }

    /// Add an entry block to an empty function, and set up the block arguments
    /// to match the signature of the function. The newly inserted entry block
    /// is returned.
    Block *addEntryBlock() {
      assert(empty() && "function already has an entry block");
      Block *entry = new Block();
      push_back(entry);

      // FIXME: Allow for passing in locations for these arguments instead of using
      // the operations location.
      ArrayRef<Type> inputTypes = $_op.getArgumentTypes();
      SmallVector<Location> locations(inputTypes.size(),
                                      $_op.getOperation()->getLoc());
      entry->addArguments(inputTypes, locations);
      return entry;
    }

    /// Add a normal block to the end of the function's block list. The function
    /// should at least already have an entry block.
    Block *addBlock() {
      assert(!empty() && "function should at least have an entry block");
      push_back(new Block());
      return &back();
    }

    //===------------------------------------------------------------------===//
    // Type Attribute Handling
    //===------------------------------------------------------------------===//

    /// Change the type of this function in place. This is an extremely dangerous
    /// operation and it is up to the caller to ensure that this is legal for
    /// this function, and to restore invariants:
    ///  - the entry block args must be updated to match the function params.
    ///  - the argument/result attributes may need an update: if the new type
    ///    has less parameters we drop the extra attributes, if there are more
    ///    parameters they won't have any attributes.
    void setType(Type newType) {
      function_interface_impl::setFunctionType(this->getOperation(), newType);
    }

    //===------------------------------------------------------------------===//
    // Argument and Result Handling
    //===------------------------------------------------------------------===//

    /// Returns the number of function arguments.
    unsigned getNumArguments() { return $_op.getArgumentTypes().size(); }

    /// Returns the number of function results.
    unsigned getNumResults() { return $_op.getResultTypes().size(); }

    /// Returns the entry block function argument at the given index.
    BlockArgument getArgument(unsigned idx) {
      return getFunctionBody().getArgument(idx);
    }

    /// Support argument iteration.
    args_iterator args_begin() { return getFunctionBody().args_begin(); }
    args_iterator args_end() { return getFunctionBody().args_end(); }
    BlockArgListType getArguments() { return getFunctionBody().getArguments(); }

    /// Insert a single argument of type `argType` with attributes `argAttrs` and
    /// location `argLoc` at `argIndex`.
    void insertArgument(unsigned argIndex, Type argType, DictionaryAttr argAttrs,
                        Location argLoc) {
      insertArguments({argIndex}, {argType}, {argAttrs}, {argLoc});
    }

    /// Inserts arguments with the listed types, attributes, and locations at the
    /// listed indices. `argIndices` must be sorted. Arguments are inserted in the
    /// order they are listed, such that arguments with identical index will
    /// appear in the same order that they were listed here.
    void insertArguments(ArrayRef<unsigned> argIndices, TypeRange argTypes,
                        ArrayRef<DictionaryAttr> argAttrs,
                        ArrayRef<Location> argLocs) {
      unsigned originalNumArgs = $_op.getNumArguments();
      Type newType = $_op.getTypeWithArgsAndResults(
          argIndices, argTypes, /*resultIndices=*/{}, /*resultTypes=*/{});
      function_interface_impl::insertFunctionArguments(
          this->getOperation(), argIndices, argTypes, argAttrs, argLocs,
          originalNumArgs, newType);
    }

    /// Insert a single result of type `resultType` at `resultIndex`.
    void insertResult(unsigned resultIndex, Type resultType,
                      DictionaryAttr resultAttrs) {
      insertResults({resultIndex}, {resultType}, {resultAttrs});
    }

    /// Inserts results with the listed types at the listed indices.
    /// `resultIndices` must be sorted. Results are inserted in the order they are
    /// listed, such that results with identical index will appear in the same
    /// order that they were listed here.
    void insertResults(ArrayRef<unsigned> resultIndices, TypeRange resultTypes,
                      ArrayRef<DictionaryAttr> resultAttrs) {
      unsigned originalNumResults = $_op.getNumResults();
      Type newType = $_op.getTypeWithArgsAndResults(
        /*argIndices=*/{}, /*argTypes=*/{}, resultIndices, resultTypes);
      function_interface_impl::insertFunctionResults(
          this->getOperation(), resultIndices, resultTypes, resultAttrs,
          originalNumResults, newType);
    }

    /// Erase a single argument at `argIndex`.
    void eraseArgument(unsigned argIndex) {
      BitVector argsToErase($_op.getNumArguments());
      argsToErase.set(argIndex);
      eraseArguments(argsToErase);
    }

    /// Erases the arguments listed in `argIndices`.
    void eraseArguments(const BitVector &argIndices) {
      Type newType = $_op.getTypeWithoutArgs(argIndices);
      function_interface_impl::eraseFunctionArguments(
        this->getOperation(), argIndices, newType);
    }

    /// Erase a single result at `resultIndex`.
    void eraseResult(unsigned resultIndex) {
      BitVector resultsToErase($_op.getNumResults());
      resultsToErase.set(resultIndex);
      eraseResults(resultsToErase);
    }

    /// Erases the results listed in `resultIndices`.
    void eraseResults(const BitVector &resultIndices) {
      Type newType = $_op.getTypeWithoutResults(resultIndices);
      function_interface_impl::eraseFunctionResults(
          this->getOperation(), resultIndices, newType);
    }

    /// Return the type of this function with the specified arguments and
    /// results inserted. This is used to update the function's signature in
    /// the `insertArguments` and `insertResults` methods. The arrays must be
    /// sorted by increasing index.
    Type getTypeWithArgsAndResults(
      ArrayRef<unsigned> argIndices, TypeRange argTypes,
      ArrayRef<unsigned> resultIndices, TypeRange resultTypes) {
      SmallVector<Type> argStorage, resultStorage;
      TypeRange newArgTypes = function_interface_impl::insertTypesInto(
          $_op.getArgumentTypes(), argIndices, argTypes, argStorage);
      TypeRange newResultTypes = function_interface_impl::insertTypesInto(
          $_op.getResultTypes(), resultIndices, resultTypes, resultStorage);
      return $_op.cloneTypeWith(newArgTypes, newResultTypes);
    }

    /// Return the type of this function without the specified arguments and
    /// results. This is used to update the function's signature in the
    /// `eraseArguments` and `eraseResults` methods.
    Type getTypeWithoutArgsAndResults(
      const BitVector &argIndices, const BitVector &resultIndices) {
      SmallVector<Type> argStorage, resultStorage;
      TypeRange newArgTypes = function_interface_impl::filterTypesOut(
          $_op.getArgumentTypes(), argIndices, argStorage);
      TypeRange newResultTypes = function_interface_impl::filterTypesOut(
          $_op.getResultTypes(), resultIndices, resultStorage);
      return $_op.cloneTypeWith(newArgTypes, newResultTypes);
    }
    Type getTypeWithoutArgs(const BitVector &argIndices) {
      SmallVector<Type> argStorage;
      TypeRange newArgTypes = function_interface_impl::filterTypesOut(
          $_op.getArgumentTypes(), argIndices, argStorage);
      return $_op.cloneTypeWith(newArgTypes, $_op.getResultTypes());
    }
    Type getTypeWithoutResults(const BitVector &resultIndices) {
      SmallVector<Type> resultStorage;
      TypeRange newResultTypes = function_interface_impl::filterTypesOut(
          $_op.getResultTypes(), resultIndices, resultStorage);
      return $_op.cloneTypeWith($_op.getArgumentTypes(), newResultTypes);
    }

    //===------------------------------------------------------------------===//
    // Argument Attributes
    //===------------------------------------------------------------------===//

    /// Return all of the attributes for the argument at 'index'.
    ArrayRef<NamedAttribute> getArgAttrs(unsigned index) {
      return function_interface_impl::getArgAttrs(this->getOperation(), index);
    }

    /// Return an ArrayAttr containing all argument attribute dictionaries of
    /// this function, or nullptr if no arguments have attributes.
    ArrayAttr getAllArgAttrs() { return $_op.getArgAttrsAttr(); }

    /// Return all argument attributes of this function.
    void getAllArgAttrs(SmallVectorImpl<DictionaryAttr> &result) {
      if (ArrayAttr argAttrs = getAllArgAttrs()) {
        auto argAttrRange = argAttrs.template getAsRange<DictionaryAttr>();
        result.append(argAttrRange.begin(), argAttrRange.end());
      } else {
        result.append($_op.getNumArguments(),
                      DictionaryAttr::get(this->getOperation()->getContext()));
      }
    }

    /// Return the specified attribute, if present, for the argument at 'index',
    /// null otherwise.
    Attribute getArgAttr(unsigned index, StringAttr name) {
      auto argDict = getArgAttrDict(index);
      return argDict ? argDict.get(name) : nullptr;
    }
    Attribute getArgAttr(unsigned index, StringRef name) {
      auto argDict = getArgAttrDict(index);
      return argDict ? argDict.get(name) : nullptr;
    }

    template <typename AttrClass>
    AttrClass getArgAttrOfType(unsigned index, StringAttr name) {
      return getArgAttr(index, name).template dyn_cast_or_null<AttrClass>();
    }
    template <typename AttrClass>
    AttrClass getArgAttrOfType(unsigned index, StringRef name) {
      return getArgAttr(index, name).template dyn_cast_or_null<AttrClass>();
    }

    /// Set the attributes held by the argument at 'index'.
    void setArgAttrs(unsigned index, ArrayRef<NamedAttribute> attributes) {
      function_interface_impl::setArgAttrs($_op, index, attributes);
    }

    /// Set the attributes held by the argument at 'index'. `attributes` may be
    /// null, in which case any existing argument attributes are removed.
    void setArgAttrs(unsigned index, DictionaryAttr attributes) {
      function_interface_impl::setArgAttrs($_op, index, attributes);
    }
    void setAllArgAttrs(ArrayRef<DictionaryAttr> attributes) {
      assert(attributes.size() == $_op.getNumArguments());
      function_interface_impl::setAllArgAttrDicts(this->getOperation(), attributes);
    }
    void setAllArgAttrs(ArrayRef<Attribute> attributes) {
      assert(attributes.size() == $_op.getNumArguments());
      function_interface_impl::setAllArgAttrDicts(this->getOperation(), attributes);
    }
    void setAllArgAttrs(ArrayAttr attributes) {
      assert(attributes.size() == $_op.getNumArguments());
      $_op.setArgAttrsAttr(attributes);
    }

    /// If the an attribute exists with the specified name, change it to the new
    /// value. Otherwise, add a new attribute with the specified name/value.
    void setArgAttr(unsigned index, StringAttr name, Attribute value) {
      function_interface_impl::setArgAttr($_op, index, name, value);
    }
    void setArgAttr(unsigned index, StringRef name, Attribute value) {
      setArgAttr(index,
                 StringAttr::get(this->getOperation()->getContext(), name),
                 value);
    }

    /// Remove the attribute 'name' from the argument at 'index'. Return the
    /// attribute that was erased, or nullptr if there was no attribute with
    /// such name.
    Attribute removeArgAttr(unsigned index, StringAttr name) {
      return function_interface_impl::removeArgAttr($_op, index, name);
    }
    Attribute removeArgAttr(unsigned index, StringRef name) {
      return removeArgAttr(
          index, StringAttr::get(this->getOperation()->getContext(), name));
    }

    //===------------------------------------------------------------------===//
    // Result Attributes
    //===------------------------------------------------------------------===//

    /// Return all of the attributes for the result at 'index'.
    ArrayRef<NamedAttribute> getResultAttrs(unsigned index) {
      return function_interface_impl::getResultAttrs(this->getOperation(), index);
    }

    /// Return an ArrayAttr containing all result attribute dictionaries of this
    /// function, or nullptr if no result have attributes.
    ArrayAttr getAllResultAttrs() { return $_op.getResAttrsAttr(); }

    /// Return all result attributes of this function.
    void getAllResultAttrs(SmallVectorImpl<DictionaryAttr> &result) {
      if (ArrayAttr argAttrs = getAllResultAttrs()) {
        auto argAttrRange = argAttrs.template getAsRange<DictionaryAttr>();
        result.append(argAttrRange.begin(), argAttrRange.end());
      } else {
        result.append($_op.getNumResults(),
                      DictionaryAttr::get(this->getOperation()->getContext()));
      }
    }

    /// Return the specified attribute, if present, for the result at 'index',
    /// null otherwise.
    Attribute getResultAttr(unsigned index, StringAttr name) {
      auto argDict = getResultAttrDict(index);
      return argDict ? argDict.get(name) : nullptr;
    }
    Attribute getResultAttr(unsigned index, StringRef name) {
      auto argDict = getResultAttrDict(index);
      return argDict ? argDict.get(name) : nullptr;
    }

    template <typename AttrClass>
    AttrClass getResultAttrOfType(unsigned index, StringAttr name) {
      return getResultAttr(index, name).template dyn_cast_or_null<AttrClass>();
    }
    template <typename AttrClass>
    AttrClass getResultAttrOfType(unsigned index, StringRef name) {
      return getResultAttr(index, name).template dyn_cast_or_null<AttrClass>();
    }

    /// Set the attributes held by the result at 'index'.
    void setResultAttrs(unsigned index, ArrayRef<NamedAttribute> attributes) {
      function_interface_impl::setResultAttrs($_op, index, attributes);
    }

    /// Set the attributes held by the result at 'index'. `attributes` may be
    /// null, in which case any existing argument attributes are removed.
    void setResultAttrs(unsigned index, DictionaryAttr attributes) {
      function_interface_impl::setResultAttrs($_op, index, attributes);
    }
    void setAllResultAttrs(ArrayRef<DictionaryAttr> attributes) {
      assert(attributes.size() == $_op.getNumResults());
      function_interface_impl::setAllResultAttrDicts(
        this->getOperation(), attributes);
    }
    void setAllResultAttrs(ArrayRef<Attribute> attributes) {
      assert(attributes.size() == $_op.getNumResults());
      function_interface_impl::setAllResultAttrDicts(
        this->getOperation(), attributes);
    }
    void setAllResultAttrs(ArrayAttr attributes) {
      assert(attributes.size() == $_op.getNumResults());
      $_op.setResAttrsAttr(attributes);
    }

    /// If the an attribute exists with the specified name, change it to the new
    /// value. Otherwise, add a new attribute with the specified name/value.
    void setResultAttr(unsigned index, StringAttr name, Attribute value) {
      function_interface_impl::setResultAttr($_op, index, name, value);
    }
    void setResultAttr(unsigned index, StringRef name, Attribute value) {
      setResultAttr(index,
                    StringAttr::get(this->getOperation()->getContext(), name),
                    value);
    }

    /// Remove the attribute 'name' from the result at 'index'. Return the
    /// attribute that was erased, or nullptr if there was no attribute with
    /// such name.
    Attribute removeResultAttr(unsigned index, StringAttr name) {
      return function_interface_impl::removeResultAttr($_op, index, name);
    }

    /// Returns the dictionary attribute corresponding to the argument at
    /// 'index'. If there are no argument attributes at 'index', a null
    /// attribute is returned.
    DictionaryAttr getArgAttrDict(unsigned index) {
      assert(index < $_op.getNumArguments() && "invalid argument number");
      return function_interface_impl::getArgAttrDict(this->getOperation(), index);
    }

    /// Returns the dictionary attribute corresponding to the result at 'index'.
    /// If there are no result attributes at 'index', a null attribute is
    /// returned.
    DictionaryAttr getResultAttrDict(unsigned index) {
      assert(index < $_op.getNumResults() && "invalid result number");
      return function_interface_impl::getResultAttrDict(this->getOperation(), index);
    }
  }];

  let verify = "return function_interface_impl::verifyTrait(cast<ConcreteOp>($_op));";
}

#endif // MLIR_IR_FUNCTIONINTERFACES_TD_
