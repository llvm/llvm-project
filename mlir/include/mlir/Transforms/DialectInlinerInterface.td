#ifndef MLIR_INTERFACES_DIALECTINLINERINTERFACE
#define MLIR_INTERFACES_DIALECTINLINERINTERFACE

include "mlir/IR/Interfaces.td"

def DialectInlinerInterface : DialectInterface<"DialectInlinerInterface"> {
  let description = [{
    This is the interface that must be implemented by the dialects of operations
    to be inlined. This interface should only handle the operations of the
    given dialect.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
        Returns true if the given operation 'callable', that implements the
        'CallableOpInterface', can be inlined into the position given call
        operation 'call', that is registered to the current dialect and implements
        the `CallOpInterface`. 'wouldBeCloned' is set to true if the region of the
        given 'callable' is set to be cloned during the inlining process, or false
        if the region is set to be moved in-place(i.e. no duplicates would be
        created).
      }],
      "bool", "isLegalToInline",
      (ins "::mlir::Operation *":$call, "::mlir::Operation *":$callable, 
           "bool":$wouldBeCloned),
      [{
        return false;
      }]
    >,
    InterfaceMethod<[{
        Returns true if the given region 'src' can be inlined into the region
        'dest' that is attached to an operation registered to the current dialect.
        'wouldBeCloned' is set to true if the given 'src' region is set to be
        cloned during the inlining process, or false if the region is set to be
        moved in-place (i.e. no duplicates would be created). 'valueMapping'
        contains any remapped values from within the 'src' region. This can be
        used to examine what values will replace entry arguments into the 'src'
        region for example.
      }],
      "bool", "isLegalToInline",
      (ins "::mlir::Region *":$dest, "::mlir::Region *":$src, "bool":$wouldBeCloned,
                        "::mlir::IRMapping &":$valueMapping),
      [{
        return false;
      }]
    >,
    InterfaceMethod<[{
        Returns true if the given region 'src' can be inlined into the region
        'dest' that is attached to an operation registered to the current dialect.
        'wouldBeCloned' is set to true if the given 'src' region is set to be
        cloned during the inlining process, or false if the region is set to be
        moved in-place(i.e. no duplicates would be created). 'valueMapping'
        contains any remapped values from within the 'src' region. This can be
        used to examine what values will replace entry arguments into the 'src'
        region for example.
      }],
      "bool", "isLegalToInline",
      (ins "::mlir::Operation *":$op, "::mlir::Region *":$dest,
           "bool":$wouldBeCloned, "::mlir::IRMapping &":$valueMapping),
      [{
        return false;
      }]
    >,
    InterfaceMethod<[{
        This hook is invoked on an operation that contains regions. It should
        return true if the analyzer should recurse within the regions of this
        operation when computing legality and cost, false otherwise. The default
        implementation returns true.
      }],
      "bool", "shouldAnalyzeRecursively",
      (ins "::mlir::Operation *":$op),
      [{
        return true;
      }]
    >,
    InterfaceMethod<[{
        Handle the given inlined terminator by replacing it with a new operation
        as necessary. This overload is called when the inlined region has more
        than one block. The 'newDest' block represents the new final branching
        destination of blocks within this region, i.e. operations that release
        control to the parent operation will likely now branch to this block.
        Its block arguments correspond to any values that need to be replaced by
        terminators within the inlined region.
      }],
      "void", "handleTerminator",
      (ins "::mlir::Operation *":$op, "::mlir::Block *":$newDest),
      [{
        llvm_unreachable("must implement handleTerminator in the case of multiple "
                         "inlined blocks");
      }]
    >,
    InterfaceMethod<[{
        Handle the given inlined terminator by replacing it with a new operation
        as necessary. This overload is called when the inlined region only
        contains one block. 'valuesToReplace' contains the previously returned
        values of the call site before inlining. These values must be replaced by
        this callback if they had any users (for example for traditional function
        calls, these are directly replaced with the operands of the `return`
        operation). The given 'op' will be removed by the caller, after this
        function has been called.
      }],
      "void", "handleTerminator",
      (ins "::mlir::Operation *":$op, "::mlir::ValueRange":$valuesToReplace),
      [{
        llvm_unreachable(
            "must implement handleTerminator in the case of one inlined block");
      }]
    >,
    InterfaceMethod<[{
        Attempt to materialize a conversion for a type mismatch between a call
        from this dialect, and a callable region. This method should generate an
        operation that takes 'input' as the only operand, and produces a single
        result of 'resultType'. If a conversion can not be generated, nullptr
        should be returned. For example, this hook may be invoked in the following
        scenarios:

          ```mlir
          func @foo(i32) -> i32 { ... }
        
          // Mismatched input operand ... = foo.call @foo(%input : i16) -> i32
        
          // Mismatched result type.
          ... = foo.call @foo(%input : i32) -> i16
          ```
        
        NOTE: This hook may be invoked before the 'isLegal' checks above.
      }],
      "::mlir::Operation *", "materializeCallConversion",
      (ins "::mlir::OpBuilder &":$builder, "::mlir::Value":$input, 
           "::mlir::Type":$resultType, "::mlir::Location":$conversionLoc),
      [{
        return nullptr;
      }]
    >,
    InterfaceMethod<[{
        Hook to transform the call arguments before using them to replace the
        callee arguments. Returns a value of the same type or the `argument`
        itself if nothing changed. The `argumentAttrs` dictionary is non-null even
        if no attribute is present. The hook is called after converting the
        callsite argument types using the materializeCallConversion callback, and
        right before inlining the callee region. Any operations created using the
        provided `builder` are inserted right before the inlined callee region. An
        example use case is the insertion of copies for by value arguments.
      }],
      "::mlir::Value", "handleArgument",
      (ins "::mlir::OpBuilder &":$builder, "::mlir::Operation *":$call, 
           "::mlir::Operation *":$callable, "::mlir::Value":$argument,
           "::mlir::DictionaryAttr":$argumentAttrs),
      [{
        return argument;
      }]
    >,
    InterfaceMethod<[{
        Hook to transform the callee results before using them to replace the call
        results. Returns a value of the same type or the `result` itself if
        nothing changed. The `resultAttrs` dictionary is non-null even if no
        attribute is present. The hook is called right before handling
        terminators, and obtains the callee result before converting its type
        using the `materializeCallConversion` callback. Any operations created
        using the provided `builder` are inserted right after the inlined callee
        region. An example use case is the insertion of copies for by value
        results. NOTE: This hook is invoked after inlining the `callable` region.
      }],
      "::mlir::Value", "handleResult",
      (ins "::mlir::OpBuilder &":$builder, "::mlir::Operation *":$call, 
           "::mlir::Operation *":$callable, "::mlir::Value":$result,
           "::mlir::DictionaryAttr":$resultAttrs),
      [{
        return result;
      }]
    >,
    InterfaceMethod<[{
        Process a set of blocks that have been inlined for a call. This callback
        is invoked before inlined terminator operations have been processed.
      }],
      "void", "processInlinedCallBlocks",
      (ins "::mlir::Operation *":$call, 
           "::mlir::iterator_range<::mlir::Region::iterator>":$inlinedBlocks),
      [{}]
    >,
    InterfaceMethod<[{
        Returns true if the inliner can assume a fast path of not creating a new
        block, if there is only one block.
      }],
      "bool", "allowSingleBlockOptimization",
      (ins "::mlir::iterator_range<::mlir::Region::iterator>":$inlinedBlocks),
      [{
        return true;
      }]
    >
  ];
}


#endif
