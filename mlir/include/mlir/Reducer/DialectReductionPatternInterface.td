#ifndef MLIR_INTERFACES_DIALECTREDUCTIONPATTERNINTERFACE
#define MLIR_INTERFACES_DIALECTREDUCTIONPATTERNINTERFACE

include "mlir/IR/Interfaces.td"

def DialectReductionPatternInterface : DialectInterface<"DialectReductionPatternInterface"> {
  let description = [{
    This is used to report the reduction patterns for a Dialect. While using
    mlir-reduce to reduce a module, we may want to transform certain cases into
    simpler forms by applying certain rewrite patterns. Implement the
    `populateReductionPatterns` to report those patterns by adding them to the
    RewritePatternSet.

    Example:
      MyDialectReductionPattern::populateReductionPatterns(
          RewritePatternSet &patterns) {
          patterns.add<TensorOpReduction>(patterns.getContext());
      }

    For DRR, mlir-tblgen will generate a helper function
    `populateWithGenerated` which has the same signature therefore you can
    delegate to the helper function as well.

    Example:
      MyDialectReductionPattern::populateReductionPatterns(
          RewritePatternSet &patterns) {
          // Include the autogen file somewhere above.
          populateWithGenerated(patterns);
      }
  }];
  let cppNamespace = "::mlir";

  let methods = [
    PureVirtualInterfaceMethod<[{
        Patterns provided here are intended to transform operations from a complex
        form to a simpler form, without breaking the semantics of the program
        being reduced. For example, you may want to replace the
        tensor<?xindex> with a known rank and type, e.g. tensor<1xi32>, or
        replacing an operation with a constant.
      }],
      "void", "populateReductionPatterns",
      (ins "::mlir::RewritePatternSet &":$patterns)
    >,
    InterfaceMethod<[{
        This method extends `populateReductionPatterns` by allowing reduction
        patterns to use a `Tester` instance. Some reduction patterns may need to
        run tester to determine whether certain transformations preserve the
        "interesting" behavior of the program. This is mostly useful when pattern
        should choose between multiple modifications.
      }],
      "void", "populateReductionPatternsWithTester",
      (ins "::mlir::RewritePatternSet &":$patterns, "::mlir::Tester &":$tester),
      [{}]
    > 
  ];
}


#endif
