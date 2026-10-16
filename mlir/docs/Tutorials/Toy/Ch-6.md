# Chapter 6: Lowering to LLVM and CodeGeneration

[TOC]

In the [previous chapter](Ch-5.md), we introduced the
[dialect conversion](../../DialectConversion.md) framework and partially lowered
many of the `Toy` operations to affine loop nests for optimization. In this
chapter, we will finally lower to LLVM for code generation.

## Lowering to LLVM

For this lowering, we will again use the dialect conversion framework to perform
the heavy lifting. However, this time, we will be performing a full conversion
to the [LLVM dialect](../../Dialects/LLVM.md). Thankfully, we have already
lowered all but one of the `toy` operations, with the last being `toy.print`.
Before going over the conversion to LLVM, let's lower the `toy.print` operation.
We will lower this operation to a non-affine loop nest that invokes `printf` for
each element. Note that, because the dialect conversion framework supports
[transitive lowering](../../../getting_started/Glossary.md/#transitive-lowering),
we don't need to directly emit operations in the LLVM dialect. By transitive
lowering, we mean that the conversion framework may apply multiple patterns to
fully legalize an operation. In this example, we are generating a structured
loop nest instead of the branch-form in the LLVM dialect. As long as we then
have a lowering from the loop operations to LLVM, the lowering will still
succeed.

During lowering we can get, or build, the declaration for printf as so:

```c++
/// Create a function declaration for printf, the signature is:
///   * `i32 (ptr, ...)`
static LLVM::LLVMFunctionType getPrintfType(MLIRContext *context) {
  auto llvmI32Ty = IntegerType::get(context, 32);
  auto llvmPtrTy = LLVM::LLVMPointerType::get(context);
  auto llvmFnType = LLVM::LLVMFunctionType::get(llvmI32Ty, llvmPtrTy,
                                                /*isVarArg=*/true);
  return llvmFnType;
}

/// Return a symbol reference to the printf function, inserting it into the
/// module if necessary.
static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter,
                                           ModuleOp module) {
  auto *context = module.getContext();
  if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
    return SymbolRefAttr::get(context, "printf");

  // Insert the printf function into the body of the parent module.
  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  LLVM::LLVMFuncOp::create(rewriter, module.getLoc(), "printf",
                           getPrintfType(context));
  return SymbolRefAttr::get(context, "printf");
}
```

Now that the lowering for the printf operation has been defined, we can specify
the components necessary for the lowering. These are largely the same as the
components defined in the [previous chapter](Ch-5.md).

### Conversion Target

For this conversion, aside from the top-level module, we will be lowering
everything to the LLVM dialect. `LLVMConversionTarget` is a `ConversionTarget`
that already marks the LLVM dialect as legal.

```c++
  mlir::LLVMConversionTarget target(getContext());
  target.addLegalOp<mlir::ModuleOp>();
```

### Type Converter

This lowering will also transform the MemRef types which are currently being
operated on into a representation in LLVM. To perform this conversion, we use a
TypeConverter as part of the lowering. This converter specifies how one type
maps to another. This is necessary now that we are performing more complicated
lowerings involving block arguments. Given that we don't have any
Toy-dialect-specific types that need to be lowered, the default converter is
enough for our use case.

```c++
  LLVMTypeConverter typeConverter(&getContext());
```

### Conversion Patterns

Now that the conversion target has been defined, we need to provide the patterns
used for lowering. At this point in the compilation process, we have a
combination of `toy`, `affine`, `arith`, `memref`, and `func` operations.
Luckily, the `affine`, `arith`, `memref`, and `func` dialects already provide
the set of patterns needed to transform them into LLVM dialect. These patterns
allow for lowering the IR in multiple stages by relying on
[transitive lowering](../../../getting_started/Glossary.md/#transitive-lowering).

```c++
  mlir::RewritePatternSet patterns(&getContext());
  mlir::populateAffineToStdConversionPatterns(patterns);
  mlir::populateSCFToControlFlowConversionPatterns(patterns);
  mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
  mlir::populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
  mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
  mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);

  // The only remaining operation, to lower from the `toy` dialect, is the
  // PrintOp.
  patterns.add<PrintOpLowering>(&getContext());
```

### Full Lowering

We want to completely lower to LLVM, so we use a `FullConversion`. This ensures
that only legal operations will remain after the conversion.

```c++
  mlir::ModuleOp module = getOperation();
  if (mlir::failed(
          mlir::applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
```

Looking back at our current working example:

```mlir
toy.func @main() {
  %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %2 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
  %3 = toy.mul %2, %2 : tensor<3x2xf64>
  toy.print %3 : tensor<3x2xf64>
  toy.return
}
```

We can now lower down to the LLVM dialect, which produces the following code:

```mlir
llvm.func @free(!llvm.ptr)
llvm.mlir.global internal constant @nl("\0A\00") {addr_space = 0 : i32}
llvm.mlir.global internal constant @frmt_spec("%f \00") {addr_space = 0 : i32}
llvm.func @printf(!llvm.ptr, ...) -> i32
llvm.func @malloc(i64) -> !llvm.ptr
llvm.func @main() {
  %0 = llvm.mlir.constant(6.000000e+00 : f64) : f64
  %1 = llvm.mlir.constant(5.000000e+00 : f64) : f64
  %2 = llvm.mlir.constant(4.000000e+00 : f64) : f64
  %3 = llvm.mlir.constant(3.000000e+00 : f64) : f64
  %4 = llvm.mlir.constant(2.000000e+00 : f64) : f64
  %5 = llvm.mlir.constant(1.000000e+00 : f64) : f64

...

^bb16:  // pred: ^bb15
  %162 = llvm.extractvalue %22[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  %163 = llvm.mlir.constant(2 : i64) : i64
  %164 = llvm.mul %155, %163 overflow<nsw, nuw> : i64
  %165 = llvm.add %164, %160 overflow<nsw, nuw> : i64
  %166 = llvm.getelementptr inbounds|nuw %162[%165] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %167 = llvm.load %166 : !llvm.ptr -> f64
  %168 = llvm.call @printf(%148, %167) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64) -> i32
  %169 = llvm.add %160, %159 : i64
  llvm.br ^bb15(%169 : i64)

...

^bb18:  // pred: ^bb13
  %172 = llvm.extractvalue %56[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  llvm.call @free(%172) : (!llvm.ptr) -> ()
  %173 = llvm.extractvalue %39[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  llvm.call @free(%173) : (!llvm.ptr) -> ()
  %174 = llvm.extractvalue %22[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  llvm.call @free(%174) : (!llvm.ptr) -> ()
  llvm.return
}
```

Each `memref` value has been lowered to an LLVM struct, the memref descriptor,
holding the allocated pointer, the aligned pointer, an offset, and the size and
stride of each dimension. See
[LLVM IR Target](../../TargetLLVMIR.md/#ranked-memref-types) for more in-depth
details on lowering to the LLVM dialect.

## CodeGen: Getting Out of MLIR

At this point we are right at the cusp of code generation. We can generate code
in the LLVM dialect, so now we just need to export to LLVM IR and setup a JIT to
run it.

### Emitting LLVM IR

Now that our module is comprised only of operations in the LLVM dialect, we can
export to LLVM IR. To do this programmatically, we can invoke the following
utility:

```c++
  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule =
      mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule)
    /* ... an error was encountered ... */
```

Exporting our module to LLVM IR generates the following. The target triple,
data layout, and debug location metadata have been elided for brevity:

```llvm
@nl = internal constant [2 x i8] c"\0A\00"
@frmt_spec = internal constant [4 x i8] c"%f \00"

declare void @free(ptr)

declare i32 @printf(ptr, ...)

declare ptr @malloc(i64)

define void @main() {
  ...

87:                                               ; preds = %84
  %88 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 1
  %89 = mul nuw nsw i64 %81, 2
  %90 = add nuw nsw i64 %89, %85
  %91 = getelementptr inbounds nuw double, ptr %88, i64 %90
  %92 = load double, ptr %91, align 8
  %93 = call i32 (ptr, ...) @printf(ptr @frmt_spec, double %92)
  %94 = add i64 %85, 1
  br label %84

  ...

98:                                               ; preds = %80
  %99 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, 0
  call void @free(ptr %99)
  %100 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %16, 0
  call void @free(ptr %100)
  %101 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %8, 0
  call void @free(ptr %101)
  ret void
}
```

If we enable optimization on the generated LLVM IR, we can trim this down quite
a bit:

```llvm
@frmt_spec = internal constant [4 x i8] c"%f \00"

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #0

; Function Attrs: nofree nounwind
define void @main() local_unnamed_addr #0 {
.preheader5:
  %0 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double 1.000000e+00)
  %1 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double 1.600000e+01)
  %putchar = tail call i32 @putchar(i32 10)
  %2 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double 4.000000e+00)
  %3 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double 2.500000e+01)
  %putchar.1 = tail call i32 @putchar(i32 10)
  %4 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double 9.000000e+00)
  %5 = tail call i32 (ptr, ...) @printf(ptr nonnull dereferenceable(1) @frmt_spec, double 3.600000e+01)
  %putchar.2 = tail call i32 @putchar(i32 10)
  ret void
}

; Function Attrs: nofree nounwind
declare noundef i32 @putchar(i32 noundef) local_unnamed_addr #0

attributes #0 = { nofree nounwind }
```

The full code listing for dumping LLVM IR can be found in
`examples/toy/Ch6/toyc.cpp` in the `dumpLLVMIR()` function:

```c++
static int dumpLLVMIR(mlir::ModuleOp module) {
  // Register the translation to LLVM IR with the MLIR context.
  mlir::registerBuiltinDialectTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());

  // Convert the module to LLVM IR in a new LLVM IR context.
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return -1;
  }

  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Configure the LLVM Module
  auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!tmBuilderOrError) {
    llvm::errs() << "Could not create JITTargetMachineBuilder\n";
    return -1;
  }

  auto tmOrError = tmBuilderOrError->createTargetMachine();
  if (!tmOrError) {
    llvm::errs() << "Could not create TargetMachine\n";
    return -1;
  }
  mlir::ExecutionEngine::setupTargetTripleAndDataLayout(llvmModule.get(),
                                                        tmOrError.get().get());

  /// Optionally run an optimization pipeline over the llvm module.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return -1;
  }
  llvm::errs() << *llvmModule << "\n";
  return 0;
}
```

### Setting up a JIT

Setting up a JIT to run the module containing the LLVM dialect can be done using
the `mlir::ExecutionEngine` infrastructure. This is a utility wrapper around
LLVM's JIT that accepts `.mlir` as input. The full code listing for setting up
the JIT can be found in `Ch6/toyc.cpp` in the `runJit()` function:

```c++
static int runJit(mlir::ModuleOp module) {
  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Register the translation from MLIR to LLVM IR, which must happen before we
  // can JIT-compile.
  mlir::registerBuiltinDialectTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());

  // An optimization pipeline to use within the execution engine.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

  // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
  // the module.
  mlir::ExecutionEngineOptions engineOptions;
  engineOptions.transformer = optPipeline;
  auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
  assert(maybeEngine && "failed to construct an execution engine");
  auto &engine = maybeEngine.get();

  // Invoke the JIT-compiled function.
  auto invocationResult = engine->invokePacked("main");
  if (invocationResult) {
    llvm::errs() << "JIT invocation failed\n";
    return -1;
  }

  return 0;
}
```

You can play around with it from the build directory:

```shell
$ echo 'def main() { print([[1, 2], [3, 4]]); }' | ./bin/toyc-ch6 -emit=jit
1.000000 2.000000
3.000000 4.000000
```

You can also play with `-emit=mlir`, `-emit=mlir-affine`, `-emit=mlir-llvm`, and
`-emit=llvm` to compare the various levels of IR involved. Also try options like
[`--mlir-print-ir-after-all`](../../PassManagement.md/#ir-printing) to track the
evolution of the IR throughout the pipeline.

The example code used throughout this section can be found in
test/Examples/Toy/Ch6/llvm-lowering.mlir.

So far, we have worked with primitive data types. In the
[next chapter](Ch-7.md), we will add a composite `struct` type.
