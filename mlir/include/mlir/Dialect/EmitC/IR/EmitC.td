//===- EmitC.td - EmitC operations--------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the MLIR EmitC operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_EMITC_IR_EMITC
#define MLIR_DIALECT_EMITC_IR_EMITC

include "mlir/Dialect/EmitC/IR/EmitCAttributes.td"
include "mlir/Dialect/EmitC/IR/EmitCTypes.td"

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"

//===----------------------------------------------------------------------===//
// EmitC op definitions
//===----------------------------------------------------------------------===//

// Base class for EmitC dialect ops.
class EmitC_Op<string mnemonic, list<Trait> traits = []>
    : Op<EmitC_Dialect, mnemonic, traits>;

// Base class for unary operations.
class EmitC_UnaryOp<string mnemonic, list<Trait> traits = []> :
    EmitC_Op<mnemonic, traits> {
  let arguments = (ins EmitCType);
  let results = (outs EmitCType);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// Base class for binary operations.
class EmitC_BinaryOp<string mnemonic, list<Trait> traits = []> :
    EmitC_Op<mnemonic, traits> {
  let arguments = (ins EmitCType:$lhs, EmitCType:$rhs);
  let results = (outs EmitCType);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
}

// EmitC OpTrait
def CExpression : NativeOpTrait<"emitc::CExpression">;

// Types only used in binary arithmetic operations.
def IntegerIndexOrOpaqueType : Type<CPred<"emitc::isIntegerIndexOrOpaqueType($_self)">,
"integer, index or opaque type supported by EmitC">;
def FloatIntegerIndexOrOpaqueType : AnyTypeOf<[EmitCFloatType, IntegerIndexOrOpaqueType]>;

def EmitC_AddOp : EmitC_BinaryOp<"add", [CExpression]> {
  let summary = "Addition operation";
  let description = [{
    With the `emitc.add` operation the arithmetic operator + (addition) can
    be applied.

    Example:

    ```mlir
    // Custom form of the addition operation.
    %0 = emitc.add %arg0, %arg1 : (i32, i32) -> i32
    %1 = emitc.add %arg2, %arg3 : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
    ```
    ```c++
    // Code emitted for the operations above.
    int32_t v5 = v1 + v2;
    float* v6 = v3 + v4;
    ```
  }];

  let hasVerifier = 1;
}

def EmitC_ApplyOp : EmitC_Op<"apply", [CExpression]> {
  let summary = "Apply operation";
  let description = [{
    With the `emitc.apply` operation the operators & (address of) and * (contents of)
    can be applied to a single operand.

    Example:

    ```mlir
    // Custom form of applying the & operator.
    %0 = emitc.apply "&"(%arg0) : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>

    // Generic form of the same operation.
    %0 = "emitc.apply"(%arg0) {applicableOperator = "&"}
        : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>

    ```
  }];
  let arguments = (ins
    Arg<StrAttr, "the operator to apply">:$applicableOperator,
    AnyTypeOf<[EmitCType, EmitC_LValueType]>:$operand
  );
  let results = (outs EmitCType:$result);
  let assemblyFormat = [{
    $applicableOperator `(` $operand `)` attr-dict `:` functional-type($operand, results)
  }];
  let hasVerifier = 1;
}

def EmitC_BitwiseAndOp : EmitC_BinaryOp<"bitwise_and", [CExpression]> {
  let summary = "Bitwise and operation";
  let description = [{
    With the `emitc.bitwise_and` operation the bitwise operator & (and) can
    be applied.

    Example:

    ```mlir
    %0 = emitc.bitwise_and %arg0, %arg1 : (i32, i32) -> i32
    ```
    ```c++
    // Code emitted for the operation above.
    int32_t v3 = v1 & v2;
    ```
  }];
}

def EmitC_BitwiseLeftShiftOp : EmitC_BinaryOp<"bitwise_left_shift",
    [CExpression]> {
  let summary = "Bitwise left shift operation";
  let description = [{
    With the `emitc.bitwise_left_shift` operation the bitwise operator <<
    (left shift) can be applied.

    Example:

    ```mlir
    %0 = emitc.bitwise_left_shift %arg0, %arg1 : (i32, i32) -> i32
    ```
    ```c++
    // Code emitted for the operation above.
    int32_t v3 = v1 << v2;
    ```
  }];
}

def EmitC_BitwiseNotOp : EmitC_UnaryOp<"bitwise_not", [CExpression]> {
  let summary = "Bitwise not operation";
  let description = [{
    With the `emitc.bitwise_not` operation the bitwise operator ~ (not) can
    be applied.

    Example:

    ```mlir
    %0 = emitc.bitwise_not %arg0 : (i32) -> i32
    ```
    ```c++
    // Code emitted for the operation above.
    int32_t v2 = ~v1;
    ```
  }];
}

def EmitC_BitwiseOrOp : EmitC_BinaryOp<"bitwise_or", [CExpression]> {
  let summary = "Bitwise or operation";
  let description = [{
    With the `emitc.bitwise_or` operation the bitwise operator | (or)
    can be applied.

    Example:

    ```mlir
    %0 = emitc.bitwise_or %arg0, %arg1 : (i32, i32) -> i32
    ```
    ```c++
    // Code emitted for the operation above.
    int32_t v3 = v1 | v2;
    ```
  }];
}

def EmitC_BitwiseRightShiftOp : EmitC_BinaryOp<"bitwise_right_shift",
    [CExpression]> {
  let summary = "Bitwise right shift operation";
  let description = [{
    With the `emitc.bitwise_right_shift` operation the bitwise operator >>
    (right shift) can be applied.

    Example:

    ```mlir
    %0 = emitc.bitwise_right_shift %arg0, %arg1 : (i32, i32) -> i32
    ```
    ```c++
    // Code emitted for the operation above.
    int32_t v3 = v1 >> v2;
    ```
  }];
}

def EmitC_BitwiseXorOp : EmitC_BinaryOp<"bitwise_xor", [CExpression]> {
  let summary = "Bitwise xor operation";
  let description = [{
    With the `emitc.bitwise_xor` operation the bitwise operator ^ (xor)
    can be applied.

    Example:

    ```mlir
    %0 = emitc.bitwise_xor %arg0, %arg1 : (i32, i32) -> i32
    ```
    ```c++
    // Code emitted for the operation above.
    int32_t v3 = v1 ^ v2;
    ```
  }];
}

def EmitC_CallOpaqueOp : EmitC_Op<"call_opaque", [CExpression]> {
  let summary = "Opaque call operation";
  let description = [{
    The `emitc.call_opaque` operation represents a C++ function call. The callee
    can be an arbitrary non-empty string. The call allows specifying order
    of operands and attributes in the call as follows:

    - integer value of index type refers to an operand;
    - attribute which will get lowered to constant value in call;

    Example:

    ```mlir
    // Custom form defining a call to `foo()`.
    %0 = emitc.call_opaque "foo" () : () -> i32

    // Generic form of the same operation.
    %0 = "emitc.call_opaque"() {callee = "foo"} : () -> i32
    ```
  }];
  let arguments = (ins
    Arg<StrAttr, "the C++ function to call">:$callee,
    Arg<OptionalAttr<ArrayAttr>, "the order of operands and further attributes">:$args,
    Arg<OptionalAttr<ArrayAttr>, "template arguments">:$template_args,
    Variadic<EmitCType>:$operands
  );
  let results = (outs Variadic<EmitCType>);
  let builders = [
    OpBuilder<(ins
      "::mlir::TypeRange":$resultTypes,
      "::llvm::StringRef":$callee,
      "::mlir::ValueRange":$operands,
      CArg<"::mlir::ArrayAttr", "{}">:$args,
      CArg<"::mlir::ArrayAttr", "{}">:$template_args), [{
        build($_builder, $_state, resultTypes, callee, args, template_args,
            operands);
      }]
    >
  ];

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
  let hasVerifier = 1;
}

def EmitC_CastOp : EmitC_Op<"cast",
    [CExpression,
     DeclareOpInterfaceMethods<CastOpInterface>,
     SameOperandsAndResultShape]> {
  let summary = "Cast operation";
  let description = [{
    The `emitc.cast` operation performs an explicit type conversion and is emitted
    as a C-style cast expression. It can be applied to integer, float, index
    and EmitC types.

    Example:

    ```mlir
    // Cast from `int32_t` to `float`
    %0 = emitc.cast %arg0: i32 to f32

    // Cast from `void` to `int32_t` pointer
    %1 = emitc.cast %arg1 :
        !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<i32>
    ```
  }];

  let arguments = (ins EmitCType:$source);
  let results = (outs EmitCType:$dest);
  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($dest)";
}

def EmitC_CmpOp : EmitC_BinaryOp<"cmp", [CExpression]> {
  let summary = "Comparison operation";
  let description = [{
    With the `emitc.cmp` operation the comparison operators ==, !=, <, <=, >, >=, <=> 
    can be applied.

    Its first argument is an attribute that defines the comparison operator:

    - equal to (mnemonic: `"eq"`; integer value: `0`)
    - not equal to (mnemonic: `"ne"`; integer value: `1`)
    - less than (mnemonic: `"lt"`; integer value: `2`)
    - less than or equal to (mnemonic: `"le"`; integer value: `3`)
    - greater than (mnemonic: `"gt"`; integer value: `4`)
    - greater than or equal to (mnemonic: `"ge"`; integer value: `5`)
    - three-way-comparison (mnemonic: `"three_way"`; integer value: `6`)

    Example:
    ```mlir
    // Custom form of the cmp operation.
    %0 = emitc.cmp eq, %arg0, %arg1 : (i32, i32) -> i1
    %1 = emitc.cmp lt, %arg2, %arg3 : 
        (
          !emitc.opaque<"std::valarray<float>">,
          !emitc.opaque<"std::valarray<float>">
        ) -> !emitc.opaque<"std::valarray<bool>">
    ```
    ```c++
    // Code emitted for the operations above.
    bool v5 = v1 == v2;
    std::valarray<bool> v6 = v3 < v4;
    ```
  }];

  let arguments = (ins EmitC_CmpPredicateAttr:$predicate,
                       EmitCType:$lhs,
                       EmitCType:$rhs);
  let results = (outs EmitCType);

  let assemblyFormat = "$predicate `,` operands attr-dict `:` functional-type(operands, results)";
}

def EmitC_ConstantOp : EmitC_Op<"constant", [ConstantLike]> {
  let summary = "Constant operation";
  let description = [{
    The `emitc.constant` operation produces an SSA value equal to some constant
    specified by an attribute. This can be used to form simple integer and
    floating point constants, as well as more exotic things like tensor
    constants. The `emitc.constant` operation also supports the EmitC opaque
    attribute and the EmitC opaque type. Since folding is supported,
    it should not be used with pointers.

    Example:

    ```mlir
    // Integer constant
    %0 = "emitc.constant"(){value = 42 : i32} : () -> i32

    // Constant emitted as `char = CHAR_MIN;`
    %1 = "emitc.constant"() {value = #emitc.opaque<"CHAR_MIN">}
      : () -> !emitc.opaque<"char">
    ```
  }];

  let arguments = (ins EmitC_OpaqueOrTypedAttr:$value);
  let results = (outs EmitCType);

  let hasFolder = 1;
  let hasVerifier = 1;
}

def EmitC_DivOp : EmitC_BinaryOp<"div", [CExpression]> {
  let summary = "Division operation";
  let description = [{
    With the `emitc.div` operation the arithmetic operator / (division) can
    be applied.

    Example:

    ```mlir
    // Custom form of the division operation.
    %0 = emitc.div %arg0, %arg1 : (i32, i32) -> i32
    %1 = emitc.div %arg2, %arg3 : (f32, f32) -> f32
    ```
    ```c++
    // Code emitted for the operations above.
    int32_t v5 = v1 / v2;
    float v6 = v3 / v4;
    ```
  }];

  let arguments = (ins FloatIntegerIndexOrOpaqueType, FloatIntegerIndexOrOpaqueType);
  let results = (outs FloatIntegerIndexOrOpaqueType);
}

def EmitC_ExpressionOp : EmitC_Op<"expression",
      [HasOnlyGraphRegion, OpAsmOpInterface,
       SingleBlockImplicitTerminator<"emitc::YieldOp">, NoRegionArguments]> {
  let summary = "Expression operation";
  let description = [{
    The `emitc.expression` operation returns a single SSA value which is yielded by
    its single-basic-block region. The operation doesn't take any arguments.

    As the operation is to be emitted as a C expression, the operations within
    its body must form a single Def-Use tree of emitc ops whose result is
    yielded by a terminating `emitc.yield`.

    Example:

    ```mlir
    %r = emitc.expression : i32 {
      %0 = emitc.add %a, %b : (i32, i32) -> i32
      %1 = emitc.call_opaque "foo"(%0) : (i32) -> i32
      %2 = emitc.add %c, %d : (i32, i32) -> i32
      %3 = emitc.mul %1, %2 : (i32, i32) -> i32
      emitc.yield %3 : i32
    }
    ```

    May be emitted as

    ```c++
    int32_t v7 = foo(v1 + v2) * (v3 + v4);
    ```

    The operations allowed within expression body are EmitC operations with the
    CExpression trait.

    When specified, the optional `do_not_inline` indicates that the expression is
    to be emitted as seen above, i.e. as the rhs of an EmitC SSA value
    definition. Otherwise, the expression may be emitted inline, i.e. directly
    at its use.
  }];

  let arguments = (ins UnitAttr:$do_not_inline);
  let results = (outs EmitCType:$result);
  let regions = (region SizedRegion<1>:$region);

  let hasVerifier = 1;
  let assemblyFormat = "attr-dict (`noinline` $do_not_inline^)? `:` type($result) $region";

  let extraClassDeclaration = [{
    bool hasSideEffects() {
      auto predicate = [](Operation &op) {
        assert(op.hasTrait<OpTrait::emitc::CExpression>() && "Expected a C expression");
        // Conservatively assume calls to read and write memory.
        if (isa<emitc::CallOpaqueOp>(op))
          return true;
        // De-referencing reads modifiable memory, address-taking has no
        // side-effect.
        auto applyOp = dyn_cast<emitc::ApplyOp>(op);
        if (applyOp)
          return applyOp.getApplicableOperator() == "*";
        // Any operation using variables is assumed to have a side effect of
        // reading memory mutable by emitc::assign ops.
        return llvm::any_of(op.getOperands(), [](Value operand) {
          Operation *def = operand.getDefiningOp();
          return def && isa<emitc::VariableOp>(def);
        });
      };
      return llvm::any_of(getRegion().front().without_terminator(), predicate);
    };
    Operation *getRootOp();

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    /// EmitC ops in the body can omit their 'emitc.' prefix in the assembly.
    static ::llvm::StringRef getDefaultDialect() {
      return "emitc";
    }
  }];
}

def EmitC_ForOp : EmitC_Op<"for",
      [AllTypesMatch<["lowerBound", "upperBound", "step"]>,
       OpAsmOpInterface, SingleBlockImplicitTerminator<"emitc::YieldOp">,
       RecursiveMemoryEffects]> {
  let summary = "For operation";
  let description = [{
    The `emitc.for` operation represents a C loop of the following form:

    ```c++
    for (T i = lb; i < ub; i += step) { /* ... */ } // where T is typeof(lb)
    ```

    The operation takes 3 SSA values as operands that represent the lower bound,
    upper bound and step respectively, and defines an SSA value for its
    induction variable. It has one region capturing the loop body. The induction
    variable is represented as an argument of this region. This SSA value is a
    signless integer, or an index. The step is a value of same type.

    This operation has no result. The body region must contain exactly one block
    that terminates with `emitc.yield`. Calling ForOp::build will create such a
    region and insert the terminator implicitly if none is defined, so will the
    parsing even in cases when it is absent from the custom format. For example:

    ```mlir
    // Index case.
    emitc.for %iv = %lb to %ub step %step {
      ... // body
    }
    ...
    // Integer case.
    emitc.for %iv_32 = %lb_32 to %ub_32 step %step_32 : i32 {
      ... // body
    }
    ```
  }];
  let arguments = (ins IntegerIndexOrOpaqueType:$lowerBound,
                       IntegerIndexOrOpaqueType:$upperBound,
                       IntegerIndexOrOpaqueType:$step);
  let results = (outs);
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$lowerBound, "Value":$upperBound, "Value":$step,
      CArg<"function_ref<void(OpBuilder &, Location, Value)>", "nullptr">)>
  ];

  let extraClassDeclaration = [{
    using BodyBuilderFn =
        function_ref<void(OpBuilder &, Location, Value)>;
    Value getInductionVar() { return getBody()->getArgument(0); }
    void setLowerBound(Value bound) { getOperation()->setOperand(0, bound); }
    void setUpperBound(Value bound) { getOperation()->setOperand(1, bound); }
    void setStep(Value step) { getOperation()->setOperand(2, step); }

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    /// EmitC ops in the body can omit their 'emitc.' prefix in the assembly.
    static ::llvm::StringRef getDefaultDialect() {
      return "emitc";
    }
  }];

  let hasCanonicalizer = 1;
  let hasCustomAssemblyFormat = 1;
  let hasRegionVerifier = 1;
}

def EmitC_CallOp : EmitC_Op<"call",
    [CallOpInterface, CExpression,
     DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Call operation";
  let description = [{
    The `emitc.call` operation represents a direct call to an `emitc.func`
    that is within the same symbol scope as the call. The operands and result type
    of the call must match the specified function type. The callee is encoded as a
    symbol reference attribute named "callee".

    Example:

    ```mlir
    %2 = emitc.call @my_add(%0, %1) : (f32, f32) -> f32
    ```
  }];
  let arguments = (ins 
    FlatSymbolRefAttr:$callee,
    Variadic<EmitCType>:$operands,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );

  let results = (outs Variadic<EmitCType>);

  let builders = [
    OpBuilder<(ins "FuncOp":$callee, CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", SymbolRefAttr::get(callee));
      $_state.addTypes(callee.getFunctionType().getResults());
    }]>,
    OpBuilder<(ins "SymbolRefAttr":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", callee);
      $_state.addTypes(results);
    }]>,
    OpBuilder<(ins "StringAttr":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, SymbolRefAttr::get(callee), results, operands);
    }]>,
    OpBuilder<(ins "StringRef":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, StringAttr::get($_builder.getContext(), callee),
            results, operands);
    }]>];

  let extraClassDeclaration = [{
    FunctionType getCalleeType();

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    MutableOperandRange getArgOperandsMutable() {
      return getOperandsMutable();
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }

    /// Set the callee for this operation.
    void setCalleeFromCallable(CallInterfaceCallable callee) {
      (*this)->setAttr("callee", cast<SymbolRefAttr>(callee));
    }
  }];

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}

def EmitC_DeclareFuncOp : EmitC_Op<"declare_func", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "An operation to declare a function";
  let description = [{
    The `emitc.declare_func` operation allows to insert a function declaration for an
    `emitc.func` at a specific position. The operation only requires the "callee"
    of the `emitc.func` to be specified as an attribute.

    Example:

    ```mlir
    emitc.declare_func @bar
    emitc.func @foo(%arg0: i32) -> i32 {
      %0 = emitc.call @bar(%arg0) : (i32) -> (i32)
      emitc.return %0 : i32
    }

    emitc.func @bar(%arg0: i32) -> i32 {
      emitc.return %arg0 : i32
    }
    ```

    ```c++
    // Code emitted for the operations above.
    int32_t bar(int32_t v1);
    int32_t foo(int32_t v1) {
      int32_t v2 = bar(v1);
      return v2;
    }

    int32_t bar(int32_t v1) {
      return v1;
    }
    ```
  }];
  let arguments = (ins FlatSymbolRefAttr:$sym_name);
  let assemblyFormat = [{
    $sym_name attr-dict
  }];
}

def EmitC_FuncOp : EmitC_Op<"func", [
  AutomaticAllocationScope,
  FunctionOpInterface, IsolatedFromAbove, OpAsmOpInterface
]> {
  let summary = "An operation with a name containing a single `SSACFG` region";
  let description = [{
    Operations within the function cannot implicitly capture values defined
    outside of the function, i.e. Functions are `IsolatedFromAbove`. All
    external references must use function arguments or attributes that establish
    a symbolic connection (e.g. symbols referenced by name via a string
    attribute like SymbolRefAttr). While the MLIR textual form provides a nice
    inline syntax for function arguments, they are internally represented as
    “block arguments” to the first block in the region.

    Only dialect attribute names may be specified in the attribute dictionaries
    for function arguments, results, or the function itself.

    Example:

    ```mlir
    // A function with no results:
    emitc.func @foo(%arg0 : i32) {
      emitc.call_opaque "bar" (%arg0) : (i32) -> ()
      emitc.return
    }

    // A function with its argument as single result:
    emitc.func @foo(%arg0 : i32) -> i32 {
      emitc.return %arg0 : i32
    }

    // A function with specifiers attribute:
    emitc.func @example_specifiers_fn_attr() -> i32
                attributes {specifiers = ["static","inline"]} {
      %0 = emitc.call_opaque "foo" (): () -> i32
      emitc.return %0 : i32
    }

    // An external function definition:
    emitc.func private @extern_func(i32)
                        attributes {specifiers = ["extern"]}
    ```
  }];
  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<StrArrayAttr>:$specifiers,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);
  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
    CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)
  >];
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the region on the current operation that is callable. This may
    /// return null in the case of an external callable object, e.g. an external
    /// function.
    ::mlir::Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    /// EmitC ops in the body can omit their 'emitc.' prefix in the assembly.
    static ::llvm::StringRef getDefaultDialect() {
      return "emitc";
    }
  }];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def EmitC_ReturnOp : EmitC_Op<"return", [Pure, HasParent<"FuncOp">,
                                ReturnLike, Terminator]> {
  let summary = "Function return operation";
  let description = [{
    The `emitc.return` operation represents a return operation within a function.
    The operation takes zero or exactly one operand and produces no results.
    The operand number and type must match the signature of the function
    that contains the operation.

    Example:

    ```mlir
    emitc.func @foo() -> (i32) {
      ...
      emitc.return %0 : i32
    }
    ```
  }];
  let arguments = (ins Optional<EmitCType>:$operand);

  let assemblyFormat = "attr-dict ($operand^ `:` type($operand))?";
  let hasVerifier = 1;
}

def EmitC_IncludeOp
    : EmitC_Op<"include", []> {
  let summary = "Include operation";
  let description = [{
    The `emitc.include` operation allows to define a source file inclusion via the
    `#include` directive.

    Example:

    ```mlir
    // Custom form defining the inclusion of `<myheader>`.
    emitc.include <"myheader.h">

    // Generic form of the same operation.
    "emitc.include" (){include = "myheader.h", is_standard_include} : () -> ()

    // Custom form defining the inclusion of `"myheader"`.
    emitc.include "myheader.h"

    // Generic form of the same operation.
    "emitc.include" (){include = "myheader.h"} : () -> ()
    ```
  }];
  let arguments = (ins
    Arg<StrAttr, "source file to include">:$include,
    UnitAttr:$is_standard_include
  );
  let hasCustomAssemblyFormat = 1;
}

def EmitC_LiteralOp : EmitC_Op<"literal", [Pure]> {
  let summary = "Literal operation";
  let description = [{
    The `emitc.literal` operation produces an SSA value equal to some constant
    specified by an attribute.
  }];

  let arguments = (ins StrAttr:$value);
  let results = (outs EmitCType:$result);

  let hasVerifier = 1;
  let assemblyFormat = "$value attr-dict `:` type($result)";
}

def EmitC_LogicalAndOp : EmitC_BinaryOp<"logical_and", [CExpression]> {
  let summary = "Logical and operation";
  let description = [{
    With the `emitc.logical_and` operation the logical operator && (and) can
    be applied.

    Example:

    ```mlir
    %0 = emitc.logical_and %arg0, %arg1 : i32, i32
    ```
    ```c++
    // Code emitted for the operation above.
    bool v3 = v1 && v2;
    ```
  }];

  let results = (outs I1);
  let assemblyFormat = "operands attr-dict `:` type(operands)";
}

def EmitC_LogicalNotOp : EmitC_UnaryOp<"logical_not", [CExpression]> {
  let summary = "Logical not operation";
  let description = [{
    With the `emitc.logical_not` operation the logical operator ! (negation) can
    be applied.

    Example:

    ```mlir
    %0 = emitc.logical_not %arg0 : i32
    ```
    ```c++
    // Code emitted for the operation above.
    bool v2 = !v1;
    ```
  }];

  let results = (outs I1);
  let assemblyFormat = "operands attr-dict `:` type(operands)";
}

def EmitC_LogicalOrOp : EmitC_BinaryOp<"logical_or", [CExpression]> {
  let summary = "Logical or operation";
  let description = [{
    With the `emitc.logical_or` operation the logical operator || (inclusive or)
    can be applied.

    Example:

    ```mlir
    %0 = emitc.logical_or %arg0, %arg1 : i32, i32
    ```
    ```c++
    // Code emitted for the operation above.
    bool v3 = v1 || v2;
    ```
  }];

  let results = (outs I1);
  let assemblyFormat = "operands attr-dict `:` type(operands)";
}

def EmitC_LoadOp : EmitC_Op<"load", [
  TypesMatchWith<"result type matches value type of 'operand'",
                  "operand", "result",
                  "::llvm::cast<LValueType>($_self).getValueType()">
]> {
  let summary = "Load an lvalue into an SSA value.";
  let description = [{
    This operation loads the content of a modifiable lvalue into an SSA value. 
    Modifications of the lvalue executed after the load are not observable on 
    the produced value.

    Example:

    ```mlir
    %1 = emitc.load %0 : !emitc.lvalue<i32>
    ```
    ```c++
    // Code emitted for the operation above.
    int32_t v2 = v1;
    ```
  }];

  let arguments = (ins 
      Res<EmitC_LValueType, "", [MemRead<DefaultResource, 0, FullEffect>]>:$operand);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$operand attr-dict `:` type($operand)"; 
}

def EmitC_MulOp : EmitC_BinaryOp<"mul", [CExpression]> {
  let summary = "Multiplication operation";
  let description = [{
    With the `emitc.mul` operation the arithmetic operator * (multiplication) can
    be applied.

    Example:

    ```mlir
    // Custom form of the multiplication operation.
    %0 = emitc.mul %arg0, %arg1 : (i32, i32) -> i32
    %1 = emitc.mul %arg2, %arg3 : (f32, f32) -> f32
    ```
    ```c++
    // Code emitted for the operations above.
    int32_t v5 = v1 * v2;
    float v6 = v3 * v4;
    ```
  }];

  let arguments = (ins FloatIntegerIndexOrOpaqueType, FloatIntegerIndexOrOpaqueType);
  let results = (outs FloatIntegerIndexOrOpaqueType);
}

def EmitC_RemOp : EmitC_BinaryOp<"rem", [CExpression]> {
  let summary = "Remainder operation";
  let description = [{
    With the `emitc.rem` operation the arithmetic operator % (remainder) can
    be applied.

    Example:

    ```mlir
    // Custom form of the remainder operation.
    %0 = emitc.rem %arg0, %arg1 : (i32, i32) -> i32
    ```
    ```c++
    // Code emitted for the operation above.
    int32_t v5 = v1 % v2;
    ```
  }];

  let arguments = (ins IntegerIndexOrOpaqueType, IntegerIndexOrOpaqueType);
  let results = (outs IntegerIndexOrOpaqueType);
}

def EmitC_SubOp : EmitC_BinaryOp<"sub", [CExpression]> {
  let summary = "Subtraction operation";
  let description = [{
    With the `emitc.sub` operation the arithmetic operator - (subtraction) can
    be applied.

    Example:

    ```mlir
    // Custom form of the substraction operation.
    %0 = emitc.sub %arg0, %arg1 : (i32, i32) -> i32
    %1 = emitc.sub %arg2, %arg3 : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
    %2 = emitc.sub %arg4, %arg5 : (!emitc.ptr<i32>, !emitc.ptr<i32>)
        -> !emitc.ptrdiff_t
    ```
    ```c++
    // Code emitted for the operations above.
    int32_t v7 = v1 - v2;
    float* v8 = v3 - v4;
    ptrdiff_t v9 = v5 - v6;
    ```
  }];

  let hasVerifier = 1;
}

def EmitC_MemberOp : EmitC_Op<"member"> {
  let summary = "Member operation";
  let description = [{
    With the `emitc.member` operation the member access operator `.` can be
    applied.

    Example:

    ```mlir
    %0 = "emitc.member" (%arg0) {member = "a"}
        : (!emitc.lvalue<!emitc.opaque<"mystruct">>) -> !emitc.lvalue<i32>
    ```
  }];

  let arguments = (ins
    Arg<StrAttr, "the member to access">:$member,
    EmitC_LValueOf<[EmitC_OpaqueType]>:$operand
  );
  let results = (outs EmitC_LValueOf<[EmitCType]>);
}

def EmitC_MemberOfPtrOp : EmitC_Op<"member_of_ptr"> {
  let summary = "Member of pointer operation";
  let description = [{
    With the `emitc.member_of_ptr` operation the member access operator `->`
    can be applied.

    Example:

    ```mlir
    %0 = "emitc.member_of_ptr" (%arg0) {member = "a"}
        : (!emitc.lvalue<!emitc.ptr<!emitc.opaque<"mystruct">>>)
        -> !emitc.lvalue<i32>
    ```
  }];

  let arguments = (ins
    Arg<StrAttr, "the member to access">:$member,
    EmitC_LValueOf<[EmitC_OpaqueType,EmitC_PointerType]>:$operand
  );
  let results = (outs EmitC_LValueOf<[EmitCType]>);
}

def EmitC_ConditionalOp : EmitC_Op<"conditional",
    [AllTypesMatch<["true_value", "false_value", "result"]>, CExpression]> {
  let summary = "Conditional (ternary) operation";
  let description = [{
    With the `emitc.conditional` operation the ternary conditional operator can
    be applied.

    Example:

    ```mlir
    %0 = emitc.cmp gt, %arg0, %arg1 : (i32, i32) -> i1

    %c0 = "emitc.constant"() {value = 10 : i32} : () -> i32
    %c1 = "emitc.constant"() {value = 11 : i32} : () -> i32

    %1 = emitc.conditional %0, %c0, %c1 : i32
    ```
    ```c++
    // Code emitted for the operations above.
    bool v3 = v1 > v2;
    int32_t v4 = 10;
    int32_t v5 = 11;
    int32_t v6 = v3 ? v4 : v5;
    ```
  }];
  let arguments = (ins I1:$condition, EmitCType:$true_value, EmitCType:$false_value);
  let results = (outs EmitCType:$result);
  let assemblyFormat = "operands attr-dict `:` type($result)";
}

def EmitC_UnaryMinusOp : EmitC_UnaryOp<"unary_minus", [CExpression]> {
  let summary = "Unary minus operation";
  let description = [{
    With the `emitc.unary_minus` operation the unary operator - (minus) can be
    applied.

    Example:

    ```mlir
    %0 = emitc.unary_minus %arg0 : (i32) -> i32
    ```
    ```c++
    // Code emitted for the operation above.
    int32_t v2 = -v1;
    ```
  }];
}

def EmitC_UnaryPlusOp : EmitC_UnaryOp<"unary_plus", [CExpression]> {
  let summary = "Unary plus operation";
  let description = [{
    With the `emitc.unary_plus` operation the unary operator + (plus) can be
    applied.

    Example:

    ```mlir
    %0 = emitc.unary_plus %arg0 : (i32) -> i32
    ```
    ```c++
    // Code emitted for the operation above.
    int32_t v2 = +v1;
    ```
  }];
}

def EmitC_VariableOp : EmitC_Op<"variable", []> {
  let summary = "Variable operation";
  let description = [{
    The `emitc.variable` operation produces an SSA value equal to some value
    specified by an attribute. This can be used to form simple integer and
    floating point variables, as well as more exotic things like tensor
    variables. The `emitc.variable` operation also supports the EmitC opaque
    attribute and the EmitC opaque type. If further supports the EmitC
    pointer type, whereas folding is not supported.
    The `emitc.variable` is emitted as a C/C++ local variable.

    Example:

    ```mlir
    // Integer variable
    %0 = "emitc.variable"(){value = 42 : i32} : () -> !emitc.lvalue<i32>

    // Variable emitted as `int32_t* = NULL;`
    %1 = "emitc.variable"() {value = #emitc.opaque<"NULL">} 
      : () -> !emitc.lvalue<!emitc.ptr<!emitc.opaque<"int32_t">>>
    ```

    Since folding is not supported, it can be used with pointers.
    As an example, it is valid to create pointers to `variable` operations
    by using `apply` operations and pass these to a `call` operation.
    ```mlir
    %0 = "emitc.variable"() {value = 0 : i32} : () -> !emitc.lvalue<i32>
    %1 = "emitc.variable"() {value = 0 : i32} : () -> !emitc.lvalue<i32>
    %2 = emitc.apply "&"(%0) : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>
    %3 = emitc.apply "&"(%1) : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>
    emitc.call_opaque "write"(%2, %3)
      : (!emitc.ptr<i32>, !emitc.ptr<i32>) -> ()
    ```
  }];

  let arguments = (ins EmitC_OpaqueOrTypedAttr:$value);
  let results = (outs Res<AnyTypeOf<[EmitC_ArrayType, EmitC_LValueType]>, "",
                          [MemAlloc<DefaultResource, 0, FullEffect>]>);

  let hasVerifier = 1;
}

def EmitC_GlobalOp : EmitC_Op<"global", [Symbol]> {
  let summary = "A global variable";
  let description = [{
    The `emitc.global` operation declares or defines a named global variable.
    The backing memory for the variable is allocated statically and is
    described by the type of the variable.
    Optionally, an `initial_value` can be provided.
    Internal linkage can be specified using the `static_specifier` unit attribute
    and external linkage can be specified using the `extern_specifier` unit attribute.
    Note that the default linkage without those two keywords depends on whether
    the target is C or C++ and whether the global variable is `const`.
    The global variable can also be marked constant using the `const_specifier`
    unit attribute. Writing to such constant global variables is
    undefined.

    The global variable can be accessed by using the `emitc.get_global` to
    retrieve the value for the global variable.

    Example:

    ```mlir
    // Global variable with an initial value.
    emitc.global @x : !emitc.array<2xf32> = dense<0.0>
    // Global variable with an initial values.
    emitc.global @x : !emitc.array<3xi32> = dense<[0, 1, 2]>
    // External global variable
    emitc.global extern @x : !emitc.array<2xf32>
    // Constant global variable with internal linkage
    emitc.global static const @x : i32 = 0
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttr:$type,
                       OptionalAttr<EmitC_OpaqueOrTypedAttr>:$initial_value,
                       UnitAttr:$extern_specifier,
                       UnitAttr:$static_specifier,
                       UnitAttr:$const_specifier);

  let assemblyFormat = [{
       (`extern` $extern_specifier^)?
       (`static` $static_specifier^)?
       (`const` $const_specifier^)?
       $sym_name
       `:` custom<EmitCGlobalOpTypeAndInitialValue>($type, $initial_value)
       attr-dict
  }];

  let hasVerifier = 1;
}

def EmitC_GetGlobalOp : EmitC_Op<"get_global",
    [Pure, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Obtain access to a global variable";
  let description = [{
     The `emitc.get_global` operation retrieves the lvalue of a
     named global variable. If the global variable is marked constant, assigning
     to that lvalue is undefined.

     Example:

     ```mlir
     %x = emitc.get_global @foo : !emitc.array<2xf32>
     %y = emitc.get_global @bar : !emitc.lvalue<i32>
     ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$name);
  let results = (outs AnyTypeOf<[EmitC_ArrayType, EmitC_LValueType]>:$result);
  let assemblyFormat = "$name `:` type($result) attr-dict";
}

def EmitC_VerbatimOp : EmitC_Op<"verbatim"> {
  let summary = "Verbatim operation";
  let description = [{
    The `emitc.verbatim` operation produces no results and the value is emitted as is
    followed by a line break  ('\n' character) during translation.

    Note: Use with caution. This operation can have arbitrary effects on the
    semantics of the emitted code. Use semantically more meaningful operations
    whenever possible. Additionally this op is *NOT* intended to be used to
    inject large snippets of code.

    This operation can be used in situations where a more suitable operation is
    not yet implemented in the dialect or where preprocessor directives
    interfere with the structure of the code. One example of this is to declare
    the linkage of external symbols to make the generated code usable in both C
    and C++ contexts:

    ```c++
    #ifdef __cplusplus
    extern "C" {
    #endif

    ...
    
    #ifdef __cplusplus
    }
    #endif
    ```
  }];

  let arguments = (ins StrAttr:$value);
  let assemblyFormat = "$value attr-dict";
}

def EmitC_AssignOp : EmitC_Op<"assign", []> {
  let summary = "Assign operation";
  let description = [{
    The `emitc.assign` operation stores an SSA value to the location designated by an
    EmitC variable. This operation doesn't return any value. The assigned value
    must be of the same type as the variable being assigned. The operation is
    emitted as a C/C++ '=' operator.

    Example:

    ```mlir
    // Integer variable
    %0 = "emitc.variable"(){value = 42 : i32} : () -> !emitc.lvalue<i32>
    %1 = emitc.call_opaque "foo"() : () -> (i32)

    // Assign emitted as `... = ...;`
    "emitc.assign"(%0, %1) : (!emitc.lvalue<i32>, i32) -> ()
    ```
  }];

  let arguments = (ins 
      Res<EmitC_LValueType, "", [MemWrite<DefaultResource, 1, FullEffect>]>:$var,
      EmitCType:$value);
  let results = (outs);

  let hasVerifier = 1;
  let assemblyFormat = "$value `:` type($value) `to` $var `:` type($var) attr-dict";
}

def EmitC_YieldOp : EmitC_Op<"yield",
      [Pure, Terminator, ParentOneOf<["ExpressionOp", "IfOp", "ForOp", "SwitchOp"]>]> {
  let summary = "Block termination operation";
  let description = [{
    The `emitc.yield` terminates its parent EmitC op's region, optionally yielding
    an SSA value. The semantics of how the values are yielded is defined by the
    parent operation.
    If `emitc.yield` has an operand, the operand must match the parent operation's
    result. If the parent operation defines no values, then the `emitc.yield`
    may be left out in the custom syntax and the builders will insert one
    implicitly. Otherwise, it has to be present in the syntax to indicate which
    value is yielded.
  }];

  let arguments = (ins Optional<EmitCType>:$result);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let hasVerifier = 1;
  let assemblyFormat = [{ attr-dict ($result^ `:` type($result))? }];
}

def EmitC_IfOp : EmitC_Op<"if",
    [DeclareOpInterfaceMethods<RegionBranchOpInterface, [
    "getNumRegionInvocations", "getRegionInvocationBounds",
    "getEntrySuccessorRegions"]>, OpAsmOpInterface, SingleBlock,
    SingleBlockImplicitTerminator<"emitc::YieldOp">,
    RecursiveMemoryEffects, NoRegionArguments]> {
  let summary = "If-then-else operation";
  let description = [{
    The `emitc.if` operation represents an if-then-else construct for
    conditionally executing two regions of code. The operand to an if operation
    is a boolean value. For example:

    ```mlir
    emitc.if %b  {
      ...
    } else {
      ...
    }
    ```

    The "then" region has exactly 1 block. The "else" region may have 0 or 1
    blocks. The blocks are always terminated with `emitc.yield`, which can be
    left out to be inserted implicitly. This operation doesn't produce any
    results.
  }];
  let arguments = (ins I1:$condition);
  let results = (outs);
  let regions = (region SizedRegion<1>:$thenRegion,
                        MaxSizedRegion<1>:$elseRegion);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$cond)>,
    OpBuilder<(ins "Value":$cond, "bool":$addThenBlock, "bool":$addElseBlock)>,
    OpBuilder<(ins "Value":$cond, "bool":$withElseRegion)>,
    OpBuilder<(ins "Value":$cond,
      CArg<"function_ref<void(OpBuilder &, Location)>",
           "buildTerminatedBody">:$thenBuilder,
      CArg<"function_ref<void(OpBuilder &, Location)>",
           "nullptr">:$elseBuilder)>,
  ];

  let extraClassDeclaration = [{
    OpBuilder getThenBodyBuilder(OpBuilder::Listener *listener = nullptr) {
      Block* body = getBody(0);
      return OpBuilder::atBlockEnd(body, listener);
    }
    OpBuilder getElseBodyBuilder(OpBuilder::Listener *listener = nullptr) {
      Block* body = getBody(1);
      return OpBuilder::atBlockEnd(body, listener);
    }

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    /// EmitC ops in the body can omit their 'emitc.' prefix in the assembly.
    static ::llvm::StringRef getDefaultDialect() {
      return "emitc";
    }
  }];
  let hasCustomAssemblyFormat = 1;
}

def EmitC_SubscriptOp : EmitC_Op<"subscript", []> {
  let summary = "Subscript operation";
  let description = [{
    With the `emitc.subscript` operation the subscript operator `[]` can be applied
    to variables or arguments of array, pointer and opaque type.

    Example:

    ```mlir
    %i = index.constant 1
    %j = index.constant 7
    %0 = emitc.subscript %arg0[%i, %j] : (!emitc.array<4x8xf32>, index, index)
           -> !emitc.lvalue<f32>
    %1 = emitc.subscript %arg1[%i] : (!emitc.ptr<i32>, index)
           -> !emitc.lvalue<i32>
    ```
  }];
  let arguments = (ins Arg<AnyTypeOf<[
      EmitC_ArrayType,
      EmitC_OpaqueType,
      EmitC_PointerType]>,
    "the value to subscript">:$value,
    Variadic<EmitCType>:$indices);
  let results = (outs EmitC_LValueType:$result);

  let builders = [
    OpBuilder<(ins "TypedValue<ArrayType>":$array, "ValueRange":$indices), [{
      build(
        $_builder,
        $_state,
        emitc::LValueType::get(array.getType().getElementType()),
        array,
        indices
      );
    }]>,
    OpBuilder<(ins "TypedValue<PointerType>":$pointer, "Value":$index), [{
      build(
        $_builder,
        $_state,
        emitc::LValueType::get(pointer.getType().getPointee()),
        pointer,
        ValueRange{index}
      );
    }]>
  ];

  let hasVerifier = 1;
  let assemblyFormat = "$value `[` $indices `]` attr-dict `:` functional-type(operands, results)";
}

def EmitC_SwitchOp : EmitC_Op<"switch", [RecursiveMemoryEffects,
    OpAsmOpInterface, SingleBlockImplicitTerminator<"emitc::YieldOp">,
    DeclareOpInterfaceMethods<RegionBranchOpInterface,
                              ["getRegionInvocationBounds",
                               "getEntrySuccessorRegions"]>]> {
  let summary = "Switch operation";
  let description = [{
    The `emitc.switch` is a control-flow operation that branches to one of
    the given regions based on the values of the argument and the cases.
    The operand to a switch operation is a opaque, integral or pointer
    wide types.

    The operation always has a "default" region and any number of case regions
    denoted by integer constants. Control-flow transfers to the case region
    whose constant value equals the value of the argument. If the argument does
    not equal any of the case values, control-flow transfer to the "default"
    region.

    The operation does not return any value. Moreover, case regions must be
    explicitly terminated using the `emitc.yield` operation. Default region is
    yielded implicitly.

    Example:

    ```mlir
    // Example:
    emitc.switch %0 : i32
    case 2 {
      %1 = emitc.call_opaque "func_b" () : () -> i32
      emitc.yield
    }
    case 5 {
      %2 = emitc.call_opaque "func_a" () : () -> i32
      emitc.yield
    }
    default {
      %3 = "emitc.constant"(){value = 42.0 : f32} : () -> f32
      emitc.call_opaque "func2" (%3) : (f32) -> ()
    }
    ```
    ```c++
    // Code emitted for the operations above.
    switch (v1) {
    case 2: {
      int32_t v2 = func_b();
      break;
    }
    case 5: {
      int32_t v3 = func_a();
      break;
    }
    default: {
      float v4 = 4.200000000e+01f;
      func2(v4);
      break;
    }
    }
    ```
  }];

  let arguments = (ins IntegerIndexOrOpaqueType:$arg, DenseI64ArrayAttr:$cases);
  let results = (outs);
  let regions = (region SizedRegion<1>:$defaultRegion,
                        VariadicRegion<SizedRegion<1>>:$caseRegions);

  let assemblyFormat = [{
    $arg `:` type($arg) attr-dict custom<SwitchCases>($cases, $caseRegions) `\n`
    `` `default` $defaultRegion
  }];

  let extraClassDeclaration = [{
    /// Get the number of cases.
    unsigned getNumCases();

    /// Get the default region body.
    Block &getDefaultBlock();

    /// Get the body of a case region.
    Block &getCaseBlock(unsigned idx);

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    /// EmitC ops in the body can omit their 'emitc.' prefix in the assembly.
    static ::llvm::StringRef getDefaultDialect() {
      return "emitc";
    }
  }];

  let hasVerifier = 1;
}

#endif // MLIR_DIALECT_EMITC_IR_EMITC
