//===- Ops.td - Toy dialect operation definitions ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the operations of the Toy dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TOY_OPS
#define TOY_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Provide a definition of the 'toy' dialect in the ODS framework so that we
// can define our operations.

// Cratels: 定义一个名为Toy_Dialect的dialect并在其中定义自己的operation
// Cratels: tablegen模块会自动将其转换为 C++源码形式，其名称会被转换为 ToyDialect
def Toy_Dialect : Dialect {
  let name = "toy";
  // Cratels: 指定C++源码的命名空间
  let cppNamespace = "::mlir::toy";
}


// Base class for toy dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.===父类所在的dialect：Toy_Dialect
//   * The mnemonic for the operation, or the name without the dialect prefix.===mnemonic：助记符，或者不带dialect前缀的名字
//   * A list of traits for the operation.===该operation的特征值列表
//
// Cratels: mnemonic助记符：
// Cratels: trait 特征值：
// Cratels: 定义toy dialect的Operation基类，这个operation从基类Op继承而来
class Toy_Op<string mnemonic, list<Trait> traits = []> :
    Op<Toy_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Toy Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

// We define a toy operation by inheriting from our base 'Toy_Op' class above.
// Cratels: 通过继承Toy_Op来定义一个operation ConstantOp，用来表示常量。
// Here we provide the mnemonic and a list of traits for the operation. The
// constant operation is marked as 'Pure' as it is a pure operation
// and may be removed if dead.
// Cratels: Pure 意味着没有side effect副作用，如果不会访问到就可以大胆的删除
def ConstantOp : Toy_Op<"constant", [Pure]> {
  // Provide a summary and description for this operation. This can be used to
  // auto-generate documentation of the operations within our dialect.
  let summary = "constant";

  let description = [{
    Constant operation turns a literal into an SSA value.
    // Cratels:  The data is attached to the operation as an attribute.
    For example:

    ```mlir
      %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>
                        : tensor<2x3xf64>
    ```
  }];

  // Cratels: ins代表op的输入值列表。outs代表op的输出列表
  // Cratels: op是用来进行转换的，所谓转换就是将来源Source转为目标值Target。ins就是指这里的Source来源，outs就是输出
  //
  //
  // Cratels: The constant operation takes an attribute as the only input.
  // Cratels: 常量必须有值，这个op的唯一属性就是常量值。
  // Cratels: F64ElementsAttr为attribute的类型，参数名称为value
  let arguments = (ins F64ElementsAttr:$value);

  // The constant operation returns a single value of TensorType.
  // Cratels: 输出类型为F64Tensor，即元素类型为F64的tensor类型
  let results = (outs F64Tensor);

  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;

  // Add custom build methods for the constant operation. These method populates
  // the `state` that MLIR uses to create operations, i.e. these are used when
  // using `builder.create<ConstantOp>(...)`.
  let builders = [
    // Build a constant with a given constant tensor value.
    OpBuilder<(ins "DenseElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,

    // Build a constant with a given constant floating-point value.
    OpBuilder<(ins "double":$value)>
  ];

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

// Cratels:从 Toy_Op 中派生出 AddOp 代表加法二元表达式，别名为 add
def AddOp : Toy_Op<"add"> {
  let summary = "element-wise addition operation";
  let description = [{
    The "add" operation performs element-wise addition between two tensors.
    The shapes of the tensor operands are expected to match.
    // Cratels:两个执行加法的 tensor 要求唯独完全一致
  }];

  // Cratels:输入为两个 tensor
  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  // Cratels:输出为一个 tensor
  let results = (outs F64Tensor);

  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;

  // Allow building an AddOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

//===----------------------------------------------------------------------===//
// FuncOp 方法定义的operation
//===----------------------------------------------------------------------===//
// Cratels:用户自定义方法的 op，别名即为 func，trait 特征值为FunctionOpInterface, IsolatedFromAbove，暂不清楚具体意义
// Cratels:代表用户定义的方法
def FuncOp : Toy_Op<"func", [
    FunctionOpInterface, IsolatedFromAbove
  ]> {
  let summary = "user defined function operation";
  let description = [{
    The "toy.func" operation represents a user defined function. These are
    callable SSA-region operations that contain toy computations.

    Example:

    ```mlir
    toy.func @main() {
      %0 = toy.constant dense<5.500000e+00> : tensor<f64>
      %1 = toy.reshape(%0 : tensor<f64>) to tensor<2x2xf64>
      toy.print %1 : tensor<2x2xf64>
      toy.return
    }
    ```
  }];

  let arguments = (ins
    // Cratels:符号表中的符号名称
    SymbolNameAttr:$sym_name,
    // Cratels:方法类型
    TypeAttrOf<FunctionType>:$function_type,
    // Cratels:入参列表
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    // Cratels:返回值列表
    OptionalAttr<DictArrayAttr>:$res_attrs
  );

  // Cratels:方法体，其类型为 region
  let regions = (region AnyRegion:$body);

  // Cratels:CArg 的作用是指当前即为 C 类型，参数直接将其参数拷贝过去即可
  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
  >];


  // Cratels:有些方法可能无法使用 tablegen 来生成，可以使用这个属性直接将其书写下来。
  // Cratels:在进行tablegen 的时候，这里的代码会被原样拷贝进生成的 cpp 文件中
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    Region *getCallableRegion() { return &getBody(); }
  }];

  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// GenericCallOp
//===----------------------------------------------------------------------===//

// Cratels:方法调用的 op
def GenericCallOp : Toy_Op<"generic_call"> {
  let summary = "generic call operation";
  let description = [{
    Generic calls represent calls to a user defined function that needs to
    be specialized for the shape of its arguments. The callee name is attached
    as a symbol reference via an attribute. The arguments list must match the
    arguments expected by the callee. For example:

    ```mlir
     %4 = toy.generic_call @my_func(%1, %3)
           : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
    ```

    This is only valid if a function named "my_func" exists and takes two
    arguments.
  }];
  // Cratels:当我们调用方法的时候，只需要知道其方法名以及参数列表就可以了
  // Cratels:方法名
  // Cratels:方法参数使用可变容器 dic 字典来实现
  // The generic call operation takes a symbol reference attribute as the
  // callee, and inputs for the call.
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<F64Tensor>:$inputs);

  // The generic call operation returns a single value of TensorType.
  // Cratels:为了简化，Toy中所有的方法返回值都是固定的为F64Tensor
  let results = (outs F64Tensor);

  // Specialize assembly printing and parsing using a declarative format.
  // Cratels: 指定当前op输出时候的格式。这里我为其添加了cratels，这后续再进行MLIR的输出中所有的方法调用中都会增加这个后缀（无意义，仅供测试）
  let assemblyFormat = [{
    $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)`cratels`
  }];

  // Add custom build methods for the generic call operation.
  let builders = [
    OpBuilder<(ins "StringRef":$callee, "ArrayRef<Value>":$arguments)>
  ];
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//
// Cratels:乘法 op
def MulOp : Toy_Op<"mul"> {
  let summary = "element-wise multiplication operation";
  let description = [{
    The "mul" operation performs element-wise multiplication between two
    tensors. The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);

  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;

  // Allow building a MulOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

//===----------------------------------------------------------------------===//
// PrintOp
//===----------------------------------------------------------------------===//
// Cratels:信息打印 op
def PrintOp : Toy_Op<"print"> {
  let summary = "print operation";
  let description = [{
    The "print" builtin operation prints a given input tensor, and produces
    no results.
  }];

  // The print operation takes an input tensor to print.
  let arguments = (ins F64Tensor:$input);

  let assemblyFormat = "$input attr-dict `:` type($input)";
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//
// Cratels:reshape，重新塑造 tensor 维度
def ReshapeOp : Toy_Op<"reshape"> {
  let summary = "tensor reshape operation";
  let description = [{
    Reshape operation is transforming its input tensor into a new tensor with
    the same number of elements but different shapes. For example:

    ```mlir
       %0 = toy.reshape (%arg1 : tensor<10xf64>) to tensor<5x2xf64>
    ```
  }];

  let arguments = (ins F64Tensor:$input);

  // We expect that the reshape operation returns a statically shaped tensor.
  let results = (outs StaticShapeTensorOf<[F64]>);

  let assemblyFormat = [{
    `(` $input `:` type($input) `)` attr-dict `to` type(results)
  }];
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def ReturnOp : Toy_Op<"return", [Pure, HasParent<"FuncOp">,
                                 Terminator]> {
  let summary = "return operation";
  let description = [{
    The "return" operation represents a return operation within a function.
    The operation takes an optional tensor operand and produces no results.
    The operand type must match the signature of the function that contains
    the operation. For example:

    ```mlir
      toy.func @foo() -> tensor<2xf64> {
        ...
        toy.return %0 : tensor<2xf64>
      }
    ```
  }];

  // The return operation takes an optional input operand to return. This
  // value must match the return type of the enclosing function.
  let arguments = (ins Variadic<F64Tensor>:$input);

  // The return operation only emits the input in the format if it is present.
  let assemblyFormat = "($input^ `:` type($input))? attr-dict ";

  // Allow building a ReturnOp with no return operand.
  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt); }]>
  ];

  // Provide extra utility definitions on the c++ operation class definition.
  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];

  // Invoke a static verify method to verify this return operation.
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

// Cratels:转置 tensor，将 M*N 的 tensor 转换为 N*M
def TransposeOp : Toy_Op<"transpose"> {
  let summary = "transpose operation";

  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor);

  let assemblyFormat = [{
    `(` $input `:` type($input) `)` attr-dict `to` type(results)
  }];

  // Allow building a TransposeOp with from the input operand.
  let builders = [
    OpBuilder<(ins "Value":$input)>
  ];

  // Invoke a static verify method to verify this transpose operation.
  let hasVerifier = 1;
}

#endif // TOY_OPS
