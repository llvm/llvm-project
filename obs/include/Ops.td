
#ifndef OBS_OPS
#define OBS_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def OBS_Dialect: Dialect {
    //The namespace of the Dialect.
    let name = "obs";

    let summary = "A high-level dialect for analyzing and optimzing the Language";

    let description = [{
        The Toy language is a tensor-based language that allows you to define functions, perform some math computation, and
        print results. This dialect provides a reprentation of the language that is amenable to analysis and optimization.
    }];

    let cppNamespace = "::mlir::obs";
}

class OBS_op<string mnemonic, list<Trait> traits = []> : Op<OBS_Dialect, mnemonic, traits>;

def ConstantOp : OBS_op<"constant", [Pure]> {

    let summary = "constant operation" ;

    let description = [{
        Constant operation turns a literal into an SSA value. The data is attached to the operation as an attribute.
    }];


    let arguments = (ins F64ElementsAttr:$value);
    let results = (outs F64Tensor);

    let hasCustomAssemblyFormat = 1;

    let hasVerifier = 1;

    let builders = [
        OpBuilder<(ins "DenseElementsAttr":$value), [{build($_builder, $_state, value.getType(), value);}] >,
        OpBuilder<(ins "double":$value)>
    ];

}

def AddOp : OBS_op<"add"> {
    let summary = "element-wise addition operation";
    let description = [{
        The "add" operation performs element-wise addition between two tensors.
        The shape of the tensor operands are expected to match.
    }];

    let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
    let results = (outs F64Tensor);

    let hasCustomAssemblyFormat = 1;

    let builders = [
        OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
    ];
}  

def FuncOp : OBS_op<"func", [FunctionOpInterface, IsolatedFromAbove]> {
    let summary = "user defined function operation";
    let description = [{
        The "obs.func" operation represents a user defined function. These are callable SSA-region operations
        that contain obs computations.
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttrOf<FunctionType>:$function_type,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs
    );
    let regions = (region AnyRegion:$body);

    let builders = [OpBuilder<(ins "StringRef":$name, "FunctionType":$type,
                                CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
                    >];

    let extraClassDeclaration = [{
        ArrayRef<Type> getArgumentTypes() {
            return getFunctionType().getInputs();
        }

        ArrayRef<Type> getResultTypes() {
            return getFunctionType().getResults();
        }

        Region *getCallableRegion() {
            return &getBody();
        }
    }];

    let hasCustomAssemblyFormat = 1;
    let skipDefaultBuilders = 1;
}

def GenericCallOp: OBS_op<"generic_call"> {
    let summary = "generic call operation";
    let description = [{
        Generic calls represent calls to a user defined function that needs to be specialized for the shape
        of its arguments.
    }];

    let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<F64Tensor>:$inputs);

    let results = (outs F64Tensor);

    let assemblyFormat = [{
        $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
    }];

    let builders = [
        OpBuilder<(ins "StringRef":$callee, "ArrayRef<Value>":$arguments)>
    ];
}

def MulOp : OBS_op<"mul"> {
    let summary = "element-wise multiplication operation";

    let description = [{
        The "mul" operation performs element-wise multiplication between two
        tensors. The shapes of the tensor operands are expected to match.
    }];

    let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
    let results = (outs F64Tensor);

    let hasCustomAssemblyFormat = 1;
    let builders = [
        OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
    ];
}

def PrintOp: OBS_op<"print"> {
    let summary = "print operation";
    let description = [{
        The print builtin operation prints a given input sensor, and produces no results.
    }];
    let arguments = (ins F64Tensor:$input);
    let assemblyFormat = "$input attr-dict `:` type($input)";
}

def ReshapeOp: OBS_op<"reshape"> {
    let summary = "tensor reshape operation";
    let description = [{
        Reshape operation is transforming its input sensor into a new tensor with the same number of
        elements but different shapes.
    }];

    let arguments = (ins F64Tensor:$input);
    let results = (outs StaticShapeTensorOf<[F64]>);

    let assemblyFormat = [{
        `(` $input `:` type($input) `)` attr-dict `to` type(results)
    }];
}

def TransposeOp: OBS_op<"transpose"> {
    let summary = "transpose operation";
    let arguments = (ins F64Tensor:$input);
    let results = (outs F64Tensor);

    let assemblyFormat = [{
        `(` $input `:` type($input) `)` attr-dict `to` type(results)
    }];

    let builders = [
        OpBuilder<(ins "Value":$input)>
    ];

    let hasVerifier = 1;
}

def ReturnOp : OBS_op<"return", [Pure, HasParent<"FuncOp">, Terminator]> {
    let summary = "return operation";
    let description = [{
        The "return" operation represents a return operation within a function;
    }];

    let arguments = (ins Variadic<F64Tensor>:$input);
    let assemblyFormat = "($input^ `:` type($input))? attr-dict ";
 
    let builders = [
        OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt); }]>
    ];

    let extraClassDeclaration = [{
        bool hasOperand() {
            return getNumOperands() != 0;
        }
    }];

    let hasVerifier = 1;
}


#endif //OBS_OPS
