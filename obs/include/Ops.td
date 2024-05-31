
#ifndef OBS_OPS
#define OBS_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def OBS_Dialect: Dialect {
    //The namespace of the Dialect.
    let name = "obs";

    let summary = "A dialect for representing abstract memory operations.";

    let description = [{
        The OBS dialect is an abstraction over memory operations. 
        It is used to reason about Ownership and Borrowing in a programming language.
        It contains the operations:
        (1) OwnOp(type, Dim);
        (2) RefOp(x)
        (3) Read(x);
        (4) Write(x);
        (5) delete(x); 
        (6) Function;
    }];

    let cppNamespace = "::mlir::obs";

    let useDefaultTypePrinterParser = 1;
}

class OBS_op<string mnemonic, list<Trait> traits = []> : Op<OBS_Dialect, mnemonic, traits>;

def OBS_OwnType : 
    DialectType<OBS_Dialect, CPred<"::llvm::isa<OwnType>($_self)">, "OBS own type">;

def OBS_RefType : 
    DialectType<OBS_Dialect, CPred<"::llvm::isa<RefType>($_self)">, "OBS ref type">;

def OBS_Type : AnyTypeOf<[OBS_OwnType, OBS_RefType]>;

def OwnOp : OBS_op<"Own", []> {

    let summary = "Allocate a memory resource and bind to an owner." ;

    let description = [{
        This operation creates a memory resource and bind it to an owner.
    }];


    let arguments = (ins StrAttr:$type, F64Tensor:$dim);
    let results = (outs OBS_OwnType);

    let assemblyFormat = [{
       `(` attr-dict `,` $dim `:` type($dim) `)` `:` type(results)
    }];


    let hasVerifier = 1;

    let builders = [
        OpBuilder<(ins "StringRef":$type, "Value":$dim )>
    ];
}

def RefOp : OBS_op<"Ref", []> {

    let summary = "Create a reference to a resource." ;

    let description = [{
        This operation creates a reference to a resource.
    }];

    let arguments = (ins OBS_Type:$ownType);
    let results = (outs OBS_RefType);

    let assemblyFormat = [{
       `(` attr-dict $ownType `:` type($ownType) `)` `:` type(results)
    }];


    let hasVerifier = 1;

    let builders = [
        OpBuilder<(ins "Value":$ownType )>
    ];
}

def ReadOp : OBS_op<"Read", []> {

    let summary = "Read the resource though a variable" ;

    let description = [{
        This operation creates a read operation.
    }];

    let arguments = (ins OBS_Type:$ownType);

    let assemblyFormat = [{
       `(` attr-dict $ownType `:` type($ownType) `)`
    }];


    let hasVerifier = 1;
}

def WriteOp : OBS_op<"Write", []> {

    let summary = "Write the resource though a variable" ;

    let description = [{
        This operation creates a write operation.
    }];

    let arguments = (ins OBS_Type:$ownType);

    let assemblyFormat = [{
       `(` attr-dict $ownType `:` type($ownType) `)`
    }];


    let hasVerifier = 1;
}

def DeleteOp : OBS_op<"Delete", []> {

    let summary = "Delete a resource through its owner." ;

    let description = [{
        This operation creates a delete operation.
    }];

    let arguments = (ins OBS_Type:$ownType);

    let assemblyFormat = [{
       `(` attr-dict $ownType `:` type($ownType) `)`
    }];


    let hasVerifier = 1;
}


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
