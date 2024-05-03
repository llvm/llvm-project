# SPIR-V Dialect to LLVM Dialect conversion manual

This manual describes the conversion from [SPIR-V Dialect](Dialects/SPIR-V.md)
to [LLVM Dialect](Dialects/LLVM.md). It assumes familiarity with both, and
describes the design choices behind the modelling of SPIR-V concepts in LLVM
Dialect. The conversion is an ongoing work, and is expected to grow as more
features are implemented.

Conversion can be performed by invoking an appropriate conversion pass:

```shell
mlir-opt -convert-spirv-to-llvm <filename.mlir>
```

This pass performs type and operation conversions for SPIR-V operations as
described in this document.

[TOC]

## Type Conversion

This section describes how SPIR-V Dialect types are mapped to LLVM Dialect.

### Scalar types

SPIR-V Dialect | LLVM Dialect
:------------: | :-----------------:
`i<bitwidth>`  | `!llvm.i<bitwidth>`
`si<bitwidth>` | `!llvm.i<bitwidth>`
`ui<bitwidth>` | `!llvm.i<bitwidth>`
`f16`          | `f16`
`f32`          | `f32`
`f64`          | `f64`

### Vector types

SPIR-V Dialect                    | LLVM Dialect
:-------------------------------: | :-------------------------------:
`vector<<count> x <scalar-type>>` | `vector<<count> x <scalar-type>>`

### Pointer types

A SPIR-V pointer also takes a Storage Class. At the moment, conversion does
**not** take it into account.

SPIR-V Dialect                                | LLVM Dialect
:-------------------------------------------: | :-------------------------:
`!spirv.ptr< <element-type>, <storage-class> >` | `!llvm.ptr`

### Array types

SPIR-V distinguishes between array type and run-time array type, the length of
which is not known at compile time. In LLVM, it is possible to index beyond the
end of the array. Therefore, runtime array can be implemented as a zero length
array type.

Moreover, SPIR-V supports the notion of array stride. Currently only natural
strides (based on [`VulkanLayoutUtils`][VulkanLayoutUtils]) are supported. They
are also mapped to LLVM array.

SPIR-V Dialect                         | LLVM Dialect
:------------------------------------: | :-------------------------------------:
`!spirv.array<<count> x <element-type>>` | `!llvm.array<<count> x <element-type>>`
`!spirv.rtarray< <element-type> >`       | `!llvm.array<0 x <element-type>>`

### Struct types

Members of SPIR-V struct types may have decorations and offset information.
Currently, there is **no** support of member decorations conversion for structs.
For more information see section on [Decorations](#Decorations-conversion).

Usually we expect that each struct member has a natural size and alignment.
However, there are cases (*e.g.* in graphics) where one would place struct
members explicitly at particular offsets. This case is **not** supported at the
moment. Hence, we adhere to the following mapping:

*   Structs with no offset are modelled as LLVM packed structures.

*   Structs with natural offset (*i.e.* offset that equals to cumulative size of
    the previous struct elements or is a natural alignment) are mapped to
    naturally padded structs.

*   Structs with unnatural offset (*i.e.* offset that is not equal to cumulative
    size of the previous struct elements) are **not** supported. In this case,
    offsets can be emulated with padding fields (*e.g.* integers). However, such
    a design would require index recalculation in the conversion of ops that
    involve memory addressing.

Examples of SPIR-V struct conversion are: ```mlir !spirv.struct<i8, i32> =>
!llvm.struct<packed (i8, i32)> !spirv.struct<i8 [0], i32 [4]> => !llvm.struct<(i8,
i32)>

// error !spirv.struct<i8 [0], i32 [8]> ```

### Not implemented types

The rest of the types not mentioned explicitly above are not supported by the
conversion. This includes `ImageType` and `MatrixType`.

## Operation Conversion

This section describes how SPIR-V Dialect operations are converted to LLVM
Dialect. It lists already working conversion patterns, as well as those that are
an ongoing work.

There are also multiple ops for which there is no clear mapping in LLVM.
Conversion for those have to be discussed within the community on the
case-by-case basis.

### Arithmetic ops

SPIR-V arithmetic ops mostly have a direct equivalent in LLVM Dialect. Such
exceptions as `spirv.SMod` and `spirv.FMod` are rare.

SPIR-V Dialect op | LLVM Dialect op
:---------------: | :-------------:
`spirv.FAdd`        | `llvm.fadd`
`spirv.FDiv`        | `llvm.fdiv`
`spirv.FNegate`     | `llvm.fneg`
`spirv.FMul`        | `llvm.fmul`
`spirv.FRem`        | `llvm.frem`
`spirv.FSub`        | `llvm.fsub`
`spirv.IAdd`        | `llvm.add`
`spirv.IMul`        | `llvm.mul`
`spirv.ISub`        | `llvm.sub`
`spirv.SDiv`        | `llvm.sdiv`
`spirv.SRem`        | `llvm.srem`
`spirv.UDiv`        | `llvm.udiv`
`spirv.UMod`        | `llvm.urem`

### Bitwise ops

SPIR-V has a range of bit ops that are mapped to LLVM dialect ops, intrinsics or
may have a specific conversion pattern.

#### Direct conversion

As with arithmetic ops, most of bitwise ops have a semantically equivalent op in
LLVM:

SPIR-V Dialect op | LLVM Dialect op
:---------------: | :-------------:
`spirv.BitwiseAnd`  | `llvm.and`
`spirv.BitwiseOr`   | `llvm.or`
`spirv.BitwiseXor`  | `llvm.xor`

Also, some of bitwise ops can be modelled with LLVM intrinsics:

SPIR-V Dialect op | LLVM Dialect intrinsic
:---------------: | :--------------------:
`spirv.BitCount`    | `llvm.intr.ctpop`
`spirv.BitReverse`  | `llvm.intr.bitreverse`

#### `spirv.Not`

`spirv.Not` is modelled with a `xor` operation with a mask with all bits set.

```mlir
                            %mask = llvm.mlir.constant(-1 : i32) : i32
%0 = spirv.Not %op : i32  =>  %0  = llvm.xor %op, %mask : i32
```

#### Bitfield ops

SPIR-V dialect has three bitfield ops: `spirv.BitFieldInsert`,
`spirv.BitFieldSExtract` and `spirv.BitFieldUExtract`. This section will first
outline the general design of conversion patterns for this ops, and then
describe each of them.

All of these ops take `base`, `offset` and `count` (`insert` for
`spirv.BitFieldInsert`) as arguments. There are two important things to note:

*   `offset` and `count` are always scalar. This means that we can have the
    following case:

    ```mlir
    %0 = spirv.BitFieldSExtract %base, %offset, %count : vector<2xi32>, i8, i8
    ```

    To be able to proceed with conversion algorithms described below, all
    operands have to be of the same type and bitwidth. This requires
    broadcasting of `offset` and `count` to vectors, for example for the case
    above it gives:

    ```mlir
    // Broadcasting offset
    %offset0 = llvm.mlir.undef : vector<2xi8>
    %zero = llvm.mlir.constant(0 : i32) : i32
    %offset1 = llvm.insertelement %offset, %offset0[%zero : i32] : vector<2xi8>
    %one = llvm.mlir.constant(1 : i32) : i32
    %vec_offset = llvm.insertelement  %offset, %offset1[%one : i32] : vector<2xi8>

    // Broadcasting count
    // ...
    ```

*   `offset` and `count` may have different bitwidths from `base`. In this case,
    both of these operands have to be zero extended (since they are treated as
    unsigned by the specification) or truncated. For the above example it would
    be:

    ```mlir
    // Zero extending offset after broadcasting
    %res_offset = llvm.zext %vec_offset: vector<2xi8> to vector<2xi32>
    ```

    Also, note that if the bitwidth of `offset` or `count` is greater than the
    bitwidth of `base`, truncation is still permitted. This is because the ops
    have a defined behaviour with `offset` and `count` being less than the size
    of `base`. It creates a natural upper bound on what values `offset` and
    `count` can take, which is 64. This can be expressed in less than 8 bits.

Now, having these two cases in mind, we can proceed with conversion for the ops
and their operands.

##### `spirv.BitFieldInsert`

This operation is implemented as a series of LLVM Dialect operations. First step
would be to create a mask with bits set outside [`offset`, `offset` + `count` -
1]. Then, unchanged bits are extracted from `base` that are outside of
[`offset`, `offset` + `count` - 1]. The result is `or`ed with shifted `insert`.

```mlir
// Create mask
// %minus_one = llvm.mlir.constant(-1 : i32) : i32
// %t0        = llvm.shl %minus_one, %count : i32
// %t1        = llvm.xor %t0, %minus_one : i32
// %t2        = llvm.shl %t1, %offset : i32
// %mask      = llvm.xor %t2, %minus_one : i32

// Extract unchanged bits from the Base
// %new_base  = llvm.and %base, %mask : i32

// Insert new bits
// %sh_insert = llvm.shl %insert, %offset : i32
// %res       = llvm.or %new_base, %sh_insert : i32
%res = spirv.BitFieldInsert %base, %insert, %offset, %count : i32, i32, i32
```

##### `spirv.BitFieldSExtract`

To implement `spirv.BitFieldSExtract`, `base` is shifted left by [sizeof(`base`) -
(`count` + `offset`)], so that the bit at `offset` + `count` - 1 is the most
significant bit. After, the result is shifted right, filling the bits with the
sign bit.

```mlir
// Calculate the amount to shift left.
// %size    = llvm.mlir.constant(32 : i32) : i32
// %t0      = llvm.add %count, %offset : i32
// %t1      = llvm.sub %size, %t0 : i32

// Shift left and then right to extract the bits
// %sh_left = llvm.shl %base, %t1 : i32
// %t2      = llvm.add %offset, %t1 : i32
// %res     = llvm.ashr %sh_left, %t2 : i32
%res = spirv.BitFieldSExtract %base, %offset, %count : i32, i32, i32
```

##### `spirv.BitFieldUExtract`

For this op a similar pattern as for `spirv.BitFieldInsert` is used. First, a mask
with bits set at [0, `count` - 1] is created. Then `base` is shifted by `offset`
and the mask is applied.

```mlir
// Create a mask
// %minus_one = llvm.mlir.constant(-1 : i32) : i32
// %t0        = llvm.shl %minus_one, %count : i32
// mask       = llvm.xor  %t0, %minus_one : i32

// Shift Base and apply mask
// %sh_base   = llvm.lshr %base, %offset : i32
// %res       = llvm.and %sh_base, %mask : i32
%res = spirv.BitFieldUExtract %base, %offset, %count : i32, i32, i32
```

### Cast ops

#### Direct conversions

SPIR-V Dialect op | LLVM Dialect op
:---------------: | :-------------:
`spirv.ConvertFToS` | `llvm.fptosi`
`spirv.ConvertFToU` | `llvm.fptoui`
`spirv.ConvertSToF` | `llvm.sitofp`
`spirv.ConvertUToF` | `llvm.uitofp`

#### spirv.Bitcast

This operation has a direct counterpart in LLVM: `llvm.bitcast`. It is treated
separately since it also supports pointer to pointer bit pattern-preserving type
conversion, apart from regular scalar or vector of numerical type.

#### Special cases

Special cases include `spirv.FConvert`, `spirv.SConvert` and `spirv.UConvert`. These
operations are either a truncate or extend. Let's denote the operand component
width as A, and result component width as R. Then, the following mappings are
used:

##### `spirv.FConvert`

Case  | LLVM Dialect op
:---: | :-------------:
A < R | `llvm.fpext`
A > R | `llvm.fptrunc`

##### `spirv.SConvert`

Case  | LLVM Dialect op
:---: | :-------------:
A < R | `llvm.sext`
A > R | `llvm.trunc`

##### `spirv.UConvert`

Case  | LLVM Dialect op
:---: | :-------------:
A < R | `llvm.zext`
A > R | `llvm.trunc`

The case when A = R is not possible, based on SPIR-V Dialect specification:

> The component width cannot equal the component width in Result Type.

### Comparison ops

SPIR-V comparison ops are mapped to LLVM `icmp` and `fcmp` operations.

SPIR-V Dialect op            | LLVM Dialect op
:--------------------------: | :---------------:
`spirv.IEqual`                 | `llvm.icmp "eq"`
`spirv.INotEqual`              | `llvm.icmp "ne"`
`spirv.FOrdEqual`              | `llvm.fcmp "oeq"`
`spirv.FOrdGreaterThan`        | `llvm.fcmp "ogt"`
`spirv.FOrdGreaterThanEqual`   | `llvm.fcmp "oge"`
`spirv.FOrdLessThan`           | `llvm.fcmp "olt"`
`spirv.FOrdLessThanEqual`      | `llvm.fcmp "ole"`
`spirv.FOrdNotEqual`           | `llvm.fcmp "one"`
`spirv.FUnordEqual`            | `llvm.fcmp "ueq"`
`spirv.FUnordGreaterThan`      | `llvm.fcmp "ugt"`
`spirv.FUnordGreaterThanEqual` | `llvm.fcmp "uge"`
`spirv.FUnordLessThan`         | `llvm.fcmp "ult"`
`spirv.FUnordLessThanEqual`    | `llvm.fcmp "ule"`
`spirv.FUnordNotEqual`         | `llvm.fcmp "une"`
`spirv.SGreaterThan`           | `llvm.icmp "sgt"`
`spirv.SGreaterThanEqual`      | `llvm.icmp "sge"`
`spirv.SLessThan`              | `llvm.icmp "slt"`
`spirv.SLessThanEqual`         | `llvm.icmp "sle"`
`spirv.UGreaterThan`           | `llvm.icmp "ugt"`
`spirv.UGreaterThanEqual`      | `llvm.icmp "uge"`
`spirv.ULessThan`              | `llvm.icmp "ult"`
`spirv.ULessThanEqual`         | `llvm.icmp "ule"`

### Composite ops

Currently, conversion supports rewrite patterns for `spirv.CompositeExtract` and
`spirv.CompositeInsert`. We distinguish two cases for these operations: when the
composite object is a vector, and when the composite object is of a non-vector
type (*i.e.* struct, array or runtime array).

Composite type | SPIR-V Dialect op      | LLVM Dialect op
:------------: | :--------------------: | :-------------------:
vector         | `spirv.CompositeExtract` | `llvm.extractelement`
vector         | `spirv.CompositeInsert`  | `llvm.insertelement`
non-vector     | `spirv.CompositeExtract` | `llvm.extractvalue`
non-vector     | `spirv.CompositeInsert`  | `llvm.insertvalue`

### `spirv.EntryPoint` and `spirv.ExecutionMode`

First of all, it is important to note that there is no direct representation of
entry points in LLVM. At the moment, we use the following approach:

*   `spirv.EntryPoint` is simply removed.

*   In contrast, `spirv.ExecutionMode` may contain important information about the
    entry point. For example, `LocalSize` provides information about the
    work-group size that can be reused.

    In order to preserve this information, `spirv.ExecutionMode` is converted to a
    struct global variable that stores the execution mode id and any variables
    associated with it. In C, the struct has the structure shown below.

    ```c
    // No values are associated      // There are values that are associated
    // with this entry point.        // with this entry point.
    struct {                         struct {
      int32_t executionMode;             int32_t executionMode;
    };                                   int32_t values[];
                                     };
    ```

    ```mlir
    // spirv.ExecutionMode @empty "ContractionOff"
    llvm.mlir.global external constant @{{.*}}() : !llvm.struct<(i32)> {
      %0   = llvm.mlir.undef : !llvm.struct<(i32)>
      %1   = llvm.mlir.constant(31 : i32) : i32
      %ret = llvm.insertvalue %1, %0[0] : !llvm.struct<(i32)>
      llvm.return %ret : !llvm.struct<(i32)>
    }
    ```

### Logical ops

Logical ops follow a similar pattern as bitwise ops, with the difference that
they operate on `i1` or vector of `i1` values. The following mapping is used to
emulate SPIR-V ops behaviour:

SPIR-V Dialect op     | LLVM Dialect op
:-------------------: | :--------------:
`spirv.LogicalAnd`      | `llvm.and`
`spirv.LogicalOr`       | `llvm.or`
`spirv.LogicalEqual`    | `llvm.icmp "eq"`
`spirv.LogicalNotEqual` | `llvm.icmp "ne"`

`spirv.LogicalNot` has the same conversion pattern as bitwise `spirv.Not`. It is
modelled with `xor` operation with a mask with all bits set.

```mlir
                                  %mask = llvm.mlir.constant(-1 : i1) : i1
%0 = spirv.LogicalNot %op : i1  =>  %0    = llvm.xor %op, %mask : i1
```

### Memory ops

This section describes the conversion patterns for SPIR-V dialect operations
that concern memory.

#### `spirv.AccessChain`

`spirv.AccessChain` is mapped to `llvm.getelementptr` op. In order to create a
valid LLVM op, we also add a 0 index to the `spirv.AccessChain`'s indices list in
order to go through the pointer.

```mlir
// Access the 1st element of the array
%i   = spirv.Constant 1: i32
%var = spirv.Variable : !spirv.ptr<!spirv.struct<f32, !spirv.array<4xf32>>, Function>
%el  = spirv.AccessChain %var[%i, %i] : !spirv.ptr<!spirv.struct<f32, !spirv.array<4xf32>>, Function>, i32, i32

// Corresponding LLVM dialect code
%i   = ...
%var = ...
%0   = llvm.mlir.constant(0 : i32) : i32
%el  = llvm.getelementptr %var[%0, %i, %i] : (!llvm.ptr, i32, i32, i32), !llvm.struct<packed (f32, array<4 x f32>)>
```

#### `spirv.Load` and `spirv.Store`

These ops are converted to their LLVM counterparts: `llvm.load` and
`llvm.store`. If the op has a memory access attribute, then there are the
following cases, based on the value of the attribute:

*   **Aligned**: alignment is passed on to LLVM op builder, for example: `mlir
    // llvm.store %ptr, %val {alignment = 4 : i64} : !llvm.ptr spirv.Store
    "Function" %ptr, %val ["Aligned", 4] : f32`
*   **None**: same case as if there is no memory access attribute.

*   **Nontemporal**: set `nontemporal` flag, for example: `mlir // %res =
    llvm.load %ptr {nontemporal} : !llvm.ptr %res = spirv.Load "Function"
    %ptr ["Nontemporal"] : f32`

*   **Volatile**: mark the op as `volatile`, for example: `mlir // %res =
    llvm.load volatile %ptr : !llvm.ptr f32> %res = spirv.Load "Function" %ptr
    ["Volatile"] : f32` Otherwise the conversion fails as other cases
    (`MakePointerAvailable`, `MakePointerVisible`, `NonPrivatePointer`) are not
    supported yet.

#### `spirv.GlobalVariable` and `spirv.mlir.addressof`

`spirv.GlobalVariable` is modelled with `llvm.mlir.global` op. However, there is a
difference that has to be pointed out.

In SPIR-V dialect, the global variable returns a pointer, whereas in LLVM
dialect the global holds an actual value. This difference is handled by
`spirv.mlir.addressof` and `llvm.mlir.addressof` ops that both return a pointer
and are used to reference the global.

```mlir
// Original SPIR-V module
spirv.module Logical GLSL450 {
  spirv.GlobalVariable @struct : !spirv.ptr<!spirv.struct<f32, !spirv.array<10xf32>>, Private>
  spirv.func @func() -> () "None" {
    %0 = spirv.mlir.addressof @struct : !spirv.ptr<!spirv.struct<f32, !spirv.array<10xf32>>, Private>
    spirv.Return
  }
}

// Converted result
module {
  llvm.mlir.global private @struct() : !llvm.struct<packed (f32, [10 x f32])>
  llvm.func @func() {
    %0 = llvm.mlir.addressof @struct : !llvm.ptr
    llvm.return
  }
}
```

The SPIR-V to LLVM conversion does not involve modelling of workgroups. Hence,
we say that only current invocation is in conversion's scope. This means that
global variables with pointers of `Input`, `Output`, and `Private` storage
classes are supported. Also, `StorageBuffer` storage class is allowed for
executing [`mlir-spirv-cpu-runner`](#mlir-spirv-cpu-runner).

Moreover, `bind` that specifies the descriptor set and the binding number and
`built_in` that specifies SPIR-V `BuiltIn` decoration have no conversion into
LLVM dialect.

Currently `llvm.mlir.global`s are created with `private` linkage for `Private`
storage class and `External` for other storage classes, based on SPIR-V spec:

> By default, functions and global variables are private to a module and cannot
> be accessed by other modules. However, a module may be written to export or
> import functions and global (module scope) variables.

If the global variable's pointer has `Input` storage class, then a `constant`
flag is added to LLVM op:

```mlir
spirv.GlobalVariable @var : !spirv.ptr<f32, Input>    =>    llvm.mlir.global external constant @var() : f32
```

#### `spirv.Variable`

Per SPIR-V dialect spec, `spirv.Variable` allocates an object in memory, resulting
in a pointer to it, which can be used with `spirv.Load` and `spirv.Store`. It is
also a function-level variable.

`spirv.Variable` is modelled as `llvm.alloca` op. If initialized, an additional
store instruction is used. Note that there is no initialization for arrays and
structs since constants of these types are not supported in LLVM dialect (TODO).
Also, at the moment initialization is only possible via `spirv.Constant`.

```mlir
// Conversion of VariableOp without initialization
                                                               %size = llvm.mlir.constant(1 : i32) : i32
%res = spirv.Variable : !spirv.ptr<vector<3xf32>, Function>   =>   %res  = llvm.alloca  %size x vector<3xf32> : (i32) -> !llvm.ptr

// Conversion of VariableOp with initialization
                                                               %c    = llvm.mlir.constant(0 : i64) : i64
%c   = spirv.Constant 0 : i64                                    %size = llvm.mlir.constant(1 : i32) : i32
%res = spirv.Variable init(%c) : !spirv.ptr<i64, Function>    =>   %res  = llvm.alloca %[[SIZE]] x i64 : (i32) -> !llvm.ptr
                                                               llvm.store %c, %res : i64, !llvm.ptr
```

Note that simple conversion to `alloca` may not be sufficient if the code has
some scoping. For example, if converting ops executed in a loop into `alloca`s,
a stack overflow may occur. For this case, `stacksave`/`stackrestore` pair can
be used (TODO).

### Miscellaneous ops with direct conversions

There are multiple SPIR-V ops that do not fit in a particular group but can be
converted directly to LLVM dialect. Their conversion is addressed in this
section.

SPIR-V Dialect op | LLVM Dialect op
:---------------: | :---------------:
`spirv.Select`      | `llvm.select`
`spirv.Undef`       | `llvm.mlir.undef`

### Shift ops

Shift operates on two operands: `shift` and `base`.

In SPIR-V dialect, `shift` and `base` may have different bit width. On the
contrary, in LLVM Dialect both `base` and `shift` have to be of the same
bitwidth. This leads to the following conversions:

*   if `base` has the same bitwidth as `shift`, the conversion is
    straightforward.

*   if `base` has a greater bit width than `shift`, shift is sign or zero
    extended first. Then the extended value is passed to the shift.

*   otherwise, the conversion is considered to be illegal.

```mlir
// Shift without extension
%res0 = spirv.ShiftRightArithmetic %0, %2 : i32, i32  =>  %res0 = llvm.ashr %0, %2 : i32

// Shift with extension
                                                        %ext  = llvm.sext %1 : i16 to i32
%res1 = spirv.ShiftRightArithmetic %0, %1 : i32, i16  =>  %res1 = llvm.ashr %0, %ext: i32
```

### `spirv.Constant`

At the moment `spirv.Constant` conversion supports scalar and vector constants
**only**.

#### Mapping

`spirv.Constant` is mapped to `llvm.mlir.constant`. This is a straightforward
conversion pattern with a special case when the argument is signed or unsigned.

#### Special case

SPIR-V constant can be a signed or unsigned integer. Since LLVM Dialect does not
have signedness semantics, this case should be handled separately.

The conversion casts constant value attribute to a signless integer or a vector
of signless integers. This is correct because in SPIR-V, like in LLVM, how to
interpret an integer number is also dictated by the opcode. However, in reality
hardware implementation might show unexpected behavior. Therefore, it is better
to handle it case-by-case, given that the purpose of the conversion is not to
cover all possible corner cases.

```mlir
// %0 = llvm.mlir.constant(0 : i8) : i8
%0 = spirv.Constant  0 : i8

// %1 = llvm.mlir.constant(dense<[2, 3, 4]> : vector<3xi32>) : vector<3xi32>
%1 = spirv.Constant dense<[2, 3, 4]> : vector<3xui32>
```

### Not implemented ops

There is no support of the following ops:

*   All atomic ops
*   All group ops
*   All matrix ops
*   All CL ops

As well as:

*   spirv.CompositeConstruct
*   spirv.ControlBarrier
*   spirv.CopyMemory
*   spirv.FMod
*   spirv.GL.Acos
*   spirv.GL.Asin
*   spirv.GL.Atan
*   spirv.GL.Cosh
*   spirv.GL.FSign
*   spirv.GL.SAbs
*   spirv.GL.Sinh
*   spirv.GL.SSign
*   spirv.MemoryBarrier
*   spirv.mlir.referenceof
*   spirv.SMod
*   spirv.SpecConstant
*   spirv.Unreachable
*   spirv.VectorExtractDynamic

## Control flow conversion

### Branch ops

`spirv.Branch` and `spirv.BranchConditional` are mapped to `llvm.br` and
`llvm.cond_br`. Branch weights for `spirv.BranchConditional` are mapped to
corresponding `branch_weights` attribute of `llvm.cond_br`. When translated to
proper LLVM, `branch_weights` are converted into LLVM metadata associated with
the conditional branch.

### `spirv.FunctionCall`

`spirv.FunctionCall` maps to `llvm.call`. For example:

```mlir
%0 = spirv.FunctionCall @foo() : () -> i32    =>    %0 = llvm.call @foo() : () -> f32
spirv.FunctionCall @bar(%0) : (i32) -> ()     =>    llvm.call @bar(%0) : (f32) -> ()
```

### `spirv.mlir.selection` and `spirv.mlir.loop`

Control flow within `spirv.mlir.selection` and `spirv.mlir.loop` is lowered directly
to LLVM via branch ops. The conversion can only be applied to selection or loop
with all blocks being reachable. Moreover, selection and loop control attributes
(such as `Flatten` or `Unroll`) are not supported at the moment.

```mlir
// Conversion of selection
%cond = spirv.Constant true                               %cond = llvm.mlir.constant(true) : i1
spirv.mlir.selection {
  spirv.BranchConditional %cond, ^true, ^false            llvm.cond_br %cond, ^true, ^false

^true:                                                                                              ^true:
  // True block code                                    // True block code
  spirv.Branch ^merge                             =>      llvm.br ^merge

^false:                                               ^false:
  // False block code                                   // False block code
  spirv.Branch ^merge                                     llvm.br ^merge

^merge:                                               ^merge:
  spirv.mlir.merge                                            llvm.br ^continue
}
// Remaining code                                                                           ^continue:
                                                        // Remaining code
```

```mlir
// Conversion of loop
%cond = spirv.Constant true                               %cond = llvm.mlir.constant(true) : i1
spirv.mlir.loop {
  spirv.Branch ^header                                    llvm.br ^header

^header:                                              ^header:
  // Header code                                        // Header code
  spirv.BranchConditional %cond, ^body, ^merge    =>      llvm.cond_br %cond, ^body, ^merge

^body:                                                ^body:
  // Body code                                          // Body code
  spirv.Branch ^continue                                  llvm.br ^continue

^continue:                                            ^continue:
  // Continue code                                      // Continue code
  spirv.Branch ^header                                    llvm.br ^header

^merge:                                               ^merge:
  spirv.mlir.merge                                            llvm.br ^remaining
}
// Remaining code                                     ^remaining:
                                                        // Remaining code
```

## Decorations conversion

**Note: these conversions have not been implemented yet**

## GLSL extended instruction set

This section describes how SPIR-V ops from GLSL extended instructions set are
mapped to LLVM Dialect.

### Direct conversions

SPIR-V Dialect op | LLVM Dialect op
:---------------: | :----------------:
`spirv.GL.Ceil`     | `llvm.intr.ceil`
`spirv.GL.Cos`      | `llvm.intr.cos`
`spirv.GL.Exp`      | `llvm.intr.exp`
`spirv.GL.FAbs`     | `llvm.intr.fabs`
`spirv.GL.Floor`    | `llvm.intr.floor`
`spirv.GL.FMax`     | `llvm.intr.maxnum`
`spirv.GL.FMin`     | `llvm.intr.minnum`
`spirv.GL.Log`      | `llvm.intr.log`
`spirv.GL.Sin`      | `llvm.intr.sin`
`spirv.GL.Sqrt`     | `llvm.intr.sqrt`
`spirv.GL.SMax`     | `llvm.intr.smax`
`spirv.GL.SMin`     | `llvm.intr.smin`

### Special cases

`spirv.InverseSqrt` is mapped to:

```mlir
                                           %one  = llvm.mlir.constant(1.0 : f32) : f32
%res = spirv.InverseSqrt %arg : f32    =>    %sqrt = "llvm.intr.sqrt"(%arg) : (f32) -> f32
                                           %res  = fdiv %one, %sqrt : f32
```

`spirv.Tan` is mapped to:

```mlir
                                   %sin = "llvm.intr.sin"(%arg) : (f32) -> f32
%res = spirv.Tan %arg : f32    =>    %cos = "llvm.intr.cos"(%arg) : (f32) -> f32
                                   %res = fdiv %sin, %cos : f32
```

`spirv.Tanh` is modelled using the equality `tanh(x) = {exp(2x) - 1}/{exp(2x) +
1}`:

```mlir
                                     %two   = llvm.mlir.constant(2.0: f32) : f32
                                     %2xArg = llvm.fmul %two, %arg : f32
                                     %exp   = "llvm.intr.exp"(%2xArg) : (f32) -> f32
%res = spirv.Tanh %arg : f32     =>    %one   = llvm.mlir.constant(1.0 : f32) : f32
                                     %num   = llvm.fsub %exp, %one : f32
                                     %den   = llvm.fadd %exp, %one : f32
                                     %res   = llvm.fdiv %num, %den : f32
```

## Function conversion and related ops

This section describes the conversion of function-related operations from SPIR-V
to LLVM dialect.

### `spirv.func`

This op declares or defines a SPIR-V function and it is converted to
`llvm.func`. This conversion handles signature conversion, and function control
attributes remapping to LLVM dialect function
[`passthrough` attribute](Dialects/LLVM.md/#attribute-pass-through).

The following mapping is used to map
[SPIR-V function control][SPIRVFunctionAttributes] to
[LLVM function attributes][LLVMFunctionAttributes]:

SPIR-V Function Control Attributes | LLVM Function Attributes
:--------------------------------: | :---------------------------:
None                               | No function attributes passed
Inline                             | `alwaysinline`
DontInline                         | `noinline`
Pure                               | `readonly`
Const                              | `readnone`

### `spirv.Return` and `spirv.ReturnValue`

In LLVM IR, functions may return either 1 or 0 value. Hence, we map both ops to
`llvm.return` with or without a return value.

## Module ops

Module in SPIR-V has one region that contains one block. It is defined via
`spirv.module` op that also takes a range of attributes:

*   Addressing model
*   Memory model
*   Version-Capability-Extension attribute

`spirv.module` is converted into `ModuleOp`. This plays a role of enclosing scope
to LLVM ops. At the moment, SPIR-V module attributes are ignored.

## `mlir-spirv-cpu-runner`

`mlir-spirv-cpu-runner` allows to execute `gpu` dialect kernel on the CPU via
SPIR-V to LLVM dialect conversion. Currently, only single-threaded kernel is
supported.

To build the runner, add the following option to `cmake`: `bash
-DMLIR_ENABLE_SPIRV_CPU_RUNNER=1`

### Pipeline

The `gpu` module with the kernel and the host code undergo the following
transformations:

*   Convert the `gpu` module into SPIR-V dialect, lower ABI attributes and
    update version, capability and extension.

*   Emulate the kernel call by converting the launching operation into a normal
    function call. The data from the host side to the device is passed via
    copying to global variables. These are created in both the host and the
    kernel code and later linked when nested modules are folded.

*   Convert SPIR-V dialect kernel to LLVM dialect via the new conversion path.

After these passes, the IR transforms into a nested LLVM module - a main module
representing the host code and a kernel module. These modules are linked and
executed using `ExecutionEngine`.

### Walk-through

This section gives a detailed overview of the IR changes while running
`mlir-spirv-cpu-runner`. First, consider that we have the following IR. (For
simplicity some type annotations and function implementations have been
omitted).

```mlir
gpu.module @foo {
  gpu.func @bar(%arg: memref<8xi32>) {
    // Kernel code.
    gpu.return
  }
}

func.func @main() {
  // Fill the buffer with some data
  %buffer = memref.alloc : memref<8xi32>
  %data = ...
  call fillBuffer(%buffer, %data)

  "gpu.launch_func"(/*grid dimensions*/, %buffer) {
    kernel = @foo::bar
  }
}
```

Lowering `gpu` dialect to SPIR-V dialect results in

```mlir
spirv.module @__spv__foo /*VCE triple and other metadata here*/ {
  spirv.GlobalVariable @__spv__foo_arg bind(0,0) : ...
  spirv.func @bar() {
    // Kernel code.
  }
  spirv.EntryPoint @bar, ...
}

func.func @main() {
  // Fill the buffer with some data.
  %buffer = memref.alloc : memref<8xi32>
  %data = ...
  call fillBuffer(%buffer, %data)

  "gpu.launch_func"(/*grid dimensions*/, %buffer) {
    kernel = @foo::bar
  }
}
```

Then, the lowering from standard dialect to LLVM dialect is applied to the host
code.

```mlir
spirv.module @__spv__foo /*VCE triple and other metadata here*/ {
  spirv.GlobalVariable @__spv__foo_arg bind(0,0) : ...
  spirv.func @bar() {
    // Kernel code.
  }
  spirv.EntryPoint @bar, ...
}

// Kernel function declaration.
llvm.func @__spv__foo_bar() : ...

llvm.func @main() {
  // Fill the buffer with some data.
  llvm.call fillBuffer(%buffer, %data)

  // Copy data to the global variable, call kernel, and copy the data back.
  %addr = llvm.mlir.addressof @__spv__foo_arg_descriptor_set0_binding0 : ...
  "llvm.intr.memcpy"(%addr, %buffer) : ...
  llvm.call @__spv__foo_bar()
  "llvm.intr.memcpy"(%buffer, %addr) : ...

  llvm.return
}
```

Finally, SPIR-V module is converted to LLVM and the symbol names are resolved
for the linkage.

```mlir
module @__spv__foo {
  llvm.mlir.global @__spv__foo_arg_descriptor_set0_binding0 : ...
  llvm.func @__spv__foo_bar() {
    // Kernel code.
  }
}

// Kernel function declaration.
llvm.func @__spv__foo_bar() : ...

llvm.func @main() {
  // Fill the buffer with some data.
  llvm.call fillBuffer(%buffer, %data)

  // Copy data to the global variable, call kernel, and copy the data back.
  %addr = llvm.mlir.addressof @__spv__foo_arg_descriptor_set0_binding0 : ...
  "llvm.intr.memcpy"(%addr, %buffer) : ...
  llvm.call @__spv__foo_bar()
  "llvm.intr.memcpy"(%buffer, %addr) : ...

  llvm.return
}
```

[LLVMFunctionAttributes]: https://llvm.org/docs/LangRef.html#function-attributes
[SPIRVFunctionAttributes]: https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#_a_id_function_control_a_function_control
[VulkanLayoutUtils]: https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/SPIRV/LayoutUtils.h
