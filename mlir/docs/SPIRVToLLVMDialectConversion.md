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

SPIR-V Dialect                       | LLVM Dialect
:----------------------------------: | :----------------------------------:
`i<bitwidth>`                        | `!llvm.i<bitwidth>`
`si<bitwidth>`                       | `!llvm.i<bitwidth>`
`ui<bitwidth>`                       | `!llvm.i<bitwidth>`
`f16`                                | `!llvm.half`
`f32`                                | `!llvm.float`
`f64`                                | `!llvm.double`

### Vector types

SPIR-V Dialect                       | LLVM Dialect
:----------------------------------: | :----------------------------------:
`vector<<count> x <scalar-type>>`    | `!llvm.vec<<count> x <scalar-type>>`

### Pointer types

A SPIR-V pointer also takes a Storage Class. At the moment, conversion does
**not** take it into account.

SPIR-V Dialect                                | LLVM Dialect
:-------------------------------------------: | :-------------------------:
`!spv.ptr< <element-type>, <storage-class> >` | `!llvm.ptr<<element-type>>`

### Array types

SPIR-V distinguishes between array type and run-time array type, the length of
which is not known at compile time. In LLVM, it is possible to index beyond the
end of the array. Therfore, runtime array can be implemented as a zero length
array type.

Moreover, SPIR-V supports the notion of array stride. Note that this is **not**
supported by type conversion at the moment.

SPIR-V Dialect                        | LLVM Dialect
:-----------------------------------: | :-----------------------------------:
`!spv.array<<count> x <element-type>>`| `!llvm.array<<count> x <element-type>>`
`!spv.rtarray< <element-type> >`      | `!llvm.array<0 x <element-type>>`

### Struct types

Members of SPIR-V struct types may have decorations and offset information.
Currently, there is **no** support of member decorations conversion for structs.
For more information see section on [Decorations](#Decorations-conversion). 

Usually we expect that each struct member has a natural size and alignment.
However, there are cases (*e.g.* in graphics) where one would place struct 
members explicitly at particular offsets. This case is **not** supported
at the moment. Hence, we adhere to the following mapping:

*   Structs with no offset are modelled as LLVM packed structures.

*   Structs with natural offset (*i.e.* offset that equals to cumulative size of
    the previous struct elements or is a natural alignment) are mapped to
    naturally padded structs. Nested structs with offset and structs with offset
    containing arrays are **not** supported.

*   Structs with unnatural offset (*i.e.* offset that is not equal to cumulative
    size of the previous struct elements) are **not** supported. In this case,
    offsets can be emulated with padding fields (*e.g.* integers). However, such
    a design would require index recalculation in the conversion of ops that
    involve memmory addressing.

Examples of SPIR-V struct conversion are:
```mlir
!spv.struct<i8, i32>          =>  !llvm.struct<packed (i8, i32)>
!spv.struct<i8 [0], i32 [4]>  =>  !llvm.struct<(i8, i32)>

// error
!spv.struct<i8 [0], i32 [8]>
```

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
exceptions as `spv.SMod` and `spv.FMod` are rare.

SPIR-V Dialect op                     | LLVM Dialect op
:-----------------------------------: | :-----------------------------------:
`spv.FAdd`                            | `llvm.fadd`
`spv.FDiv`                            | `llvm.fdiv`
`spv.FNegate`                         | `llvm.fneg`
`spv.FMul`                            | `llvm.fmul`
`spv.FRem`                            | `llvm.frem`
`spv.FSub`                            | `llvm.fsub`
`spv.IAdd`                            | `llvm.add`
`spv.IMul`                            | `llvm.mul`
`spv.ISub`                            | `llvm.sub`
`spv.SDiv`                            | `llvm.sdiv`
`spv.SRem`                            | `llvm.srem`
`spv.UDiv`                            | `llvm.udiv`
`spv.UMod`                            | `llvm.urem`

### Bitwise ops

SPIR-V has a range of bit ops that are mapped to LLVM dialect ops, intrinsics or
may have a specific conversion pattern.

#### Direct conversion

As with arithmetic ops, most of bitwise ops have a semantically equivalent op in
LLVM:

SPIR-V Dialect op                     | LLVM Dialect op
:-----------------------------------: | :-----------------------------------:
`spv.BitwiseAnd`                      | `llvm.and`
`spv.BitwiseOr`                       | `llvm.or`
`spv.BitwiseXor`                      | `llvm.xor`

Also, some of bitwise ops can be modelled with LLVM intrinsics:

SPIR-V Dialect op                     | LLVM Dialect intrinsic
:-----------------------------------: | :-----------------------------------:
`spv.BitCount`                        | `llvm.intr.ctpop`
`spv.BitReverse`                      | `llvm.intr.bitreverse`

#### `spv.Not`

`spv.Not` is modelled with a `xor` operation with a mask with all bits set.

```mlir
                            %mask = llvm.mlir.constant(-1 : i32) : !llvm.i32
%0 = spv.Not %op : i32  =>  %0  = llvm.xor %op, %mask : !llvm.i32
```

#### Bitfield ops

SPIR-V dialect has three bitfield ops: `spv.BitFieldInsert`,
`spv.BitFieldSExtract` and `spv.BitFieldUExtract`. This section will first
outline the general design of conversion patterns for this ops, and then
describe each of them.

All of these ops take `base`, `offset` and `count` (`insert` for 
`spv.BitFieldInsert`) as arguments. There are two important things
to note:

*   `offset` and `count` are always scalar. This means that we can have the
    following case:

    ```mlir
    %0 = spv.BitFieldSExtract %base, %offset, %count : vector<2xi32>, i8, i8
    ```

    To be able to proceed with conversion algorithms described below, all
    operands have to be of the same type and bitwidth. This requires
    broadcasting of `offset` and `count` to vectors, for example for the case
    above it gives:

    ```mlir
    // Broadcasting offset
    %offset0 = llvm.mlir.undef : !llvm.vec<2 x i8>
    %zero = llvm.mlir.constant(0 : i32) : !llvm.i32
    %offset1 = llvm.insertelement %offset, %offset0[%zero : !llvm.i32] : !llvm.vec<2 x i8>
    %one = llvm.mlir.constant(1 : i32) : !llvm.i32
    %vec_offset = llvm.insertelement  %offset, %offset1[%one : !llvm.i32] : !llvm.vec<2 x i8>

    // Broadcasting count
    // ...
    ```

*   `offset` and `count` may have different bitwidths from `base`. In this case,
    both of these operands have to be zero extended (since they are treated as
    unsigned by the specification) or truncated. For the above example it would
    be:

    ```mlir
    // Zero extending offest after broadcasting
    %res_offset = llvm.zext %vec_offset: !llvm.vec<2 x i8> to !llvm.vec<2 x i32>
    ```

    Also, note that if the bitwidth of `offset` or `count` is greater than the
    bitwidth of `base`, truncation is still permitted. This is because the ops have a
    defined behaviour with `offset` and `count` being less than the size of
    `base`. It creates a natural upper bound on what values `offset` and `count`
    can take, which is 64. This can be expressed in less than 8 bits.

Now, having these two cases in mind, we can proceed with conversion for the ops
and their operands.

##### `spv.BitFieldInsert`

This operation is implemented as a series of LLVM Dialect operations. First step
would be to create a mask with bits set outside
[`offset`, `offset` + `count` - 1]. Then, unchanged bits are extracted from
`base` that are outside of [`offset`, `offset` + `count` - 1]. The result is
`or`ed with shifted `insert`.

```mlir
// Create mask
// %minus_one = llvm.mlir.constant(-1 : i32) : !llvm.i32
// %t0        = llvm.shl %minus_one, %count : !llvm.i32
// %t1        = llvm.xor %t0, %minus_one : !llvm.i32
// %t2        = llvm.shl %t1, %offset : !llvm.i32
// %mask      = llvm.xor %t2, %minus_one : !llvm.i32

// Extract unchanged bits from the Base
// %new_base  = llvm.and %base, %mask : !llvm.i32

// Insert new bits
// %sh_insert = llvm.shl %insert, %offset : !llvm.i32
// %res       = llvm.or %new_base, %sh_insert : !llvm.i32
%res = spv.BitFieldInsert %base, %insert, %offset, %count : i32, i32, i32
```

##### `spv.BitFieldSExtract`

To implement `spv.BitFieldSExtract`, `base` is shifted left by [sizeof(`base`) -
(`count` + `offset`)], so that the bit at `offset` + `count` - 1 is the most
significant bit. After, the result is shifted right, filling the bits with the
sign bit.

```mlir
// Calculate the amount to shift left.
// %size    = llvm.mlir.constant(32 : i32) : !llvm.i32
// %t0      = llvm.add %count, %offset : !llvm.i32
// %t1      = llvm.sub %size, %t0 : !llvm.i32

// Shift left and then right to extract the bits
// %sh_left = llvm.shl %base, %t1 : !llvm.i32
// %t2      = llvm.add %offset, %t1 : !llvm.i32
// %res     = llvm.ashr %sh_left, %t2 : !llvm.i32
%res = spv.BitFieldSExtract %base, %offset, %count : i32, i32, i32
```

##### `spv.BitFieldUExtract`

For this op a similar pattern as for `spv.BitFieldInsert` is used. First, a mask
with bits set at [0, `count` - 1] is created. Then `base` is shifted by `offset`
and the mask is applied.

```mlir
// Create a mask
// %minus_one = llvm.mlir.constant(-1 : i32) : !llvm.i32
// %t0        = llvm.shl %minus_one, %count : !llvm.i32
// mask       = llvm.xor  %t0, %minus_one : !llvm.i32

// Shift Base and apply mask
// %sh_base   = llvm.lshr %base, %offset : !llvm.i32
// %res       = llvm.and %sh_base, %mask : !llvm.i32
%res = spv.BitFieldUExtract %base, %offset, %count : i32, i32, i32
```

### Cast ops

#### Direct conversions

SPIR-V Dialect op                     | LLVM Dialect op
:-----------------------------------: | :-----------------------------------:
`spv.ConvertFToS`                     | `llvm.fptosi`
`spv.ConvertFToU`                     | `llvm.fptoui`
`spv.ConvertSToF`                     | `llvm.sitofp`
`spv.ConvertUToF`                     | `llvm.uitofp`

#### spv.Bitcast
This operation has a direct counterpart in LLVM: `llvm.bitcast`. It is treated
separately since it also supports pointer to pointer bit pattern-preserving type
conversion, apart from regular scalar or vector of numerical type.

#### Special cases
Special cases include `spv.FConvert`, `spv.SConvert` and `spv.UConvert`. These
operations are either a truncate or extend. Let's denote the operand component
width as A, and result component width as R. Then, the following mappings are
used:

##### `spv.FConvert` 
Case            | LLVM Dialect op
:-------------: | :-----------------------------------:
A < R           | `llvm.fpext`
A > R           | `llvm.fptrunc`

##### `spv.SConvert` 
Case            | LLVM Dialect op
:-------------: | :-----------------------------------:
A < R           | `llvm.sext`
A > R           | `llvm.trunc`

##### `spv.UConvert` 
Case            | LLVM Dialect op
:-------------: | :-----------------------------------:
A < R           | `llvm.zext`
A > R           | `llvm.trunc`

The case when A = R is not possible, based on SPIR-V Dialect specification:
> The component width cannot equal the component width in Result Type.

### Comparison ops

SPIR-V comparison ops are mapped to LLVM `icmp` and `fcmp` operations.

SPIR-V Dialect op                     | LLVM Dialect op
:-----------------------------------: | :-----------------------------------:
`spv.IEqual`                          | `llvm.icmp "eq"`
`spv.INotEqual`                       | `llvm.icmp "ne"`
`spv.FOrdEqual`                       | `llvm.fcmp "oeq"`
`spv.FOrdGreaterThan`                 | `llvm.fcmp "ogt"`
`spv.FOrdGreaterThanEqual`            | `llvm.fcmp "oge"`
`spv.FOrdLessThan`                    | `llvm.fcmp "olt"`
`spv.FOrdLessThanEqual`               | `llvm.fcmp "ole"`
`spv.FOrdNotEqual`                    | `llvm.fcmp "one"`
`spv.FUnordEqual`                     | `llvm.fcmp "ueq"`
`spv.FUnordGreaterThan`               | `llvm.fcmp "ugt"`
`spv.FUnordGreaterThanEqual`          | `llvm.fcmp "uge"`
`spv.FUnordLessThan`                  | `llvm.fcmp "ult"`
`spv.FUnordLessThanEqual`             | `llvm.fcmp "ule"`
`spv.FUnordNotEqual`                  | `llvm.fcmp "une"`
`spv.SGreaterThan`                    | `llvm.icmp "sgt"`
`spv.SGreaterThanEqual`               | `llvm.icmp "sge"`
`spv.SLessThan`                       | `llvm.icmp "slt"`
`spv.SLessThanEqual`                  | `llvm.icmp "sle"`
`spv.UGreaterThan`                    | `llvm.icmp "ugt"`
`spv.UGreaterThanEqual`               | `llvm.icmp "uge"`
`spv.ULessThan`                       | `llvm.icmp "ult"`
`spv.ULessThanEqual`                  | `llvm.icmp "ule"`

### Logical ops

Logical ops follow a similar pattern as bitwise ops, with the difference that
they operate on `i1` or vector of `i1` values. The following mapping is used to
emulate SPIR-V ops behaviour:

SPIR-V Dialect op                     | LLVM Dialect op
:-----------------------------------: | :-----------------------------------:
`spv.LogicalAnd`                      | `llvm.and`
`spv.LogicalOr`                       | `llvm.or`
`spv.LogicalEqual`                    | `llvm.icmp "eq"`
`spv.LogicalNotEqual`                 | `llvm.icmp "ne"`

`spv.LogicalNot` has the same conversion pattern as bitwise `spv.Not`. It is
modelled with `xor` operation with a mask with all bits set.

```mlir
                                  %mask = llvm.mlir.constant(-1 : i1) : !llvm.i1
%0 = spv.LogicalNot %op : i1  =>  %0    = llvm.xor %op, %mask : !llvm.i1
```

### Memory ops

This section describes the conversion patterns for SPIR-V dialect operations
that concern memory.

#### `spv.Load` and `spv.Store`

These ops are converted to their LLVM counterparts: `llvm.load` and
`llvm.store`. If the op has a memory access attribute, then there are the
following cases, based on the value of the attribute:

* **Aligned**: alignment is passed on to LLVM op builder, for example:
  ```mlir
  // llvm.store %ptr, %val {alignment = 4 : i64} : !llvm.ptr<float>
  spv.Store "Function" %ptr, %val ["Aligned", 4] : f32
  ```
* **None**: same case as if there is no memory access attribute.

* **Nontemporal**: set `nontemporal` flag, for example:
  ```mlir
  // %res = llvm.load %ptr {nontemporal} : !llvm.ptr<float>
  %res = spv.Load "Function" %ptr ["Nontemporal"] : f32
  ```
* **Volatile**: mark the op as `volatile`, for example:
  ```mlir
  // %res = llvm.load volatile %ptr : !llvm.ptr<float>
  %res = spv.Load "Function" %ptr ["Volatile"] : f32
  ```
Otherwise the conversion fails as other cases (`MakePointerAvailable`,
`MakePointerVisible`, `NonPrivatePointer`) are not supported yet.

#### `spv.globalVariable` and `spv._address_of`

`spv.globalVariable` is modelled with `llvm.mlir.global` op. However, there
is a difference that has to be pointed out.

In SPIR-V dialect, the global variable returns a pointer, whereas in LLVM
dialect the global holds an actual value. This difference is handled by
`spv._address_of` and `llvm.mlir.addressof` ops that both return a pointer and
are used to reference the global.

```mlir
// Original SPIR-V module
spv.module Logical GLSL450 {
  spv.globalVariable @struct : !spv.ptr<!spv.struct<f32, !spv.array<10xf32>>, Private>
  spv.func @func() -> () "None" {
    %0 = spv._address_of @struct : !spv.ptr<!spv.struct<f32, !spv.array<10xf32>>, Private>
    spv.Return
  }
}

// Converted result
module {
  llvm.mlir.global private @struct() : !llvm.struct<packed (float, [10 x float])>
  llvm.func @func() {
    %0 = llvm.mlir.addressof @struct : !llvm.ptr<struct<packed (float, [10 x float])>>
    llvm.return
  }
}
```

At the moment, only current invocation is in conversion's scope. This means that
global variables with pointers of `Input`, `Output` and `Private` storage
classes are supported. Moreover, `bind` that specifies the descriptor set and
binding number and `built_in` that specifies SPIR-V `BuiltIn` decoration have
also no conversion.

Currently `llvm.mlir.global`s are created with `private` linkage for
`Private` storage class and `External` for `Input`/`Output` storage classes,
based on SPIR-V spec:

> By default, functions and global variables are private to a module and cannot
be accessed by other modules. However, a module may be written to export or
import functions and global (module scope) variables.

If the global variable's pointer has `Input` storage class, then a `constant`
flag is added to LLVM op:

```mlir
spv.globalVariable @var : !spv.ptr<f32, Input>    =>    llvm.mlir.global external constant @var() : !llvm.float
```

#### `spv.Variable`

Per SPIR-V dialect spec, `spv.Variable` allocates an object in memory, resulting
in a pointer to it, which can be used with `spv.Load` and `spv.Store`. It is
also a function-level variable.

`spv.Variable` is modelled as `llvm.alloca` op. If initialized, an additional
store instruction is used. Note that there is no initialization for arrays and
structs since constants of these types are not supported in LLVM dialect (TODO).
Also, at the moment initialization is only possible via `spv.constant`.

```mlir
// Conversion of VariableOp without initialization
                                                               %size = llvm.mlir.constant(1 : i32) : !llvm.i32
%res = spv.Variable : !spv.ptr<vector<3xf32>, Function>   =>   %res  = llvm.alloca  %size x !llvm.vec<3 x float> : (!llvm.i32) -> !llvm.ptr<vec<3 x float>>

// Conversion of VariableOp with initialization
                                                               %c    = llvm.mlir.constant(0 : i64) : !llvm.i64
%c   = spv.constant 0 : i64                                    %size = llvm.mlir.constant(1 : i32) : !llvm.i32
%res = spv.Variable init(%c) : !spv.ptr<i64, Function>    =>   %res	 = llvm.alloca %[[SIZE]] x !llvm.i64 : (!llvm.i32) -> !llvm.ptr<i64>
                                                               llvm.store %c, %res : !llvm.ptr<i64>
```

Note that simple conversion to `alloca` may not be sufficent if the code has
some scoping. For example, if converting ops executed in a loop into `alloca`s,
a stack overflow may occur. For this case, `stacksave`/`stackrestore` pair can
be used (TODO).

### Miscellaneous ops with direct conversions

There are multiple SPIR-V ops that do not fit in a particular group but can be
converted directly to LLVM dialect. Their conversion is addressed in this
section.

SPIR-V Dialect op                     | LLVM Dialect op
:-----------------------------------: | :-----------------------------------:
`spv.Select`                          | `llvm.select`
`spv.Undef`                           | `llvm.mlir.undef`

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
%res0 = spv.ShiftRightArithmetic %0, %2 : i32, i32  =>  %res0 = llvm.ashr %0, %2 : !llvm.i32

// Shift with extension
                                                        %ext  = llvm.sext %1 : !llvm.i16 to !llvm.i32
%res1 = spv.ShiftRightArithmetic %0, %1 : i32, i16  =>  %res1 = llvm.ashr %0, %ext: !llvm.i32
```

### `spv.constant`

At the moment `spv.constant` conversion supports scalar and vector constants
**only**.

#### Mapping

`spv.constant` is mapped to `llvm.mlir.constant`. This is a straightforward
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
// %0 = llvm.mlir.constant(0 : i8) : !llvm.i8
%0 = spv.constant  0 : i8

// %1 = llvm.mlir.constant(dense<[2, 3, 4]> : vector<3xi32>) : !llvm.vec<3 x i32>
%1 = spv.constant dense<[2, 3, 4]> : vector<3xui32>
```

### Not implemented ops

There is no support of the following ops:

*   All Atomic ops
*   All matrix ops
*   All GroupNonUniform ops

As well as:

*   spv.AccessChain
*   spv.CompositeConstruct
*   spv.CompositeExtract
*   spv.CompositeInsert
*   spv.ControlBarrier
*   spv.CopyMemory
*   spv.EntryPoint
*   spv.ExecutionMode
*   spv.FMod
*   spv.GLSL.SAbs
*   spv.GLSL.SSign
*   spv.GLSL.FSign
*   spv.MemoryBarrier
*   spv._reference_of
*   spv.SMod
*   spv.specConstant
*   spv.SubgroupBallotKHR
*   spv.Unreachable

## Control flow conversion

### Branch ops

`spv.Branch` and `spv.BranchConditional` are mapped to `llvm.br` and
`llvm.cond_br`. Branch weigths for `spv.BranchConditional` are mapped to
coresponding `branch_weights` attribute of `llvm.cond_br`. When translated to
proper LLVM, `branch_weights` are converted into LLVM metadata associated with
the conditional branch.

### `spv.FunctionCall`

`spv.FunctionCall` maps to `llvm.call`. For example:

```mlir
%0 = spv.FunctionCall @foo() : () -> i32    =>    %0 = llvm.call @foo() : () -> !llvm.float
spv.FunctionCall @bar(%0) : (i32) -> ()     =>    llvm.call @bar(%0) : (!llvm.float) -> ()
```

### `spv.selection` and `spv.loop`

Control flow within `spv.selection` and `spv.loop` is lowered directly to LLVM
via branch ops. The conversion can only be applied to selection or loop with all
blocks being reachable. Moreover, selection and loop control attributes (such as
`Flatten` or `Unroll`) are not supported at the moment.

```mlir
// Conversion of selection
%cond = spv.constant true                               %cond = llvm.mlir.constant(true) : !llvm.i1
spv.selection {
  spv.BranchConditional %cond, ^true, ^false            llvm.cond_br %cond, ^true, ^false

^true:																								^true:
  // True block code                                    // True block code
  spv.Branch ^merge                             =>      llvm.br ^merge

^false:                                               ^false:
  // False block code                                   // False block code
  spv.Branch ^merge                                     llvm.br ^merge

^merge:                                               ^merge:
  spv._merge                                            llvm.br ^continue
}
// Remaining code																			^continue:
                                                        // Remaining code
```

```mlir
// Conversion of loop
%cond = spv.constant true                               %cond = llvm.mlir.constant(true) : !llvm.i1
spv.loop {
  spv.Branch ^header                                    llvm.br ^header

^header:                                              ^header:
  // Header code                                        // Header code
  spv.BranchConditional %cond, ^body, ^merge    =>      llvm.cond_br %cond, ^body, ^merge

^body:                                                ^body:
  // Body code                                          // Body code
  spv.Branch ^continue                                  llvm.br ^continue

^continue:                                            ^continue:
  // Continue code                                      // Continue code
  spv.Branch ^header                                    llvm.br ^header

^merge:                                               ^merge:
  spv._merge                                            llvm.br ^remaining
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

SPIR-V Dialect op                     | LLVM Dialect op
:-----------------------------------: | :-----------------------------------:
`spv.GLSL.Ceil`                       | `llvm.intr.ceil`
`spv.GLSL.Cos`                        | `llvm.intr.cos`
`spv.GLSL.Exp`                        | `llvm.intr.exp`
`spv.GLSL.FAbs`                       | `llvm.intr.fabs`
`spv.GLSL.Floor`                      | `llvm.intr.floor`
`spv.GLSL.FMax`                       | `llvm.intr.maxnum`
`spv.GLSL.FMin`                       | `llvm.intr.minnum`
`spv.GLSL.Log`                        | `llvm.intr.log`
`spv.GLSL.Sin`                        | `llvm.intr.sin`
`spv.GLSL.Sqrt`                       | `llvm.intr.sqrt`
`spv.GLSL.SMax`                       | `llvm.intr.smax`
`spv.GLSL.SMin`                       | `llvm.intr.smin`

### Special cases

`spv.InverseSqrt` is mapped to:
```mlir
                                           %one  = llvm.mlir.constant(1.0 : f32) : !llvm.float
%res = spv.InverseSqrt %arg : f32    =>    %sqrt = "llvm.intr.sqrt"(%arg) : (!llvm.float) -> !llvm.float
                                           %res  = fdiv %one, %sqrt : !llvm.float
```

`spv.Tan` is mapped to:
```mlir
                                   %sin = "llvm.intr.sin"(%arg) : (!llvm.float) -> !llvm.float
%res = spv.Tan %arg : f32    =>    %cos = "llvm.intr.cos"(%arg) : (!llvm.float) -> !llvm.float
                                   %res = fdiv %sin, %cos : !llvm.float
```

`spv.Tanh` is modelled using the equality `tanh(x) = {exp(2x) - 1}/{exp(2x) + 1}`:
```mlir
                                     %two   = llvm.mlir.constant(2.0: f32) : !llvm.float
                                     %2xArg = llvm.fmul %two, %arg : !llvm.float
                                     %exp   = "llvm.intr.exp"(%2xArg) : (!llvm.float) -> !llvm.float
%res = spv.Tanh %arg : f32     =>    %one   = llvm.mlir.constant(1.0 : f32) : !llvm.float
                                     %num   = llvm.fsub %exp, %one : !llvm.float
                                     %den   = llvm.fadd %exp, %one : !llvm.float
                                     %res   = llvm.fdiv %num, %den : !llvm.float
```

## Function conversion and related ops

This section describes the conversion of function-related operations from SPIR-V
to LLVM dialect.

### `spv.func`
This op declares or defines a SPIR-V function and it is converted to `llvm.func`.
This conversion handles signarture conversion, and function control attributes
remapping to LLVM dialect function [`passthrough` attribute](Dialects/LLVM.md#Attribute-pass-through).

The following mapping is used to map [SPIR-V function control](SPIRVFunctionAttributes) to
[LLVM function attributes](LLVMFunctionAttributes):

SPIR-V Function Control Attributes    | LLVM Function Attributes
:-----------------------------------: | :-----------------------------------:
None                                  | No function attributes passed
Inline                                | `alwaysinline`
DontInline                            | `noinline`
Pure                                  | `readonly`
Const                                 | `readnone`

### `spv.Return` and `spv.ReturnValue`

In LLVM IR, functions may return either 1 or 0 value. Hence, we map both ops to
`llvm.return` with or without a return value.

## Module ops

Module in SPIR-V has one region that contains one block. It is defined via
`spv.module` op that also takes a range of attributes:

*   Addressing model
*   Memory model
*   Version-Capability-Extension attribute

`spv.module` is converted into `ModuleOp`. This plays a role of enclosing scope
to LLVM ops. At the moment, SPIR-V module attributes are ignored.

`spv._module_end` is mapped to an equivalent terminator `ModuleTerminatorOp`.

## SPIR-V special ops

**Note: this section is due to be implemented in August**

This section describes how SPIR-V specific ops, *e.g* `spv.specConstant`, are
modelled in LLVM. It also provides information on `mlir-spirv-runner`.

[LLVMFunctionAttributes]: https://llvm.org/docs/LangRef.html#function-attributes
[SPIRVFunctionAttributes]: https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#_a_id_function_control_a_function_control
