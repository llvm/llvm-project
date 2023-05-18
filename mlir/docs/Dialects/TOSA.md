# Tensor Operator Set Architecture (TOSA) Dialect

[TOC]

## Rationale

The MLIR TOSA dialect implements the [TOSA
specification](https://developer.mlplatform.org/w/tosa/).  This document
describes the decision process for how TOSA expresses operators in
high level dialects.

TOSA was developed after parallel efforts to rationalize the top-down picture
from multiple high-level frameworks, as well as a bottom-up view of different
hardware target concerns (CPU, GPU and NPU), and reflects a set of choices
that attempt to manage both sets of requirements.

## TOSA and Tensor Level Expressiveness

TOSA endeavors to provide an operator set that tries to fulfil the following
expressiveness goals at the *tensor level of abstraction* :

### Complete

This is driven by the top-down perspective, needing to express as much of
multiple high level frameworks fully in TOSA, as possible. This was originally
done from an operator frequency analysis done upon dozens of high level
networks in different frameworks, to select the most frequently occurring ones
and establish a common set of tensor-level operators that could express them.

TOSA categorizes its operator set into classes and attempts to address major
functional operations at the tensor level, including compute, reduction,
elementwise transformations, comparison and control flow.

### Minimal

This takes the bottom-up approach - keep the TOSA operator set minimal in
order to bound the design of hardware, operator kernels, code generation
strategies and associated considerations that effect the executability of TOSA
content.

In this regard TOSA seeks to avoid creating compound operators, instead
leaving it to compiler backend to fuse multiple TOSA ops if required. This
choice also benefits the numerical precision goal, since it is easier to fuse the
numerical functionality of successive operators, than to split the numerical
functionality of a compound operator.

### Numerical Precision

TOSA began as a means to address operator-level numerical precision for
code generation and hardware development. It therefore incorporates precision
detail into the operator set.

In this regard, TOSA operators are best understood as a combination of the visible
quantization information embedded within an operation, together with the
functional information about how that information is used, as described in the
specification of the operation.

## TOSA Operator Rationale

The general basis of selection of the operator set that constitutes TOSA is
described in the TOSA specification document  under Section 1.3 Operator
Selection. Explanation of the thinking behind some operators is listed here:

### COND\_IF and WHILE\_LOOP

Several neural networks express conditional control flow at the tensor level.
A survey of multiple high level frameworks indicated that conditional if and
a loop construct are common in all major frameworks, with some variation.
Since TOSA endeavors to be complete in expressing tensor level functionality
including control flow, it implements these constructs.

The COND\_IF and WHILE\_LOOP operators implement such structured control
flow forms and should be lowerable to corresponding ops in the scf dialect.
Since the dialect seeks to remain isomorphic with an external, serialized form,
the decision was to keep these ops in the dialect (as opposed to deferring
completely to scf), and this may be re-evaluated if this turns out to not yield
the expected value.

## Using TOSA In A Compiler

The TOSA specification describes each operator in functional detail. It is
expected that compilers that use TOSA will use its builders to construct the
operators so that the quantization information for the operator is correctly
generated.

The functional steps described in the pseudocode of the specification enables
the construction of code generation for that operation, or decisions on the
design of underlying hardware. The functional pseudocode also describes
how the quantization parameters are utilized within the operation.

### Quantization Parameters in Ops vs Tensors

TOSA uses the quantization parameters embedded in the input and output
tensors to construct the quantization attributes that sit within the operator.
Once these attributes are constructed, the quantization information within
the tensors are no longer necessary for code generation.

This enables the tensors to be subsequently interpreted simply as contiguous
buffers containing raw data, with no 'meta information' in the form of the
quantization_type. Precision related manipulation of the input or output are
instead described by the operator itself which describes, for example, when
the zero point is applied, or when the scale multiplication is done.

However, TOSA does *not* eliminate the existing MLIR QuantOps quantization
type information within the tensors; this leaves the choice of how to handle
quantization information, to later backend code generation steps.

Maintaining the ability to overlap these different representations of
quantization parameters (i.e. tensor-carried vs op-carried) is an important
capability when considering progressive lowering between uses that expect one
scheme vs the other.

## Operation definitions

[include "Dialects/TosaOps.md"]
