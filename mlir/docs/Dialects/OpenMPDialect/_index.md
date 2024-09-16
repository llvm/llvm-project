# 'omp' Dialect

The `omp` dialect is for representing directives, clauses and other definitions
of the [OpenMP programming model](https://www.openmp.org). This directive-based
programming model, defined for the C, C++ and Fortran programming languages,
provides abstractions to simplify the development of parallel and accelerated
programs. All versions of the OpenMP specification can be found
[here](https://www.openmp.org/specifications/).

Operations in this MLIR dialect generally correspond to a single OpenMP
directive, taking arguments that represent their supported clauses, though this
is not always the case. For a detailed information of operations, types and
other definitions in this dialect, refer to the automatically-generated
[ODS Documentation](ODS.md).

[TOC]

## Operation Naming Conventions

This section aims to standardize how dialect operation names are chosen, to
ensure a level of consistency. There are two categories of names: tablegen names
and assembly names. The former also corresponds to the C++ class that is
generated for the operation, whereas the latter is used to represent it in MLIR
text form.

Tablegen names are CamelCase, with the first letter capitalized and an "Op"
suffix, whereas assembly names are snake_case, with all lowercase letters and
words separated by underscores.

If the operation corresponds to a directive, clause or other kind of definition
in the OpenMP specification, it must use the same name split into words in the
same way. For example, the `target data` directive would become `TargetDataOp` /
`omp.target_data`, whereas `taskloop` would become `TaskloopOp` /
`omp.taskloop`.

Operations intended to carry extra information for another particular operation
or clause must be named after that other operation or clause, followed by the
name of the additional information. The assembly name must use a period to
separate both parts. For example, the operation used to define some extra
mapping information is named `MapInfoOp` / `omp.map.info`. The same rules are
followed if multiple operations are created for different variants of the same
directive, e.g. `atomic` becomes `Atomic{Read,Write,Update,Capture}Op` /
`omp.atomic.{read,write,update,capture}`.
