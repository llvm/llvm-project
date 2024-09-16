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
