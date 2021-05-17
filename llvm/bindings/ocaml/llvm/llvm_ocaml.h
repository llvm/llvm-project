/*===-- llvm_ocaml.h - LLVM OCaml Glue --------------------------*- C++ -*-===*\
|*                                                                            *|
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM          *|
|* Exceptions.                                                                *|
|* See https://llvm.org/LICENSE.txt for license information.                  *|
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception                    *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This file glues LLVM's OCaml interface to its C interface. These functions *|
|* are by and large transparent wrappers to the corresponding C functions.    *|
|*                                                                            *|
|* Note that these functions intentionally take liberties with the CAMLparamX *|
|* macros, since most of the parameters are not GC heap objects.              *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef LLVM_LLVM_OCAML_H
#define LLVM_LLVM_OCAML_H

#include "caml/alloc.h"
#include "caml/custom.h"
#include "caml/version.h"

#if OCAML_VERSION < 41200
/* operations on OCaml option values, defined by OCaml 4.12 */
#define Val_none Val_int(0)
#define Some_val(v) Field(v, 0)
#define Tag_some 0
#define Is_none(v) ((v) == Val_none)
#define Is_some(v) Is_block(v)
value caml_alloc_some(value);
#endif

/* Convert a C pointer to an OCaml option */
value ptr_to_option(void *Ptr);

/* Convert a C string into an OCaml string */
value cstr_to_string(const char *Str, mlsize_t Len);

#endif // LLVM_LLVM_OCAML_H
