//===- AccDefs.td - OpenACC runtime API record types --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A type that specifies its fortran spelling along with its c binding spelling.
// `ignore_tkr`, if non-empty, is the letter list for `!dir$ ignore_tkr (...)`
// (e.g. "k" or "tkr"). `print_fn` is an optional function name that is used to
// wrap values of this type when printing it out.
class AccType<string c, string fortran, string igntkr = "", string print = ""> {
  string c_spelling = c;
  string f_spelling = fortran;
  string ignore_tkr = igntkr;
  string print_fn = print;
}

// A parameter that appears in a function signature.
class Param<AccType ty, string param_name, bit value = 0, string intent = ""> {
  AccType type = ty;
  string name = param_name;
  bit f_value = value;
  string f_intent = intent;
}

class ValParam<AccType ty, string param_name>
    : Param<ty, param_name, 1, "">;

class RefParam<AccType ty, string param_name>
    : Param<ty, param_name, 0, "">;

class InParam<AccType ty, string param_name>
    : Param<ty, param_name, 0, "in">;

class OutParam<AccType ty, string param_name>
    : Param<ty, param_name, 0, "out">;

class Signature<AccType result_ty, list<Param> param_list> {
  AccType result = result_ty;
  list<Param> params = param_list;
}

// Fortran API definition. `name` is the name of the generic interface. `prefix`
// is prepended to the name of the specific procedure and to its binding label.
// `alias_names` are further generic interface names (such as acc_pcopyin for
// acc_copyin) that resolve to the same specific procedure.
class AccFFunc<string fname, Signature sig, list<string> aliases = []> {
  string name = fname;
  Signature signature = sig;
  list<string> alias_names = aliases;
  string prefix = "";
}

// A Fortran API routine that takes its data argument as an assumed-rank array
// (a CFI descriptor) instead of an address and a byte count. Its specific
// procedure and binding label are prefixed with `cfi_`.
class AccCfiFFunc<string fname, Signature sig, list<string> aliases = []>
    : AccFFunc<fname, sig, aliases> {
  let prefix = "cfi_";
}

// A Fortran kind parameter of the openacc module. `name` is the parameter name
// (e.g. acc_handle_kind) and `value` its value (a literal or a named constant).
class FortranKind<string kname, string kvalue> {
  string name = kname;
  string value = kvalue;
}

// One enumerator of an AccEnum, with its name and value.
class Enumerator<string nm, int val> {
  string name = nm;
  int value = val;
}

// An enumeration shared between C and Fortran. `c_name` is the name of the C
// enum type, `fortran_kind` the kind of the Fortran integer parameters that
// are generated for its enumerators, and `enumerators` its members.
class AccEnum<string cname, FortranKind kind, list<Enumerator> etors> {
  string c_name = cname;
  FortranKind fortran_kind = kind;
  list<Enumerator> enumerators = etors;
}

// A named constant of the OpenACC API that is a macro in C and a parameter of
// the given Fortran kind in Fortran (e.g. acc_async_sync).
class AccConst<string cname, FortranKind kind, int val> {
  string name = cname;
  FortranKind fortran_kind = kind;
  int value = val;
}

// A Fortran-only integer parameter of the openacc module (e.g.
// openacc_version).
class FortranInt<string iname, int val> {
  string name = iname;
  int value = val;
}

// A symbol that the shared library exports, listed in the generated linker
// version script.
class AccExport<string ename> {
  string name = ename;
}
