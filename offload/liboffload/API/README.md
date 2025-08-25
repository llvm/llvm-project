# Offload API definitions

The Tablegen files in this directory are used to define the Offload API. They
are used with the `offload-tblgen` tool to generate API headers, print headers,
and other implementation details.

The root file is `OffloadAPI.td` - additional `.td` files can be included in
this file to add them to the API.

## Modifying the API

API modifications, including additions, can be made by modifying the existing
`.td` files. It is also possible to add a new tablegen file to the API by adding
it to the includes in `OffloadAPI.td`. When Offload is rebuilt the new
definition will be included in the generated files.

Most API changes and additions do not require any additional work beyond this,
other than new functions which are described below.

### Adding a new function to the API

When a new function is added (e.g. `offloadDeviceFoo`), the actual entry
point is automatically generated, which contains validation and tracing code.
It expects an implementation function (`offloadDeviceFoo_impl`) to be defined,
which it will call into. The definition of this implementation function should
be added to `liboffload/src/OffloadImpl.cpp`

In short, the steps to add a new function are:
* Add the new function definition to the `.td` files.
* Build the `LLVMOffload` target. The relevant files will be regenerated, but
  the library will fail to link because it is missing the implementation
  function.
* Add the new implementation function to `liboffload/src/OffloadImpl.cpp`. You
  can copy the new function declaration from the generated
  `OffloadImplFuncDecls.inc` file.
* Rebuild `LLVMOffload`

## API Objects

The API consists of a number of objects, which always have a *name* field and
*description* field, and are one of the following types:

### Function

Represents an API entry point function. Has a list of returns and parameters.
Also has fields for details (representing a bullet-point list of information
about the function that would otherwise be too detailed for the description),
and analogues (equivalent functions in other APIs).

#### Parameter

Represents a parameter to a function, has *type*, *name*, and *desc* fields.
Also has a *flags* field containing flags representing whether the parameter is
in, out, or optional.

The *type* field is used to infer if the parameter is a pointer or handle type.
A *handle* type is a pointer to an opaque struct, used to abstract over
plugin-specific implementation details.

There are two special variants of a *parameter*:
* **RangedParameter** - Represents a parameter that has a range described by
  other parameters. Generally these are pointers to an arbitrary number of
  objects. The range is used for generating validation and printing code. E.g,
  a range might be between `(0, NumDevices)`
* **TypeTaggedParameter** - Represents a parameter (usually of `void*` type)
  that has the type and size of its pointee data described by other function
  parameters. The type is usually described by a type-tagged enum. This allows
  functions (e.g. `olGetDeviceInfo`) to return data of an arbitrary type.

#### Return

A return represents a possible return code from the function, and optionally a
list of conditions in which this value may be returned. The conditions list is
not expected to be exhaustive. A condition is considered free-form text, but if
it is wrapped in \`backticks\` then it is treated as literal code representing
an error condition (e.g. `someParam < 1`). These conditions are used to
automatically create validation checks by the `offload-tblgen` validation
generator.

Returns are automatically generated for functions with pointer or handle
parameters, so API authors do not need to exhaustively add null checks for
these types of parameters. All functions also get a number of default return
values automatically.


### Struct

Represents a struct. Contains a list of members, which each have a *type*,
*name*, and *desc*.

Also optionally takes a *base_class* field. If this is either of the special
`offload_base_properties_t` or `offload_base_desc_t` structs, then the struct
will inherit members from those structs. The generated struct does **not** use
actual C++ inheritance, but instead explicitly has those members copied in,
which preserves ABI compatibility with C.

### Enum

Represents a C-style enum. Contains a list of `etor` values, which have a name
and description.

A `TaggedEtor` record type also exists which additionally takes a type. This
type is used when the enum is used as a parameter to a function with a
type-tagged function parameter (e.g. `olGetDeviceInfo`).

All enums automatically get a `<enum_name>_FORCE_UINT32 = 0x7fffffff` value,
which forces the underlying type to be uint32.

### Handle

Represents a pointer to an opaque struct, as described in the Parameter section.
It does not take any extra fields.

### Typedef

Represents a typedef, contains only a *value* field.

### Macro

Represents a C preprocessor `#define`. Contains a *value* field. Optionally
takes a *condition* field, which allows the macro to be conditionally defined,
and an *alt_value* field, which represents the value if the condition is false.

Macro arguments are presented in the *name* field (e.g. name = `mymacro(arg)`).

While there may seem little point generating a macro from tablegen, doing this
allows the entire source of the header file to be generated from the tablegen
files, rather than requiring a mix of C source and tablegen.

## Generation

### API header

```
./offload-tblgen -I <path-to-llvm>/offload/API  <path-to-llvm>/offload/API/OffloadAPI.td --gen-api
```
The comments in the generated header are in Doxygen format, although
generating documentation from them hasn't been implemented yet.

The entirety of this header is generated by Tablegen, rather than having a
predefined header file that includes one or more `.inc` files. This is because
this header is expected to be part of the installation and distributed to
end-users, so should be self-contained.

### Entry Points

```
./offload-tblgen -I <path-to-llvm>/offload/API  <path-to-llvm>/offload/API/OffloadAPI.td --gen-entry-points
```
These functions form the actual Offload interface, and are wrappers over the
functions that contain the actual implementation (see
'Adding a new entry point').

They implement automatically generated validation checks, and tracing of
function calls with arguments and results. The tracing can be enabled with the
`OFFLOAD_TRACE` environment variable.

### Implementation function declarations

```
./offload-tblgen -I <path-to-llvm>/offload/API  <path-to-llvm>/offload/API/OffloadAPI.td --gen-impl-func-decls
```
Generates declarations of the implementation of functions of every entry point
in the API, e.g. `offloadDeviceFoo_impl` for `offloadDeviceFoo`.

### Print header

```
./offload-tblgen -I <path-to-llvm>/offload/API  <path-to-llvm>/offload/API/OffloadAPI.td --gen-print-header
```
This header contains `llvm::raw_ostream &operator<<(llvm::raw_ostream &)`
definitions for various API objects, including function parameters.

As with the API header, it is expected that this header is part of the installed
package, so it is entirely generated by Tablegen.

For ease of implementation, and since it is not strictly part of the API, this
is a C++ header file. If a C version is desirable it could be added.

### Additional Tablegen backends

`RecordTypes.hpp` contains wrappers for all of the API object types, which
allows new backends to be easily added if needed.
