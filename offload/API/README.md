# Offload API definitions

**Note**: This is a work-in-progress. The intention is for this to serve as a
starting off point for design discussion. It is loosely based on equivalent
tooling in Unified Runtime.

The Tablegen files in this directory are used to define the Offload API. They
are used with the `offload-tblgen` tool to generate API headers and (stub)
validation code. There are plans to add support for tracing, printing (e.g. 
adding `operator<<(std::ostream)` defs to API structs, enums, etc), and test
generation.

The root file is `OffloadAPI.td` - additional `.td` files can be included in
this file to add them to the API.

## API Objects
The API consists of a number of objects, which always have a *name* field and
*description* field, and are one of the following types:

### Function
Represents an API entry point function. Has a list of returns and parameters.
Also has fields for details (representing a bullet-point list of
information about the function that would otherwise be too detailed for the
description), and analogues (equivalent functions in other APIs).

#### Parameter
Represents a parameter to a function, has *type*, *name*, and *desc* fields.
Also has a *flags* field containing flags representing whether the parameter is
in, out, or optional.

The *type* field is used to infer if the parameter is a pointer or handle type.
A *handle* type is a pointer to an opaque struct, used to abstract over
plugin-specific implementation details.

#### Return
A return represents a possible return code from the function, and optionally a
list of conditions in which this value may be returned. The conditions list is
not expected to be exhaustive. A condition is considered free-form text, but
if it is wrapped in \`backticks\` then it is treated as literal code
representing an error condition (e.g. `someParam < 1`). These conditions are
used to automatically create validation checks by the `offload-tblgen`
validation generator.

Returns are automatically generated for functions with pointer or handle
parameters, so API authors do not need to exhaustively add null checks for
these types of parameters. All functions also get a number of default return
values automatically.


### Struct
Represents a struct. Contains a list of members, which each have a *type*,
*name*, and *desc*.

Also optionally takes a *base_class* field. If this is either of the special
`ol_base_properties_t` or `ol_base_desc_t` structs, then the struct will inherit
members from those structs. The generated struct does **not** use actual C++
inheritance, but instead explicitly has those members copied in, which preserves
compatibility with C.

### Enum
Represents a C-style enum. Contains a list of `etor` values.

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
generating documentation from them hasn't been tested yet.

### Validation functions
```
./offload-tblgen -I <path-to-llvm>/offload/API  <path-to-llvm>/offload/API/OffloadAPI.td --gen-validation
```
The functions are partially stubbed and are designed to be used in conjunction
with code that can track live handle references, etc. See the equivalent code
in Unified Runtime for an idea of how this might work.

### Future Tablegen backends
`RecordTypes.hpp` contains wrappers for all of the API object types, which will
allow more backends to be easily added in future.
