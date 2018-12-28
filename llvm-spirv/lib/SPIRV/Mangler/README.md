Contributed by: Intel Corporation.

SPIR Name Mangler
=================

The NameMangler Library Converts the given function descriptor to a string
that represents the function's prototype.

The mangling algorithm is based on clang 3.0 Itanium mangling algorithm
(http://sourcery.mentor.com/public/cxx-abi/abi.html#mangling).

The algorithm is adapted to support mangling of SPIR built-in
functions and was tested on SPIR built-ins only.

The mangler supports mangling according to SPIR 1.2 and SPIR 2.0
For usage examples see unittest/spir_name_mangler.
