# There is no clear way of keeping track of compiler command-line
# options chosen via `add_definitions', so we need our own method for
# using it on tools/llvm-config/CMakeLists.txt.

# Beware that there is no implementation of remove_llvm_definitions.

macro(add_llvm_definitions)
  # We don't want no semicolons on LLVM_DEFINITIONS:
  foreach(arg ${ARGN})
    set(LLVM_DEFINITIONS "${LLVM_DEFINITIONS} ${arg}")
  endforeach(arg)
  add_definitions( ${ARGN} )
endmacro(add_llvm_definitions)
