# This library is unusual, since mlir-tblgen depends on it, which is
# built with DISABLE_LLVM_LINK_LLVM_DYLIB, this must also be built
# with that option.  Otherwise builds with LLVM_BUILD_LLVM_DYLIB and
# LLVM_LINK_LLVM_DYLIB fail.  (Note that even if this has no llvm
# component dependencies, LLVM_LINK_LLVM_DYLIB tends to introduce a
# dependence on libLLVM.so)  However, it must also be linkable against
# libMLIR.so in some contexts (see unittests/Tablegen, for instance, which
# has a dependence on MLIRIR, which must depend on libLLVM.so).  This works
# in this special case because this library is static.
#
# In order for this arrangement to be sound, it must not depend on libraries
# that may be transitively included in libMLIR.so or libLLVM.so. Specifically,
# this means that MLIRSupport (outside of header-only access) cannot be used.
llvm_add_library(MLIRTableGen STATIC
  Argument.cpp
  Attribute.cpp
  AttrOrTypeDef.cpp
  Builder.cpp
  Class.cpp
  CodeGenHelpers.cpp
  Constraint.cpp
  Dialect.cpp
  Format.cpp
  GenInfo.cpp
  Interfaces.cpp
  Operator.cpp
  Pass.cpp
  Pattern.cpp
  Predicate.cpp
  Property.cpp
  Region.cpp
  SideEffects.cpp
  Successor.cpp
  Trait.cpp
  Type.cpp

  DISABLE_LLVM_LINK_LLVM_DYLIB

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/TableGen
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Support
)

mlir_check_all_link_libraries(MLIRTableGen)

add_mlir_library_install(MLIRTableGen)
