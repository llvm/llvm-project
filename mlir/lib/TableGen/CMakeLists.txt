# This library is unusual, since mlir-tblgen depends on it.
# For non-obvious reasons, linking mlir-tblgen fails with
# LLVM_BUILD_LLVM_DYLIB and LLVM_LINK_LLVM_DYLIB unless
# DISABLE_LLVM_LINK_LLVM_DYLIB is set.
llvm_add_library(LLVMMLIRTableGen STATIC
  Argument.cpp
  Attribute.cpp
  Constraint.cpp
  Dialect.cpp
  Format.cpp
  Operator.cpp
  OpClass.cpp
  OpInterfaces.cpp
  OpTrait.cpp
  Pass.cpp
  Pattern.cpp
  Predicate.cpp
  Region.cpp
  SideEffects.cpp
  Successor.cpp
  Type.cpp

  DISABLE_LLVM_LINK_LLVM_DYLIB

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/TableGen

  LINK_COMPONENTS
  TableGen
  Demangle
  )

mlir_check_all_link_libraries(LLVMMLIRTableGen)
