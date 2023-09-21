add_library(GenISAIntrinsics STATIC IMPORTED)
set_target_properties(GenISAIntrinsics
  PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/libGenISAIntrinsics.a
)

add_mlir_translation_library(MLIRGENXToLLVMIRTranslation
  GENXToLLVMIRTranslation.cpp

  DEPENDS
  MLIRGENXConversionsIncGen
  GenISAIntrinsics

  LINK_COMPONENTS
  Core
  CodeGen
  Support

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRGENXDialect
  MLIRSupport
  MLIRTargetLLVMIRExport
  GenISAIntrinsics
)
