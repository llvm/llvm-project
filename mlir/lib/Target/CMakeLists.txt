add_mlir_translation_library(MLIRTargetLLVMIRModuleTranslation
  LLVMIR/DebugTranslation.cpp
  LLVMIR/ModuleTranslation.cpp
  LLVMIR/TypeTranslation.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Target/LLVMIR

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Core
  FrontendOpenMP
  TransformUtils

  LINK_LIBS PUBLIC
  MLIRLLVMIR
  MLIROpenMP
  MLIRLLVMIRTransforms
  MLIRTranslation
  )

add_mlir_translation_library(MLIRTargetAVX512
  LLVMIR/LLVMAVX512Intr.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Target/LLVMIR

  DEPENDS
  MLIRLLVMAVX512ConversionsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMAVX512
  MLIRLLVMIR
  MLIRTargetLLVMIRModuleTranslation
  )

add_mlir_translation_library(MLIRTargetLLVMIR
  LLVMIR/ConvertFromLLVMIR.cpp
  LLVMIR/ConvertToLLVMIR.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Target/LLVMIR

  LINK_COMPONENTS
  Core
  IRReader

  LINK_LIBS PUBLIC
  MLIRTargetLLVMIRModuleTranslation
  )

add_mlir_translation_library(MLIRTargetArmNeon
  LLVMIR/LLVMArmNeonIntr.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Target/LLVMIR

  DEPENDS
  MLIRLLVMArmNeonConversionsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMArmNeon
  MLIRLLVMIR
  MLIRTargetLLVMIRModuleTranslation
  )

add_mlir_translation_library(MLIRTargetNVVMIR
  LLVMIR/ConvertToNVVMIR.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Target/LLVMIR

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRGPU
  MLIRIR
  MLIRLLVMIR
  MLIRNVVMIR
  MLIRTargetLLVMIRModuleTranslation
  )

add_mlir_translation_library(MLIRTargetROCDLIR
  LLVMIR/ConvertToROCDLIR.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Target/LLVMIR

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRGPU
  MLIRIR
  MLIRLLVMIR
  MLIRROCDLIR
  MLIRTargetLLVMIRModuleTranslation
  )

add_mlir_translation_library(MLIRSPIRVBinaryUtils
  SPIRV/SPIRVBinaryUtils.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSPIRV
  MLIRSupport
  )

add_mlir_translation_library(MLIRSPIRVSerialization
  SPIRV/Serialization.cpp

  DEPENDS
  MLIRSPIRVSerializationGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSPIRV
  MLIRSPIRVBinaryUtils
  MLIRSupport
  MLIRTranslation
  )

add_mlir_translation_library(MLIRSPIRVDeserialization
  SPIRV/Deserialization.cpp

  DEPENDS
  MLIRSPIRVSerializationGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSPIRV
  MLIRSPIRVBinaryUtils
  MLIRSupport
  MLIRTranslation
  )

add_mlir_translation_library(MLIRSPIRVTranslateRegistration
  SPIRV/TranslateRegistration.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSPIRV
  MLIRSPIRVSerialization
  MLIRSPIRVDeserialization
  MLIRSupport
  MLIRTranslation
  )
