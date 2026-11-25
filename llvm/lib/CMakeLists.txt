include(LLVM-Build)

# HAS_WERROR_GLOBAL_CTORS and LLVM_HAS_NOGLOBAL_CTOR_MUTEX are set by
# lib/Support/CMakeLists.txt.
if (HAS_WERROR_GLOBAL_CTORS AND LLVM_HAS_NOGLOBAL_CTOR_MUTEX)
  set(LLVM_LIB_WERROR_GLOBAL_CTORS ON)
  string(APPEND CMAKE_CXX_FLAGS " -Werror=global-constructors")
endif()

# `Demangle', `Support' and `TableGen' libraries are added on the top-level
# CMakeLists.txt

add_subdirectory(IR)
add_subdirectory(ABI)
add_subdirectory(FuzzMutate)
add_subdirectory(FileCheck)
add_subdirectory(InterfaceStub)
add_subdirectory(IRPrinter)
add_subdirectory(IRReader)
add_subdirectory(CAS)
add_subdirectory(CGData)
add_subdirectory(CodeGen)
add_subdirectory(CodeGenTypes)
add_subdirectory(BinaryFormat)
add_subdirectory(Bitcode)
add_subdirectory(Bitstream)
add_subdirectory(DWARFLinker)
add_subdirectory(Extensions)
add_subdirectory(Frontend)
add_subdirectory(Transforms)
add_subdirectory(Linker)
add_subdirectory(Analysis)
add_subdirectory(DTLTO)
add_subdirectory(LTO)
add_subdirectory(MC)
add_subdirectory(MCA)
add_subdirectory(ObjCopy)
add_subdirectory(Object)
add_subdirectory(ObjectYAML)
add_subdirectory(Option)
add_subdirectory(Remarks)
add_subdirectory(HTTP)
add_subdirectory(Debuginfod)
add_subdirectory(DebugInfo)
add_subdirectory(DWP)
add_subdirectory(ExecutionEngine)
add_subdirectory(Target)
add_subdirectory(SandboxIR)
add_subdirectory(AsmParser)
add_subdirectory(LineEditor)
add_subdirectory(ProfileData)
add_subdirectory(Passes)
add_subdirectory(Plugins)
add_subdirectory(TargetParser)
add_subdirectory(TextAPI)
add_subdirectory(Telemetry)
add_subdirectory(ToolDrivers)
add_subdirectory(XRay)
if (LLVM_INCLUDE_TESTS)
  add_subdirectory(Testing)
endif()
add_subdirectory(WindowsDriver)
add_subdirectory(WindowsManifest)

# Libraries that still have static initializers. Remove a library once it has
# none, so that it cannot regain them.
if (LLVM_LIB_WERROR_GLOBAL_CTORS)
  foreach(lib
      LLVMAArch64CodeGen
      LLVMAMDGPUCodeGen
      LLVMAMDGPUDesc
      LLVMAMDGPUUtils
      LLVMARMAsmParser
      LLVMARMCodeGen
      LLVMAggressiveInstCombine
      LLVMAnalysis
      LLVMAsmParser
      LLVMAsmPrinter
      LLVMBPFCodeGen
      LLVMBitReader
      LLVMBitWriter
      LLVMCGData
      LLVMCodeGen
      LLVMCore
      LLVMCoroutines
      LLVMDXILBitWriter
      LLVMDebugInfoCodeView
      LLVMDebugInfoLogicalView
      LLVMDebuginfod
      LLVMDirectXCodeGen
      LLVMDirectXDesc
      LLVMFrontendOpenMP
      LLVMFuzzMutate
      LLVMGlobalISel
      LLVMHexagonAsmParser
      LLVMHexagonCodeGen
      LLVMHexagonDesc
      LLVMInstCombine
      LLVMInstrumentation
      LLVMInterpreter
      LLVMJITLink
      LLVMLTO
      LLVMLoongArchCodeGen
      LLVMLoongArchDesc
      LLVMM68kAsmParser
      LLVMM68kCodeGen
      LLVMMC
      LLVMMCJIT
      LLVMMCParser
      LLVMMipsCodeGen
      LLVMNVPTXCodeGen
      LLVMObjCARCOpts
      LLVMObject
      LLVMOrcDebugging
      LLVMOrcJIT
      LLVMOrcShared
      LLVMOrcTargetProcess
      LLVMPasses
      LLVMPowerPCCodeGen
      LLVMPowerPCDesc
      LLVMProfileData
      LLVMRISCVCodeGen
      LLVMRemarks
      LLVMSPIRVCodeGen
      LLVMScalarOpts
      LLVMSelectionDAG
      LLVMSystemZCodeGen
      LLVMTarget
      LLVMTransformUtils
      LLVMVectorize
      LLVMWebAssemblyCodeGen
      LLVMX86CodeGen
      LLVMipo
      )
    if (TARGET ${lib})
      target_compile_options(${lib} PRIVATE -Wno-global-constructors)
    endif()
  endforeach()
endif()

set(LLVMCONFIGLIBRARYDEPENDENCIESINC "${LLVM_BINARY_DIR}/tools/llvm-config/LibraryDependencies.inc")

# Special components which don't have any source attached but aggregate other
# components
add_llvm_component_group(all-targets LINK_COMPONENTS ${LLVM_TARGETS_TO_BUILD})
add_llvm_component_group(Engine)

# The native target may not be enabled when cross compiling
if(TARGET ${LLVM_NATIVE_ARCH})
  add_llvm_component_group(Native LINK_COMPONENTS ${LLVM_NATIVE_ARCH})
  add_llvm_component_group(NativeCodeGen LINK_COMPONENTS ${LLVM_NATIVE_ARCH}CodeGen)
else()
  add_llvm_component_group(Native)
  add_llvm_component_group(NativeCodeGen)
endif()

# Component post-processing
LLVMBuildResolveComponentsLink()
LLVMBuildGenerateCFragment(OUTPUT ${LLVMCONFIGLIBRARYDEPENDENCIESINC})
