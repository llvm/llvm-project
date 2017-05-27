# List of all GlobalISel files.
set(GLOBAL_ISEL_FILES
      CallLowering.cpp
      IRTranslator.cpp
      InstructionSelect.cpp
      InstructionSelector.cpp
      MachineIRBuilder.cpp
      LegalizerHelper.cpp
      Legalizer.cpp
      LegalizerInfo.cpp
      Localizer.cpp
      RegBankSelect.cpp
      RegisterBank.cpp
      RegisterBankInfo.cpp
      Utils.cpp
      )

# Add GlobalISel files to the dependencies if the user wants to build it.
if(LLVM_BUILD_GLOBAL_ISEL)
  set(GLOBAL_ISEL_BUILD_FILES ${GLOBAL_ISEL_FILES})
else()
  set(GLOBAL_ISEL_BUILD_FILES"")
  set(LLVM_OPTIONAL_SOURCES LLVMGlobalISel ${GLOBAL_ISEL_FILES})
endif()

# In LLVMBuild.txt files, it is not possible to mark a dependency to a
# library as optional. So instead, generate an empty library if we did
# not ask for it. 
add_llvm_library(LLVMGlobalISel
        ${GLOBAL_ISEL_BUILD_FILES}
        GlobalISel.cpp
        
        DEPENDS
        intrinsics_gen
  )
