# In LLVMBuild.txt files, it is not possible to mark a dependency to a
# library as optional. So instead, generate an empty library if we did
# not ask for it.
add_llvm_library(LLVMGlobalISel
        CallLowering.cpp
        GlobalISel.cpp
        IRTranslator.cpp
        InstructionSelect.cpp
        InstructionSelector.cpp
        LegalizerHelper.cpp
        Legalizer.cpp
        LegalizerInfo.cpp
        Localizer.cpp
        MachineIRBuilder.cpp
        RegBankSelect.cpp
        RegisterBank.cpp
        RegisterBankInfo.cpp
        Utils.cpp

        DEPENDS
        intrinsics_gen
  )
