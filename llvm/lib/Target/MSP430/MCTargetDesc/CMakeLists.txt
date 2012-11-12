add_llvm_library(LLVMMSP430Desc
  MSP430MCTargetDesc.cpp
  MSP430MCAsmInfo.cpp
  )

add_dependencies(LLVMMSP430Desc MSP430CommonTableGen)
