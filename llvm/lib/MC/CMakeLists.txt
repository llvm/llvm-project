set(MCSOURCES
  ConstantPools.cpp
  ELFObjectWriter.cpp
  MCAsmBackend.cpp
  MCAsmInfo.cpp
  MCAsmInfoELF.cpp
  MCAsmMacro.cpp
  MCAsmStreamer.cpp
  MCAssembler.cpp
  MCCodeEmitter.cpp
  MCCodeView.cpp
  MCContext.cpp
  MCDwarf.cpp
  MCELFObjectTargetWriter.cpp
  MCELFStreamer.cpp
  MCExpr.cpp
  MCFragment.cpp
  MCInst.cpp
  MCInstPrinter.cpp
  MCInstrAnalysis.cpp
  MCInstrDesc.cpp
  MCInstrInfo.cpp
  MCLabel.cpp
  MCLinkerOptimizationHint.cpp
  MCNullStreamer.cpp
  MCObjectFileInfo.cpp
  MCObjectStreamer.cpp
  MCObjectWriter.cpp
  MCPseudoProbe.cpp
  MCRegisterInfo.cpp
  MCSchedule.cpp
  MCSection.cpp
  MCSectionELF.cpp
  MCStreamer.cpp
  MCSubtargetInfo.cpp
  MCSymbol.cpp
  MCSymbolELF.cpp
  MCTargetOptions.cpp
  MCTargetOptionsCommandFlags.cpp
  MCValue.cpp
  StringTableBuilder.cpp
  TargetRegistry.cpp
  )

# OS/format-specific sources not needed for bare-metal EJIT
set(MCSOURCES_OPT
  DXContainerPSVInfo.cpp
  DXContainerRootSignature.cpp
  GOFFObjectWriter.cpp
  MCAsmInfoCOFF.cpp
  MCAsmInfoDarwin.cpp
  MCAsmInfoGOFF.cpp
  MCAsmInfoWasm.cpp
  MCAsmInfoXCOFF.cpp
  MCDXContainerStreamer.cpp
  MCDXContainerWriter.cpp
  MCGOFFStreamer.cpp
  MCMachOStreamer.cpp
  MCMachObjectTargetWriter.cpp
  MCSectionCOFF.cpp
  MCSectionDXContainer.cpp
  MCSectionGOFF.cpp
  MCSectionMachO.cpp
  MCSectionWasm.cpp
  MCSectionXCOFF.cpp
  MCSPIRVStreamer.cpp
  MCSymbolXCOFF.cpp
  MCWasmObjectTargetWriter.cpp
  MCWasmStreamer.cpp
  MCWin64EH.cpp
  MCWinCOFFStreamer.cpp
  MCWinEH.cpp
  MCXCOFFObjectTargetWriter.cpp
  MCXCOFFStreamer.cpp
  MachObjectWriter.cpp
  SPIRVObjectWriter.cpp
  WasmObjectWriter.cpp
  WinCOFFObjectWriter.cpp
  XCOFFObjectWriter.cpp
  )

if(NOT EJIT_BARE_METAL)
  list(APPEND MCSOURCES ${MCSOURCES_OPT})
endif()

add_llvm_component_library(LLVMMC
  ${MCSOURCES}

  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/MC

  LINK_COMPONENTS
  Support
  TargetParser
  BinaryFormat

  DEPENDS
  intrinsics_gen
  )

add_subdirectory(MCParser)
add_subdirectory(MCDisassembler)
