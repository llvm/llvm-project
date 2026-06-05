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

# MachO/COFF — needed by x86 (macOS/Windows), not needed by AArch64 bare-metal
set(MCSOURCES_MACHO_COFF
  MCAsmInfoCOFF.cpp
  MCAsmInfoDarwin.cpp
  MCMachOStreamer.cpp
  MCMachObjectTargetWriter.cpp
  MCSectionCOFF.cpp
  MCSectionMachO.cpp
  MCWin64EH.cpp
  MCWinCOFFStreamer.cpp
  MCWinEH.cpp
  MachObjectWriter.cpp
  WinCOFFObjectWriter.cpp
  )

# GOFF/SPIRV/DXContainer/Wasm/XCOFF — not needed by any EJIT bare-metal arch
set(MCSOURCES_BLOAT
  DXContainerPSVInfo.cpp
  DXContainerRootSignature.cpp
  GOFFObjectWriter.cpp
  MCAsmInfoGOFF.cpp
  MCAsmInfoWasm.cpp
  MCAsmInfoXCOFF.cpp
  MCDXContainerStreamer.cpp
  MCDXContainerWriter.cpp
  MCGOFFStreamer.cpp
  MCSectionDXContainer.cpp
  MCSectionGOFF.cpp
  MCSectionWasm.cpp
  MCSectionXCOFF.cpp
  MCSPIRVStreamer.cpp
  MCSymbolXCOFF.cpp
  MCWasmObjectTargetWriter.cpp
  MCWasmStreamer.cpp
  MCXCOFFObjectTargetWriter.cpp
  MCXCOFFStreamer.cpp
  SPIRVObjectWriter.cpp
  WasmObjectWriter.cpp
  XCOFFObjectWriter.cpp
  )

if(NOT EJIT_BARE_METAL)
  # Normal build: include everything
  list(APPEND MCSOURCES ${MCSOURCES_MACHO_COFF})
  list(APPEND MCSOURCES ${MCSOURCES_BLOAT})
elseif(LLVM_TARGETS_TO_BUILD STREQUAL "AArch64")
  # AArch64 bare-metal: MACHO_COFF needed for MCContext template instantiations;
  # BLOAT needed for section type vtables (GOFF/Wasm/XCOFF/DXContainer)
  list(APPEND MCSOURCES ${MCSOURCES_MACHO_COFF})
  list(APPEND MCSOURCES ${MCSOURCES_BLOAT})
else()
  # x86 bare-metal: need ELF + MachO + COFF
  list(APPEND MCSOURCES ${MCSOURCES_MACHO_COFF})
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
