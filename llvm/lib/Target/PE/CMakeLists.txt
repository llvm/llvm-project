#创建一个PE的LLVM组件，具有JIT即时编译功能
add_llvm_component_group(PE HAS_JIT)

#指定目标架构的定义文件为PE.td，该文件使用TableGen语言描述目标架构的指令集、寄存器、调度模型等核心元数据
set(LLVM_TARGET_DEFINITIONS PE.td)
set(LLVM_EXPERIMENTAL_TARGETS_TO_BUILD "PE")

#通过LLVM的TableGen工具，从PE.td生成C++头文件（.inc），用于后续编译
#[[tablegen(LLVM PEGenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM PEGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM PEGenCallingConv.inc -gen-callingconv)
tablegen(LLVM PEGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM PEGenDisassemblerTables.inc -gen-disassembler)
tablegen(LLVM PEGenFastISel.inc -gen-fast-isel)
tablegen(LLVM PEGenGlobalISel.inc -gen-global-isel)
tablegen(LLVM PEGenPostLegalizeGICombiner.inc -gen-global-isel-combiner
              -combiners="PEPostLegalizerCombiner")
tablegen(LLVM PEGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM PEGenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM PEGenMCPseudoLowering.inc -gen-pseudo-lowering)
tablegen(LLVM PEGenRegisterBank.inc -gen-register-bank)
tablegen(LLVM PEGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM PEGenSubtargetInfo.inc -gen-subtarget)
tablegen(LLVM PEGenExegesis.inc -gen-exegesis)]]

#将生成的TableGen文件（如PEGen*.inc）声明为公共构建目标，确保其他LLVM组件（如汇编器、反汇编器）能访问这些文件
#add_public_tablegen_target(PECommonTableGen)

add_llvm_target(PECodeGen
  PEFrameLowering.cpp
  PEInstrInfo.cpp
  PEISelDAGToDAG.cpp
  PEISelLowering.cpp
  PERegisterInfo.cpp
  PESubtarget.cpp
  PETargetMachine.cpp
  PE.h  # 添加新的头文件

  LINK_COMPONENTS
  Analysis
  AsmPrinter
  CodeGen
  CodeGenTypes
  Core
  GlobalISel
  MC
  PEDesc
  PEInfo
  SelectionDAG
  Support
  Target
  TargetParser
  TransformUtils

  ADD_TO_COMPONENT #添加PE组件
  PE
  )
#递归构建子目录中的模块，通常包括
add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)
