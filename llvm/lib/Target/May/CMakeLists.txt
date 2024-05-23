add_llvm_component_group(May)

add_llvm_target(MayCodeGen
        MayTargetMachine.cpp

        LINK_COMPONENTS
        CodeGen
        Core
        Support
        Target
        MayInfo

        ADD_TO_COMPONENT
        May
    )

add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)