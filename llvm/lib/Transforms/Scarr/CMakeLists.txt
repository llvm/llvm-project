add_llvm_component_library(LLVMMyCFG
        ScarrCpMarker.cpp
        TopoSorter.cpp

        DEPENDS
        intrinsics_gen

        LINK_COMPONENTS
        Core
        Support
        )
