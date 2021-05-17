set(LLVM_LINK_COMPONENTS
  Support
  TableGen
  )

add_llvm_library(LLVMTableGenGlobalISel STATIC DISABLE_LLVM_LINK_LLVM_DYLIB
                 CodeExpander.cpp
                 GIMatchDag.cpp
                 GIMatchDagEdge.cpp
                 GIMatchDagInstr.cpp
                 GIMatchDagOperands.cpp
                 GIMatchDagPredicate.cpp
                 GIMatchDagPredicateDependencyEdge.cpp
                 GIMatchTree.cpp
                )
