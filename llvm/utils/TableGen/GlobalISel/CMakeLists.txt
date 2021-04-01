set(LLVM_LINK_COMPONENTS
  Support
  TableGen
  )

add_llvm_library(LLVMTableGenGlobalISel OBJECT
                 CodeExpander.cpp
                 GIMatchDag.cpp
                 GIMatchDagEdge.cpp
                 GIMatchDagInstr.cpp
                 GIMatchDagOperands.cpp
                 GIMatchDagPredicate.cpp
                 GIMatchDagPredicateDependencyEdge.cpp
                 GIMatchTree.cpp
                )
