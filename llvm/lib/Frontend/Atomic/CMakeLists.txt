add_llvm_component_library(LLVMFrontendAtomic
	Atomic.cpp

	ADDITIONAL_HEADER_DIRS
	${LLVM_MAIN_INCLUDE_DIR}/llvm/Frontend/Atomic

	DEPENDS
	LLVMAnalysis
	LLVMTargetParser

	LINK_COMPONENTS
	Core
	Support
	Analysis
)
