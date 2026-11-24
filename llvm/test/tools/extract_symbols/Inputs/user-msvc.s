# A TU that iterates the registry: references the accessor (declared-only via LLVM_DECLARE_REGISTRY) and
# carries its own inline copy of begin(), but never touches add_node.
	.text
	.globl	"?HasPlugins@@YA_NXZ"
"?HasPlugins@@YA_NXZ":
	callq	"??$getRegistryLinkListInstance@V?$Registry@VPluginASTAction@clang@@$$V@llvm@@@detail@llvm@@YAAEAU?$RegistryLinkListStorage@V?$Registry@VPluginASTAction@clang@@$$V@llvm@@@01@XZ"
	retq
	.globl	"?begin@?$Registry@VPluginASTAction@clang@@$$V@llvm@@SA?AViterator@12@XZ"
"?begin@?$Registry@VPluginASTAction@clang@@$$V@llvm@@SA?AViterator@12@XZ":
	retq
