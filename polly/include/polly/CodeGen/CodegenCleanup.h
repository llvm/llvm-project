#ifndef POLLY_CODEGENCLEANUP_H
#define POLLY_CODEGENCLEANUP_H

namespace llvm {
class FunctionPass;
class PassRegistry;
} // namespace llvm

namespace polly {
llvm::FunctionPass *createCodegenCleanupPass();
} // namespace polly

namespace llvm {
void initializeCodegenCleanupPass(llvm::PassRegistry &);
} // namespace llvm

#endif
