#ifndef POLLY_CODEGENCLEANUP_H
#define POLLY_CODEGENCLEANUP_H

namespace llvm {
class FunctionPass;
class PassRegistry;
}

namespace polly {
llvm::FunctionPass *createCodegenCleanupPass();
}

namespace llvm {
void initializeCodegenCleanupPass(llvm::PassRegistry &);
}

#endif
