
#include "llvm/Support/CommandLine.h"
#include <iostream>
#include <vector>

#include "OBSGen.h"

// `main` function translates a C program into a OBS.
int main(int argc, const char **argv) {

  llvm::cl::OptionCategory CodeGenCategory("OBS code generation");
  auto OptionsParser =
      clang::tooling::CommonOptionsParser::create(argc, argv, CodeGenCategory);

  if (!OptionsParser) {
    // Fail gracefully for unsupported options.
    std::cout << "error ----" << std::endl;
    llvm::errs() << OptionsParser.takeError() << "error";
    return 1;
  }

  auto sources = OptionsParser->getSourcePathList();

  clang::tooling::ClangTool Tool(OptionsParser->getCompilations(), sources);

  Tool.run(new mlir::obs::CodeGenFrontendActionFactory());
}
