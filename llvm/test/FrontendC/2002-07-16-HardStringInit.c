// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null

  char      auto_kibitz_list[100][20] = {
                                      {"diepx"},
                                      {"ferret"},
                                      {"knightc"},
                                      {"knightcap"}};

