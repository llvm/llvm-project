-- RUN: %llvmgcc -S -emit-llvm %s -o - | not grep ptrtoint
package Constant_Fold is
  Error : exception;
end;
