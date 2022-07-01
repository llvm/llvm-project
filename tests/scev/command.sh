# first command is to emit ir for test case
clang -S -emit-llvm scev.c -Xclang -disable-O0-optnone

#second command is to clean up ir so that scev can understand it
opt -mem2reg -loop-simplify -instcombine -instnamer -indvars scev.ll -S -o out.ll

#Third command will run scev
opt -load  ${LLVM_HOME}/build/lib/LLVMScev.so -scev out.ll -enable-new-pm=0 -S -o out1.ll