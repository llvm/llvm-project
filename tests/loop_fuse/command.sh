# first command is to emit ir for test case
clang -S -emit-llvm loop_fuse.c -Xclang -disable-O0-optnone

#second command is to clean up ir so that scev can understand it
opt -mem2reg -loop-simplify -instcombine -instnamer -indvars loop_fuse.ll -S -o loop_fuse_out.ll

#Third command will run loopfusion
opt -load  ${LLVM_HOME}/build/lib/LLVMLoopFusion.so -loopfusion loop_fuse_out.ll -enable-new-pm=0 -S -o loop_fuse_out1.ll

#To create cfg
opt -analyze -dot-cfg -enable-new-pm=0 loop_fuse_out1.ll