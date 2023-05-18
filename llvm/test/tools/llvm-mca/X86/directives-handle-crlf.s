# RUN: llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=generic %s
# LLVM-MCA-BEGIN foo
addl	$42, %eax
# LLVM-MCA-END
