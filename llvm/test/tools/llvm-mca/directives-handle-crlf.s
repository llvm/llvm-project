# RUN: llvm-mca -mtriple=x86_64-unknown-unknown %s
# LLVM-MCA-BEGIN foo
addl	$42, %eax
# LLVM-MCA-END
