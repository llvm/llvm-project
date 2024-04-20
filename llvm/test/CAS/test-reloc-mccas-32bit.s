# This test tests to make sure mccas can handle scattered relocations properly on 32 bit x86
# REQUIRES: x86-registered-target
# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc --cas=%t/cas --cas-backend --mccas-verify  -triple=i386-apple-macosx10.4 -filetype=obj -o %t/reloc.o %s
        movl    y+4, %ecx
.zerofill __DATA,__common,y,8,3
