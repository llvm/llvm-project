# RUN: llvm-mc %s -o %t.o -filetype=obj -triple=x86_64-pc-linux
# RUN: obj2yaml %t.o | FileCheck %s

# CHECK: Sections:
# CHECK: - Name:            .group{{$}}
# CHECK:   Members:
# CHECK:     - SectionOrType:   .text.foo{{$}}
# CHECK:     - SectionOrType:   .rela.text.foo{{$}}
# CHECK: - Name:            .text.foo{{$}}
# CHECK: - Name:            .rela.text.foo{{$}}
# CHECK:   Info:            .text.foo{{$}}
# CHECK: - Name:            '.group (1)'
# CHECK:   Members:
# CHECK:     - SectionOrType:   '.text.foo (1)'
# CHECK:     - SectionOrType:   '.rela.text.foo (1)'
# CHECK: - Name:            '.text.foo (1)'
# CHECK: - Name:            '.rela.text.foo (1)'
# CHECK:   Info:            '.text.foo (1)'
# CHECK: Symbols:
# CHECK:   Section:         .group{{$}}
# CHECK:   Section:         '.group (1)'


        .section        .text.foo,"axG",@progbits,sym1,comdat
        .quad undef

        .section        .text.foo,"axG",@progbits,sym2,comdat
        .quad undef
